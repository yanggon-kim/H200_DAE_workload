#!/usr/bin/env python3
"""
SpMV Profiling on NVIDIA H200 using SuiteSparse matrices.

Profiles CSR SpMV via torch.sparse.mm (cuSPARSE backend) across 6 matrices
from the SuiteSparse Matrix Collection, with CUDA event timing, NVTX markers,
and comprehensive bandwidth/throughput analysis.

Usage:
  python scripts/profile_spmv.py --mode sweep
  python scripts/profile_spmv.py --mode single --matrix cage15
  python scripts/profile_spmv.py --mode nsys-prep --matrix cage15
"""

import argparse
import gc
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse
import torch

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analysis"
TRACE_DIR = PROJECT_ROOT / "traces"
PROFILE_DIR = PROJECT_ROOT / "profiles"
MATRIX_DIR = PROJECT_ROOT / "models" / "suitesparse"

for d in [OUTPUT_DIR, TRACE_DIR, PROFILE_DIR, MATRIX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# SuiteSparse matrices — same 6 as B200 report
MATRICES = {
    "webbase-1M": {"group": "Williams",    "name": "webbase-1M"},
    "cant":       {"group": "Williams",    "name": "cant"},
    "pwtk":       {"group": "Boeing",      "name": "pwtk"},
    "ldoor":      {"group": "GHS_psdef",   "name": "ldoor"},
    "circuit5M":  {"group": "Freescale",   "name": "circuit5M"},
    "cage15":     {"group": "vanHeukelum", "name": "cage15"},
}

NUM_WARMUP = 5
NUM_MEASURED = 20


# ============================================================================
# Utilities
# ============================================================================

class CUDATimer:
    """GPU-synchronized timer using CUDA events."""
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def download_matrix(matrix_key):
    """Download a SuiteSparse matrix if not already cached, using ssgetpy."""
    mat_path = MATRIX_DIR / f"{matrix_key}.mtx"

    if mat_path.exists():
        print(f"  Matrix {matrix_key} already cached at {mat_path}")
        return mat_path

    import ssgetpy
    print(f"  Downloading {matrix_key} via ssgetpy...")
    results = ssgetpy.search(name=matrix_key)
    for r in results:
        if r.name == matrix_key:
            r.download(format='MM', destpath=str(MATRIX_DIR))
            # ssgetpy downloads to MATRIX_DIR/group/name/name.mtx
            expected = MATRIX_DIR / r.group / matrix_key / f"{matrix_key}.mtx"
            if expected.exists():
                expected.rename(mat_path)
                try:
                    (MATRIX_DIR / r.group / matrix_key).rmdir()
                    (MATRIX_DIR / r.group).rmdir()
                except OSError:
                    pass
            else:
                # search for the .mtx file
                for p in MATRIX_DIR.rglob("*.mtx"):
                    if matrix_key in p.name:
                        p.rename(mat_path)
                        break
            print(f"  Saved to {mat_path}")
            return mat_path

    raise RuntimeError(f"Matrix {matrix_key} not found in SuiteSparse")


def load_sparse_matrix(matrix_key):
    """Load a SuiteSparse matrix as a torch sparse CSR tensor on GPU."""
    mat_path = download_matrix(matrix_key)

    print(f"  Loading {matrix_key}...")
    mat = scipy.io.mmread(str(mat_path))
    csr = scipy.sparse.csr_matrix(mat, dtype=np.float32)

    rows, cols = csr.shape
    nnz = csr.nnz
    avg_nnz_per_row = nnz / rows if rows > 0 else 0

    print(f"  Shape: {rows} x {cols}, NNZ: {nnz:,}, Avg NNZ/row: {avg_nnz_per_row:.1f}")

    # Convert to torch sparse CSR on GPU
    crow_indices = torch.from_numpy(csr.indptr.astype(np.int64)).cuda()
    col_indices = torch.from_numpy(csr.indices.astype(np.int64)).cuda()
    values = torch.from_numpy(csr.data.astype(np.float32)).cuda()

    sparse_tensor = torch.sparse_csr_tensor(
        crow_indices, col_indices, values,
        size=(rows, cols), dtype=torch.float32, device="cuda"
    )

    return sparse_tensor, {"rows": rows, "cols": cols, "nnz": nnz,
                           "avg_nnz_per_row": avg_nnz_per_row}


def compute_bytes_moved(info):
    """Compute total bytes moved for one SpMV: y = A * x.
    CSR arrays: values (4B per nnz), col_indices (8B per nnz), row_ptr (8B per (rows+1))
    Vectors: x (4B per cols), y (8B per rows — read + write for accumulation)

    Note: The merge-based csrmv_v3_kernel reads y before accumulating (y[i] += ...),
    so y is both read and written, costing 8 bytes per row rather than 4.
    """
    nnz = info["nnz"]
    rows = info["rows"]
    cols = info["cols"]
    # values: float32 = 4B each
    # col_indices: int64 = 8B each (torch uses int64 for indices)
    # row_ptr: int64 = 8B each (rows+1 entries)
    # x vector: float32 = 4B each (cols entries)
    # y vector: float32 = 4B read + 4B write = 8B each (merge-based accumulation)
    bytes_moved = (
        nnz * 4 +           # values
        nnz * 8 +           # col_indices (int64)
        (rows + 1) * 8 +    # crow_indices (int64)
        cols * 4 +           # x vector
        rows * 8             # y vector (read + write for accumulation)
    )
    return bytes_moved


def print_system_info():
    print(f"{'=' * 80}")
    print("SYSTEM INFO")
    print(f"{'=' * 80}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}")
    print(f"  GPU Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"  SMs: {props.multi_processor_count}")
    print(f"  Compute Capability: {props.major}.{props.minor}")


# ============================================================================
# SpMV Profiling
# ============================================================================

def profile_spmv_single(sparse_tensor, x, info, matrix_name,
                         num_warmup=NUM_WARMUP, num_measured=NUM_MEASURED,
                         use_nvtx=False):
    """Profile SpMV for a single matrix."""
    bytes_moved = compute_bytes_moved(info)
    flops = 2 * info["nnz"]  # FMA: multiply + add per nonzero
    ai = flops / bytes_moved

    # Warmup
    for _ in range(num_warmup):
        y = torch.sparse.mm(sparse_tensor, x)
    torch.cuda.synchronize()

    # Measured iterations
    timer = CUDATimer()
    times_ms = []

    for i in range(num_measured):
        if use_nvtx:
            torch.cuda.nvtx.range_push(f"spmv_{matrix_name}")

        timer.start()
        y = torch.sparse.mm(sparse_tensor, x)
        elapsed = timer.stop()

        if use_nvtx:
            torch.cuda.nvtx.range_pop()

        times_ms.append(elapsed)

    times_ms = np.array(times_ms)
    avg_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    med_ms = float(np.median(times_ms))

    # Compute effective bandwidth
    eff_bw = bytes_moved / (avg_ms * 1e-3) / 1e9  # GB/s
    gflops = flops / (avg_ms * 1e-3) / 1e9

    return {
        "matrix": matrix_name,
        "rows": info["rows"],
        "cols": info["cols"],
        "nnz": info["nnz"],
        "avg_nnz_per_row": info["avg_nnz_per_row"],
        "bytes_moved": bytes_moved,
        "flops": flops,
        "arithmetic_intensity": ai,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "med_ms": med_ms,
        "cov_pct": 100 * std_ms / avg_ms if avg_ms > 0 else 0,
        "eff_bw_gbs": eff_bw,
        "gflops": gflops,
        "times_ms": times_ms.tolist(),
    }


def run_sweep(matrices_to_profile=None, use_nvtx=False):
    """Profile SpMV across all matrices."""
    print_system_info()

    if matrices_to_profile is None:
        matrices_to_profile = list(MATRICES.keys())

    results = []

    for matrix_name in matrices_to_profile:
        print(f"\n{'=' * 60}")
        print(f"Matrix: {matrix_name}")
        print(f"{'=' * 60}")

        try:
            sparse_tensor, info = load_sparse_matrix(matrix_name)
            # Create random x vector
            x = torch.randn(info["cols"], 1, device="cuda", dtype=torch.float32)

            result = profile_spmv_single(
                sparse_tensor, x, info, matrix_name, use_nvtx=use_nvtx
            )
            results.append(result)

            print(f"  Time:  {result['avg_ms']:.4f} ms (std: {result['std_ms']:.4f}, CoV: {result['cov_pct']:.1f}%)")
            print(f"  BW:    {result['eff_bw_gbs']:.1f} GB/s")
            print(f"  GFLOPS: {result['gflops']:.1f}")
            print(f"  AI:    {result['arithmetic_intensity']:.3f} FLOP/byte")

            # Cleanup
            del sparse_tensor, x
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results


def run_nsys_prep(matrix_name, num_iters=5):
    """Run SpMV with NVTX markers for nsys profiling."""
    print_system_info()
    print(f"\nPreparing nsys run for {matrix_name} ({num_iters} iterations)...")

    sparse_tensor, info = load_sparse_matrix(matrix_name)
    x = torch.randn(info["cols"], 1, device="cuda", dtype=torch.float32)

    # Warmup (no markers)
    for _ in range(NUM_WARMUP):
        y = torch.sparse.mm(sparse_tensor, x)
    torch.cuda.synchronize()

    # Profiled iterations with NVTX
    for i in range(num_iters):
        torch.cuda.nvtx.range_push(f"spmv_iter_{i}")
        y = torch.sparse.mm(sparse_tensor, x)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    print("Done. Run with: nsys profile -o profiles/spmv_nsys --trace=cuda,nvtx -f true python scripts/profile_spmv.py --mode nsys-prep --matrix <name>")


def run_ncu_prep(matrix_name):
    """Run SpMV with cudaProfiler markers for ncu profiling."""
    print_system_info()
    print(f"\nPreparing ncu run for {matrix_name}...")

    sparse_tensor, info = load_sparse_matrix(matrix_name)
    x = torch.randn(info["cols"], 1, device="cuda", dtype=torch.float32)

    # Warmup (outside profiler range)
    for _ in range(NUM_WARMUP):
        y = torch.sparse.mm(sparse_tensor, x)
    torch.cuda.synchronize()

    # Single profiled iteration
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"spmv_{matrix_name}")
    y = torch.sparse.mm(sparse_tensor, x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"Done. NCU will capture the profiled iteration.")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SpMV Profiling on H200")
    parser.add_argument("--mode", default="sweep",
                        choices=["sweep", "single", "nsys-prep", "ncu-prep"],
                        help="Profiling mode")
    parser.add_argument("--matrix", default="cage15",
                        help="Matrix name for single/nsys/ncu modes")
    parser.add_argument("--use-nvtx", action="store_true",
                        help="Enable NVTX markers in sweep mode")
    args = parser.parse_args()

    if args.mode == "sweep":
        results = run_sweep(use_nvtx=args.use_nvtx)

        # Save results
        output_path = OUTPUT_DIR / "spmv_profiling_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        # Print summary table
        print(f"\n{'=' * 100}")
        print(f"{'Matrix':<12} {'NNZ':>10} {'Time (ms)':>10} {'Eff. BW':>10} {'GFLOPS':>8} {'AI':>6} {'CoV%':>6}")
        print(f"{'=' * 100}")
        for r in results:
            print(f"  {r['matrix']:<10} {r['nnz']:>10,} {r['avg_ms']:>10.4f} "
                  f"{r['eff_bw_gbs']:>9.1f} {r['gflops']:>8.1f} {r['arithmetic_intensity']:>6.3f} "
                  f"{r['cov_pct']:>5.1f}%")

    elif args.mode == "single":
        results = run_sweep(matrices_to_profile=[args.matrix])
        output_path = OUTPUT_DIR / f"spmv_{args.matrix}_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    elif args.mode == "nsys-prep":
        run_nsys_prep(args.matrix)

    elif args.mode == "ncu-prep":
        run_ncu_prep(args.matrix)


if __name__ == "__main__":
    main()
