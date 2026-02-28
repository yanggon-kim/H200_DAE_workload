#!/usr/bin/env python3
"""Generate additional plots for the verified SpMV H200 report.

Creates:
  1. Gap decomposition stacked bar chart
  2. DAE speedup comparison (BW-based vs CPI-based)
  3. DRAM latency sensitivity visualization
  4. INT32 vs INT64 byte comparison
  5. Physical floor vs actual (waterfall-style)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "..", "analysis")

# ── Data ─────────────────────────────────────────────────────────────────
MATRICES = ["webbase-1M", "cant", "pwtk", "ldoor", "circuit5M", "cage15"]
NNZ =      [3_105_536, 4_007_383, 11_634_424, 46_522_475, 59_524_291, 99_199_551]
ROWS =     [1_000_005, 62_451, 217_918, 952_203, 5_558_326, 5_154_859]
COLS =     [1_000_005, 62_451, 217_918, 952_203, 5_558_326, 5_154_859]
ACTUAL_MS = [0.0411, 0.0408, 0.0663, 0.1964, 0.3211, 0.4636]

PEAK_BW = 4800  # GB/s
HEADROOM_300 = 0.751  # Little's Law ceiling at 300 ns

# Corrected byte counts (y = rows*8)
def bytes_int64(nnz, rows, cols):
    return nnz*4 + nnz*8 + (rows+1)*8 + cols*4 + rows*8

def bytes_int32(nnz, rows, cols):
    return nnz*4 + nnz*4 + (rows+1)*4 + cols*4 + rows*8

BYTES_64 = [bytes_int64(n, r, c) for n, r, c in zip(NNZ, ROWS, COLS)]
BYTES_32 = [bytes_int32(n, r, c) for n, r, c in zip(NNZ, ROWS, COLS)]

FLOOR_MS  = [b / (PEAK_BW * 1e9) * 1000 for b in BYTES_64]
LL_TIME   = [f / HEADROOM_300 for f in FLOOR_MS]
LL_DEFICIT = [lt - f for lt, f in zip(LL_TIME, FLOOR_MS)]
DEP_OTHER  = [a - lt for a, lt in zip(ACTUAL_MS, LL_TIME)]

# CPI-based DAE speedup (from NCU)
CPI_DATA = {
    "cant":   {"cpi": 11.3, "ls_stall": 4.25},
    "ldoor":  {"cpi": 10.3, "ls_stall": 3.41},
    "cage15": {"cpi": 11.0, "ls_stall": 4.23},
}

# Colors
C_FLOOR  = "#2196F3"
C_LL     = "#FF9800"
C_DEP    = "#F44336"
C_INT64  = "#5C6BC0"
C_INT32  = "#66BB6A"
C_DAE_BW = "#AB47BC"
C_DAE_CPI = "#EF5350"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ── Plot 1: Gap Decomposition ────────────────────────────────────────────
def plot_gap_decomposition():
    fig, ax = plt.subplots(figsize=(12, 5.5))

    y = np.arange(len(MATRICES))
    h = 0.55

    bars_floor = ax.barh(y, FLOOR_MS, height=h, color=C_FLOOR, label="Physical Floor")
    bars_ll    = ax.barh(y, LL_DEFICIT, height=h, left=FLOOR_MS, color=C_LL,
                         label="Little's Law Deficit")
    left2 = [f + l for f, l in zip(FLOOR_MS, LL_DEFICIT)]
    bars_dep   = ax.barh(y, DEP_OTHER, height=h, left=left2, color=C_DEP,
                         label="Dep Chain + Other", alpha=0.85)

    # Annotate actual time at bar end
    for i, (m, a) in enumerate(zip(MATRICES, ACTUAL_MS)):
        ax.text(a + 0.005, i, f"{a:.4f} ms", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(MATRICES, fontsize=11)
    ax.set_xlabel("Execution Time (ms)")
    ax.set_title("SpMV Gap Decomposition — H200 (300 ns DRAM latency, 75.1% LL ceiling)")
    ax.legend(loc="lower right", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(ACTUAL_MS) * 1.25)

    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "spmv_gap_decomposition.png"))
    plt.close(fig)
    print("  -> spmv_gap_decomposition.png")


# ── Plot 2: DAE Speedup Comparison ───────────────────────────────────────
def plot_dae_speedup():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # BW-based speedup for all 6
    bw_current = [b / (a * 1e-3) / 1e9 for b, a in zip(BYTES_64, ACTUAL_MS)]
    dae_ceiling_bw = HEADROOM_300 * PEAK_BW  # 3605 GB/s
    bw_speedup = [min(dae_ceiling_bw / bw, 2.0) for bw in bw_current]

    x = np.arange(len(MATRICES))
    w = 0.35

    bars1 = ax.bar(x - w/2, bw_speedup, w, color=C_DAE_BW, label="BW-based (Little's Law)")

    # CPI-based speedup for the 3 profiled matrices
    cpi_speedup = [None] * len(MATRICES)
    for i, m in enumerate(MATRICES):
        if m in CPI_DATA:
            d = CPI_DATA[m]
            cpi_speedup[i] = d["cpi"] / (d["cpi"] - d["ls_stall"])

    cpi_vals = [v if v else 0 for v in cpi_speedup]
    cpi_colors = [C_DAE_CPI if v else "#EEEEEE" for v in cpi_speedup]
    bars2 = ax.bar(x + w/2, cpi_vals, w, color=cpi_colors,
                   label="CPI-based (NCU stall removal)", edgecolor="#999", linewidth=0.5)

    # Annotate
    for i, v in enumerate(bw_speedup):
        ax.text(i - w/2, v + 0.03, f"{v:.2f}x", ha="center", fontsize=8.5, fontweight="bold")
    for i, v in enumerate(cpi_speedup):
        if v:
            ax.text(i + w/2, v + 0.03, f"{v:.2f}x", ha="center", fontsize=8.5, fontweight="bold",
                    color=C_DAE_CPI)

    ax.axhline(1.0, color="#999", ls="--", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(MATRICES, fontsize=10)
    ax.set_ylabel("Predicted Speedup with DAE")
    ax.set_title("DAE Benefit Prediction — H200 SpMV\n(BW-based vs CPI-based analysis)")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 2.3)

    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "spmv_dae_speedup.png"))
    plt.close(fig)
    print("  -> spmv_dae_speedup.png")


# ── Plot 3: DRAM Latency Sensitivity ─────────────────────────────────────
def plot_latency_sensitivity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    latencies = np.arange(150, 501, 10)
    max_warps = 64
    sm_count = 132
    cache_line = 128

    warps_req = [(PEAK_BW * 1e9 * lat * 1e-9) / cache_line / sm_count for lat in latencies]
    ll_ceiling = [min(max_warps / wr, 1.0) * 100 for wr in warps_req]
    dep_ceiling = [min(max_warps / (wr * 2), 1.0) * 100 for wr in warps_req]

    # Left: Ceiling vs latency
    ax1.plot(latencies, ll_ceiling, color=C_FLOOR, lw=2.5, label="Little's Law Ceiling")
    ax1.plot(latencies, dep_ceiling, color=C_DEP, lw=2.5, label="Dep-Chain Ceiling", ls="--")
    ax1.axhline(60.8, color="#4CAF50", ls=":", lw=1.5, label="Best measured (ldoor 60.8%)")
    ax1.axvline(300, color="#999", ls=":", lw=1, alpha=0.7)
    ax1.text(305, 95, "300 ns\n(primary)", fontsize=8, color="#666")

    # Shade plausible range
    ax1.axvspan(250, 400, alpha=0.08, color="orange", label="Plausible range")

    ax1.set_xlabel("DRAM Latency (ns)")
    ax1.set_ylabel("% Peak Bandwidth Ceiling")
    ax1.set_title("BW Ceiling vs DRAM Latency — H200")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, 110)
    ax1.set_xlim(150, 500)
    ax1.grid(True, alpha=0.3)

    # Right: DAE benefit vs latency
    dae_ceiling = [100.0] * len(latencies)  # DAE with FIFO>=2 saturates BW
    dae_speedup_ldoor = [min(100.0 / max(dep, 1), 5.0) for dep in dep_ceiling]
    # More realistic: speedup = LL_ceiling / current_pct for cage15
    cage15_pct = 57.2
    dae_speedup_realistic = [min(ll / cage15_pct, 3.0) for ll in ll_ceiling]

    ax2.plot(latencies, dae_speedup_realistic, color=C_DAE_BW, lw=2.5,
             label="DAE BW speedup (cage15)")
    ax2.axhline(1.6, color=C_DAE_CPI, ls="--", lw=2, label="CPI-based speedup (1.6x)")
    ax2.axhline(1.0, color="#999", ls=":", lw=0.8)
    ax2.axvline(300, color="#999", ls=":", lw=1, alpha=0.7)

    ax2.set_xlabel("DRAM Latency (ns)")
    ax2.set_ylabel("DAE Speedup (x)")
    ax2.set_title("DAE Benefit vs DRAM Latency — H200")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.8, 2.5)
    ax2.set_xlim(150, 500)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "spmv_latency_sensitivity.png"))
    plt.close(fig)
    print("  -> spmv_latency_sensitivity.png")


# ── Plot 4: INT32 vs INT64 Byte Comparison ───────────────────────────────
def plot_int32_vs_int64():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(MATRICES))
    w = 0.35

    b64_mb = [b / 1e6 for b in BYTES_64]
    b32_mb = [b / 1e6 for b in BYTES_32]
    waste_pct = [(1 - b32/b64) * 100 for b32, b64 in zip(BYTES_32, BYTES_64)]

    # Left: stacked comparison
    ax1.bar(x - w/2, b64_mb, w, color=C_INT64, label="INT64 (current)")
    ax1.bar(x + w/2, b32_mb, w, color=C_INT32, label="INT32 (ideal)")

    for i in range(len(MATRICES)):
        ax1.text(i, max(b64_mb[i], b32_mb[i]) + 15, f"-{waste_pct[i]:.0f}%",
                 ha="center", fontsize=9, color=C_DEP, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(MATRICES, fontsize=9, rotation=15, ha="right")
    ax1.set_ylabel("Bytes Moved (MB)")
    ax1.set_title("Data Movement: INT64 vs INT32 Indices")
    ax1.legend(fontsize=10)

    # Right: Arithmetic intensity comparison
    ai_64 = [2*n / b for n, b in zip(NNZ, BYTES_64)]
    ai_32 = [2*n / b for n, b in zip(NNZ, BYTES_32)]

    ax2.bar(x - w/2, ai_64, w, color=C_INT64, label="INT64 (current)")
    ax2.bar(x + w/2, ai_32, w, color=C_INT32, label="INT32 (ideal)")

    for i in range(len(MATRICES)):
        ax2.text(i - w/2, ai_64[i] + 0.003, f"{ai_64[i]:.3f}", ha="center", fontsize=7.5)
        ax2.text(i + w/2, ai_32[i] + 0.003, f"{ai_32[i]:.3f}", ha="center", fontsize=7.5)

    ax2.axhline(13.9, color="red", ls="--", lw=1.5, alpha=0.5)
    ax2.text(len(MATRICES)-0.5, 0.30, "Ridge point: 13.9\n(far above)", fontsize=8,
             color="red", alpha=0.7, ha="right")

    ax2.set_xticks(x)
    ax2.set_xticklabels(MATRICES, fontsize=9, rotation=15, ha="right")
    ax2.set_ylabel("Arithmetic Intensity (FLOP/byte)")
    ax2.set_title("Arithmetic Intensity: INT64 vs INT32")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 0.32)

    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "spmv_int32_vs_int64.png"))
    plt.close(fig)
    print("  -> spmv_int32_vs_int64.png")


# ── Plot 5: Floor vs Actual (visual gap) ─────────────────────────────────
def plot_floor_vs_actual():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(MATRICES))
    w = 0.35

    ax.bar(x - w/2, FLOOR_MS, w, color=C_FLOOR, label="Physical Floor", edgecolor="white")
    ax.bar(x + w/2, ACTUAL_MS, w, color=C_DEP, label="Measured", alpha=0.85, edgecolor="white")

    # Annotate gap ratios
    for i in range(len(MATRICES)):
        gap = ACTUAL_MS[i] / FLOOR_MS[i]
        mid_y = max(ACTUAL_MS[i], FLOOR_MS[i])
        ax.text(i, mid_y + 0.008, f"{gap:.1f}x gap", ha="center", fontsize=9,
                fontweight="bold", color="#333")

    # Annotate % peak BW on actual bars
    for i in range(len(MATRICES)):
        pct = BYTES_64[i] / (ACTUAL_MS[i] * 1e-3) / (PEAK_BW * 1e9) * 100
        ax.text(i + w/2, ACTUAL_MS[i] / 2, f"{pct:.0f}%\npeak", ha="center",
                fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({n/1e6:.0f}M NNZ)" for m, n in zip(MATRICES, NNZ)], fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Physical Floor vs Measured Execution Time — H200 SpMV")
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "spmv_floor_vs_actual.png"))
    plt.close(fig)
    print("  -> spmv_floor_vs_actual.png")


if __name__ == "__main__":
    print("Generating verification plots...")
    plot_gap_decomposition()
    plot_dae_speedup()
    plot_latency_sensitivity()
    plot_int32_vs_int64()
    plot_floor_vs_actual()
    print("Done.")
