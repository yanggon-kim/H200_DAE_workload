#!/usr/bin/env python3
"""
Generate all SpMV profiling visualization plots for H200 GPU.

Produces 5 plots in analysis/:
  1. spmv_roofline_h200.png       - Roofline model
  2. spmv_bw_efficiency.png       - Bandwidth efficiency bar chart
  3. spmv_stall_breakdown_cage15.png - Warp stall pie chart (cage15)
  4. spmv_nnz_vs_bw.png           - NNZ vs bandwidth scatter
  5. spmv_stall_comparison.png    - Comparative stall stacked bars
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# H200 GPU specifications
# ---------------------------------------------------------------------------
PEAK_FP32_TFLOPS = 66.9        # non-tensor FP32
PEAK_BW_GBS = 4800             # HBM3e
RIDGE_POINT = PEAK_FP32_TFLOPS * 1e3 / PEAK_BW_GBS  # ~13.9 FLOP/byte

# ---------------------------------------------------------------------------
# Profiling data
# ---------------------------------------------------------------------------
MATRICES = ["webbase-1M", "cant", "pwtk", "ldoor", "circuit5M", "cage15"]
NNZ       = [3_105_536, 4_007_383, 11_634_424, 46_522_475, 59_524_291, 99_199_551]
TIME_MS   = [0.0411, 0.0408, 0.0663, 0.1964, 0.3211, 0.4636]
EFF_BW    = [1296.4, 1204.6, 2157.6, 2920.3, 2501.4, 2745.4]
GFLOPS    = [151.2, 196.7, 350.8, 473.8, 370.7, 427.9]
AI        = [0.117, 0.163, 0.163, 0.162, 0.148, 0.156]

# Per-matrix colors (consistent across plots)
COLORS = ["#e74c3c", "#e67e22", "#27ae60", "#3498db", "#9b59b6", "#2c3e50"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Stall breakdown data (percentages)
STALL_CATEGORIES = [
    "long_scoreboard", "wait", "short_scoreboard",
    "not_selected", "selected", "math_pipe_throttle", "other",
]
STALL_DATA = {
    "cant":   [37.6, 16.1, 10.3, 12.2, 8.9, 7.4, 7.5],
    "ldoor":  [33.0, 17.8, 11.0, 14.1, 9.7, 8.4, 6.0],
    "cage15": [38.4, 17.0, 12.2, 11.1, 9.1, 6.5, 5.7],
}

# Pretty labels for stall categories
STALL_LABELS = {
    "long_scoreboard":    "Long Scoreboard",
    "wait":               "Wait",
    "short_scoreboard":   "Short Scoreboard",
    "not_selected":       "Not Selected",
    "selected":           "Selected",
    "math_pipe_throttle": "Math Pipe Throttle",
    "other":              "Other",
}

STALL_COLORS = ["#e74c3c", "#3498db", "#e67e22", "#9b59b6", "#2ecc71", "#f1c40f", "#95a5a6"]


def _apply_style(ax):
    """Apply common professional styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


# ===== PLOT 1: Roofline =====
def plot_roofline():
    peak_gflops = PEAK_FP32_TFLOPS * 1e3  # 66900 GFLOP/s

    oi_range = np.logspace(-2, 3, 600)
    roofline = np.minimum(peak_gflops, PEAK_BW_GBS * oi_range)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Roofline curve
    ax.loglog(oi_range, roofline, "k-", linewidth=2.5, label="Roofline", zorder=2)

    # Ridge point
    ax.axvline(x=RIDGE_POINT, color="gray", linestyle=":", alpha=0.6,
               label=f"Ridge point ({RIDGE_POINT:.1f} FLOP/byte)")

    # Peak compute reference
    ax.axhline(y=peak_gflops, color="gray", linestyle="--", alpha=0.3)
    ax.text(300, peak_gflops * 1.12, f"Peak FP32: {PEAK_FP32_TFLOPS} TFLOPS",
            fontsize=9, color="gray")

    # Region shading
    ax.fill_between(oi_range, 0.01, roofline, where=(oi_range < RIDGE_POINT),
                    alpha=0.07, color="blue")
    ax.fill_between(oi_range, 0.01, roofline, where=(oi_range >= RIDGE_POINT),
                    alpha=0.07, color="red")
    ax.text(RIDGE_POINT * 0.015, peak_gflops * 0.25, "Memory\nBound",
            fontsize=15, color="blue", alpha=0.5, fontweight="bold")
    ax.text(RIDGE_POINT * 5, peak_gflops * 0.25, "Compute\nBound",
            fontsize=15, color="red", alpha=0.5, fontweight="bold")

    # Data points
    for i, name in enumerate(MATRICES):
        bw_eff_pct = EFF_BW[i] / PEAK_BW_GBS * 100
        ax.scatter(AI[i], GFLOPS[i], marker=MARKERS[i], s=220, c=COLORS[i],
                   edgecolors="black", linewidth=1.5, zorder=5,
                   label=f"{name} (BW {bw_eff_pct:.0f}%)")

        # Annotate with name + bandwidth
        ax.annotate(f"{name}\n{EFF_BW[i]:.0f} GB/s",
                    (AI[i], GFLOPS[i]),
                    textcoords="offset points", xytext=(12, 8), fontsize=8,
                    fontweight="bold", color=COLORS[i],
                    arrowprops=dict(arrowstyle="-", color=COLORS[i], alpha=0.5))

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Throughput (GFLOP/s)", fontsize=12)
    ax.set_title("SpMV Roofline Analysis -- NVIDIA H200 (FP32, Non-Tensor)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(10, peak_gflops * 1.5)

    # GPU info box
    textstr = (f"GPU: NVIDIA H200 SXM\n"
               f"Peak FP32: {PEAK_FP32_TFLOPS} TFLOPS\n"
               f"Peak BW: {PEAK_BW_GBS} GB/s (HBM3e)\n"
               f"Ridge: {RIDGE_POINT:.1f} FLOP/byte")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    plt.tight_layout()
    out = OUTPUT_DIR / "spmv_roofline_h200.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[1/5] Roofline saved: {out}")


# ===== PLOT 2: Bandwidth Efficiency Bar Chart =====
def plot_bw_efficiency():
    bw_pct = [bw / PEAK_BW_GBS * 100 for bw in EFF_BW]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by bandwidth for a cleaner look
    order = np.argsort(EFF_BW)
    sorted_names = [MATRICES[i] for i in order]
    sorted_bw = [EFF_BW[i] for i in order]
    sorted_pct = [bw_pct[i] for i in order]

    # Color by efficiency tier
    bar_colors = []
    for pct in sorted_pct:
        if pct > 50:
            bar_colors.append("#27ae60")   # green
        elif pct > 25:
            bar_colors.append("#f39c12")   # yellow/amber
        else:
            bar_colors.append("#e74c3c")   # red

    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_bw, color=bar_colors, edgecolor="black",
                   linewidth=0.8, height=0.6)

    # Peak BW reference line
    ax.axvline(x=PEAK_BW_GBS, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Peak BW ({PEAK_BW_GBS} GB/s)")

    # Percentage labels on bars
    for j, (bar, pct, bw) in enumerate(zip(bars, sorted_pct, sorted_bw)):
        ax.text(bw + 50, bar.get_y() + bar.get_height() / 2,
                f"  {bw:.0f} GB/s ({pct:.1f}%)",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel("Effective Bandwidth (GB/s)", fontsize=12)
    ax.set_title("SpMV Effective Bandwidth -- H200 (4800 GB/s Peak HBM3e)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, PEAK_BW_GBS * 1.12)
    ax.legend(fontsize=10, loc="lower right")
    _apply_style(ax)
    ax.grid(axis="x", alpha=0.3)

    # Color legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#27ae60", edgecolor="black", label="> 50% efficiency"),
        Patch(facecolor="#f39c12", edgecolor="black", label="25-50% efficiency"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="< 25% efficiency"),
    ]
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color="#e74c3c",
              linestyle="--", linewidth=1.5, label=f"Peak BW ({PEAK_BW_GBS} GB/s)")],
              fontsize=9, loc="lower right")

    plt.tight_layout()
    out = OUTPUT_DIR / "spmv_bw_efficiency.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[2/5] BW efficiency saved: {out}")


# ===== PLOT 3: Warp Stall Pie Chart (cage15) =====
def plot_stall_pie_cage15():
    # cage15 stall data from NCU
    labels_raw = ["long_scoreboard", "wait", "short_scoreboard",
                  "not_selected", "selected", "math_pipe_throttle"]
    values = [4.23, 1.87, 1.35, 1.22, 1.00, 0.72]
    pct = [38.4, 17.0, 12.2, 11.1, 9.1, 6.5]

    # Remaining goes to "Other"
    other_pct = 100.0 - sum(pct)
    labels = [STALL_LABELS[k] for k in labels_raw] + ["Other"]
    sizes = pct + [other_pct]
    colors = STALL_COLORS

    fig, ax = plt.subplots(figsize=(9, 7))

    explode = [0.04] * len(sizes)
    explode[0] = 0.08  # emphasize the dominant stall

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=140,
        colors=colors, explode=explode, pctdistance=0.8,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=10))

    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax.set_title("Warp Stall Breakdown -- cage15 (99.2M NNZ)\nNCU Profiling on H200",
                 fontsize=13, fontweight="bold", pad=20)

    # Add warps/cycle info as a text box
    info = ("Total active warps: ~11.0/cycle\n"
            "Dominant bottleneck: Long Scoreboard\n"
            "(memory-latency bound)")
    props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9)
    ax.text(-1.35, -1.15, info, fontsize=9, bbox=props)

    plt.tight_layout()
    out = OUTPUT_DIR / "spmv_stall_breakdown_cage15.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[3/5] Stall pie chart saved: {out}")


# ===== PLOT 4: NNZ vs Bandwidth Scatter =====
def plot_nnz_vs_bw():
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(MATRICES):
        ax.scatter(NNZ[i], EFF_BW[i], marker=MARKERS[i], s=200, c=COLORS[i],
                   edgecolors="black", linewidth=1.5, zorder=5, label=name)
        ax.annotate(name, (NNZ[i], EFF_BW[i]),
                    textcoords="offset points", xytext=(10, 8), fontsize=9,
                    fontweight="bold", color=COLORS[i])

    # Trend line (log fit)
    log_nnz = np.log10(NNZ)
    coeffs = np.polyfit(log_nnz, EFF_BW, 2)
    x_fit = np.logspace(np.log10(min(NNZ) * 0.8), np.log10(max(NNZ) * 1.2), 100)
    y_fit = np.polyval(coeffs, np.log10(x_fit))
    ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.6, linewidth=1.5,
            label="Quadratic trend (log NNZ)")

    # Peak BW reference
    ax.axhline(y=PEAK_BW_GBS, color="#e74c3c", linestyle="--", alpha=0.4,
               linewidth=1.2, label=f"Peak BW ({PEAK_BW_GBS} GB/s)")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Non-Zeros (NNZ)", fontsize=12)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=12)
    ax.set_title("SpMV: NNZ vs Effective Bandwidth -- H200",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    _apply_style(ax)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.set_ylim(800, PEAK_BW_GBS * 0.85)

    plt.tight_layout()
    out = OUTPUT_DIR / "spmv_nnz_vs_bw.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[4/5] NNZ vs BW scatter saved: {out}")


# ===== PLOT 5: Comparative Stall Stacked Bar =====
def plot_stall_comparison():
    matrix_names = ["cant", "ldoor", "cage15"]
    categories = [STALL_LABELS[c] for c in STALL_CATEGORIES]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(matrix_names))
    bar_width = 0.5

    # Build stacked bars
    bottoms = np.zeros(len(matrix_names))
    for j, cat_key in enumerate(STALL_CATEGORIES):
        values = [STALL_DATA[m][j] for m in matrix_names]
        bars = ax.bar(x, values, bar_width, bottom=bottoms,
                      color=STALL_COLORS[j], edgecolor="white", linewidth=0.8,
                      label=STALL_LABELS[cat_key])

        # Add percentage labels inside bars if segment is large enough
        for k, (v, b) in enumerate(zip(values, bottoms)):
            if v >= 6:  # only label segments >= 6%
                ax.text(x[k], b + v / 2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white")

        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(matrix_names, fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of Warp Cycles (%)", fontsize=12)
    ax.set_title("Warp Stall Comparison Across Matrices -- H200 NCU Profiling",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.28, 1.0))
    _apply_style(ax)
    ax.grid(axis="y", alpha=0.3)

    # Add NNZ annotations below matrix names
    nnz_map = {"cant": "4.0M NNZ", "ldoor": "46.5M NNZ", "cage15": "99.2M NNZ"}
    for k, m in enumerate(matrix_names):
        ax.text(x[k], -4, nnz_map[m], ha="center", va="top", fontsize=9, color="gray")

    plt.tight_layout()
    out = OUTPUT_DIR / "spmv_stall_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[5/5] Stall comparison saved: {out}")


# ===== Main =====
if __name__ == "__main__":
    print("Generating SpMV profiling plots for H200 GPU...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    plot_roofline()
    plot_bw_efficiency()
    plot_stall_pie_cage15()
    plot_nnz_vs_bw()
    plot_stall_comparison()

    print("\nAll 5 plots generated successfully.")
