"""Generate the four schematic figures used by lecture_summary.md.

Run from the repo root with the project's venv:

    .venv/bin/python raw/figures/generate.py

Outputs PNGs alongside this script in raw/figures/.

Figures generated:
    roofline.png             — H100 FP32 roofline with example kernel positions
    ai_scaling.png           — AI vs problem size N for addition (flat) vs matmul (linear)
    memory_pyramid.png       — GPU memory hierarchy with bandwidth + size per tier
    reuse_diagram.png        — why addition has low AI and matmul has high AI
    fp32_vs_bf16_roofline.png — same H100, FP32 vs BF16 ceilings; transformer kernels positioned
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

# H100 FP32 (no tensor cores) — the numbers HW1 actually uses.
PEAK_COMPUTE = 67e12      # 67 TFLOP/s
PEAK_BW = 3.35e12         # 3.35 TB/s
RIDGE_AI = PEAK_COMPUTE / PEAK_BW  # 20 FLOP/byte


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: the headline roofline plot
# ──────────────────────────────────────────────────────────────────────────────

def roofline():
    fig, ax = plt.subplots(figsize=(10, 6))

    ai_range = np.logspace(-2, 4, 500)
    mem_ceiling = PEAK_BW * ai_range
    compute_ceiling = np.full_like(ai_range, PEAK_COMPUTE)
    roof = np.minimum(mem_ceiling, compute_ceiling)

    # Region shading
    ax.axvspan(1e-2, RIDGE_AI, alpha=0.06, color="tab:blue")
    ax.axvspan(RIDGE_AI, 1e4, alpha=0.06, color="tab:red")

    # Ceilings
    ax.loglog(ai_range, roof, "k-", linewidth=2.5, label="Roofline (the achievable max)")
    ax.loglog(ai_range, mem_ceiling, "b--", linewidth=1, alpha=0.55,
              label=f"Memory ceiling = bandwidth × AI  ({PEAK_BW/1e12:.2f} TB/s slope)")
    ax.loglog(ai_range, compute_ceiling, "r--", linewidth=1, alpha=0.55,
              label=f"Compute ceiling = peak FLOP/s  ({PEAK_COMPUTE/1e12:.0f} TFLOP/s)")

    # Ridge point
    ax.plot([RIDGE_AI], [PEAK_COMPUTE], "ko", markersize=11, zorder=5)
    ax.annotate(f"Ridge point\nAI* = {RIDGE_AI:.0f} FLOP/byte\n(peak_compute / bandwidth)",
                xy=(RIDGE_AI, PEAK_COMPUTE), xytext=(RIDGE_AI * 4, PEAK_COMPUTE * 0.18),
                fontsize=10, ha="left",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Example kernels positioned on the achievable max
    examples = [
        (0.083, "c = a + b\n(FP32, 1M elements)", "#555555"),
        (5,     "small matmul (N≈15)",            "#1f77b4"),
        (170,   "matmul N=1024 (FP32)",           "#2ca02c"),
        (683,   "matmul N=4096 (FP32)",           "#d62728"),
    ]
    for ai, label, color in examples:
        perf = min(PEAK_COMPUTE, PEAK_BW * ai)
        ax.plot([ai], [perf], "o", color=color, markersize=11, zorder=4,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(label, xy=(ai, perf), xytext=(10, -3),
                    textcoords="offset points", fontsize=9, color=color)

    # Region labels
    ax.text(0.25, 2.5e9, "memory-bound region\n(on slanted ceiling)",
            color="tab:blue", fontsize=10, ha="center", alpha=0.85)
    ax.text(1500, 2.5e9, "compute-bound region\n(on flat ceiling)",
            color="tab:red", fontsize=10, ha="center", alpha=0.85)

    ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Achievable performance  (FLOP/s)", fontsize=12)
    ax.set_title("Roofline model — NVIDIA H100, FP32 (no tensor cores)",
                 fontsize=13)
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e9, 5e14)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUT / "roofline.png", dpi=140, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: AI grows with N for matmul, flat for addition
# ──────────────────────────────────────────────────────────────────────────────

def ai_scaling():
    fig, ax = plt.subplots(figsize=(8.5, 5))

    N = np.array([1, 4, 16, 64, 256, 1024, 4096])
    ai_add = np.full_like(N, 1.0 / 12, dtype=float)
    ai_matmul = N / 6.0

    ax.loglog(N, ai_add, "o-", color="#888888", linewidth=2, markersize=8,
              label="c = a + b  (AI = 1/12, constant in N)")
    ax.loglog(N, ai_matmul, "s-", color="#1f77b4", linewidth=2, markersize=8,
              label="N×N matmul  (AI = N/6, grows linearly)")

    # Ridge
    ax.axhline(RIDGE_AI, color="red", linestyle="--", alpha=0.6,
               label=f"H100 FP32 ridge = {RIDGE_AI:.0f} FLOP/byte")

    # Annotate where matmul crosses the ridge
    cross_N = RIDGE_AI * 6
    ax.plot([cross_N], [RIDGE_AI], "rx", markersize=14, markeredgewidth=2.5)
    ax.annotate(f"matmul becomes\ncompute-bound\nat N ≈ {cross_N:.0f}",
                xy=(cross_N, RIDGE_AI),
                xytext=(cross_N * 0.06, RIDGE_AI * 4.5),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1))

    ax.set_xlabel("Problem size N", fontsize=12)
    ax.set_ylabel("Arithmetic Intensity  (FLOP/byte, log scale)", fontsize=12)
    ax.set_title("AI vs problem size — addition stays low, matmul climbs",
                 fontsize=13)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / "ai_scaling.png", dpi=140, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: memory hierarchy pyramid
# ──────────────────────────────────────────────────────────────────────────────

def memory_pyramid():
    fig, ax = plt.subplots(figsize=(9, 6))

    # (name, bandwidth, size, y-position, fill colour)
    tiers = [
        ("Registers",      "≈ instant",   "~64 KB / SM",  0.90, "#2ca02c"),
        ("Shared / L1",    "~31 TB/s",    "~256 KB / SM", 0.74, "#7fbf7f"),
        ("L2 cache",       "~12 TB/s",    "~50 MB",       0.58, "#bfdfbf"),
        ("HBM (GPU DRAM)", "~3.35 TB/s",  "~80 GB",       0.36, "#ffd27f"),
        ("CPU↔GPU PCIe",  "~64 GB/s",    "host memory",  0.12, "#ff9b7f"),
    ]

    for name, bw, size, y, color in tiers:
        width = 0.18 + (1 - y) * 0.78
        rect = patches.Rectangle((0.5 - width / 2, y - 0.065), width, 0.13,
                                  facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(0.5, y, name, ha="center", va="center",
                fontsize=11, fontweight="bold")
        ax.text(0.5 + width / 2 + 0.02, y + 0.022, bw,
                ha="left", va="center", fontsize=10)
        ax.text(0.5 + width / 2 + 0.02, y - 0.030, size,
                ha="left", va="center", fontsize=9, color="dimgray")

    ax.annotate("", xy=(0.05, 0.93), xytext=(0.05, 0.12),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(0.025, 0.93, "fast\nsmall", ha="center", va="top", fontsize=9, color="gray")
    ax.text(0.025, 0.12, "slow\nlarge", ha="center", va="bottom", fontsize=9, color="gray")

    ax.set_xlim(0, 1.6)
    ax.set_ylim(0.0, 1.05)
    ax.axis("off")
    ax.set_title("GPU memory hierarchy — each tier up costs ≈10× bandwidth",
                 fontsize=13)

    plt.tight_layout()
    plt.savefig(OUT / "memory_pyramid.png", dpi=140, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: why c=a+b has tiny AI and matmul has huge AI (reuse)
# ──────────────────────────────────────────────────────────────────────────────

def reuse_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

    # --- LEFT panel: c = a + b ---
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.set_title("c = a + b  —  each loaded byte feeds 1 FLOP",
                  fontsize=12, fontweight="bold")

    # HBM input boxes
    ax1.add_patch(patches.Rectangle((0.4, 4.0), 2, 0.9, facecolor="#cce5ff",
                                     edgecolor="black"))
    ax1.text(1.4, 4.45, "a[i]  (HBM)", ha="center", va="center", fontsize=10)
    ax1.add_patch(patches.Rectangle((0.4, 2.1), 2, 0.9, facecolor="#cce5ff",
                                     edgecolor="black"))
    ax1.text(1.4, 2.55, "b[i]  (HBM)", ha="center", va="center", fontsize=10)

    # Core box
    ax1.add_patch(patches.Rectangle((4.2, 3.0), 1.9, 0.9, facecolor="#fff2cc",
                                     edgecolor="black"))
    ax1.text(5.15, 3.45, "+\nin core", ha="center", va="center", fontsize=10)

    # Output
    ax1.add_patch(patches.Rectangle((7.8, 3.0), 2, 0.9, facecolor="#d5e8d4",
                                     edgecolor="black"))
    ax1.text(8.8, 3.45, "c[i]  (HBM)", ha="center", va="center", fontsize=10)

    ax1.annotate("", xy=(4.2, 3.7), xytext=(2.4, 4.4),
                 arrowprops=dict(arrowstyle="->", lw=1.4))
    ax1.annotate("", xy=(4.2, 3.2), xytext=(2.4, 2.55),
                 arrowprops=dict(arrowstyle="->", lw=1.4))
    ax1.annotate("", xy=(7.8, 3.45), xytext=(6.1, 3.45),
                 arrowprops=dict(arrowstyle="->", lw=1.4))

    ax1.text(5, 1.05,
             "Each element loaded once, used in 1 FLOP, discarded.\n"
             "FLOPs = N, bytes moved = 12N (FP32). AI ≈ 0.083.",
             ha="center", va="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc",
                       edgecolor="#888"))
    ax1.axis("off")

    # --- RIGHT panel: matmul ---
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.set_title("matmul  —  each loaded A[i,k] feeds N FLOPs",
                  fontsize=12, fontweight="bold")

    # HBM input
    ax2.add_patch(patches.Rectangle((0.4, 3.5), 2, 0.9, facecolor="#cce5ff",
                                     edgecolor="black"))
    ax2.text(1.4, 3.95, "A[i,k]  (HBM)", ha="center", va="center", fontsize=10)

    # N uses
    use_rows = [(4.7, "A[i,k] × B[k, j=0]"),
                (3.7, "A[i,k] × B[k, j=1]"),
                (2.7, "A[i,k] × B[k, j=2]"),
                (1.7, "        ⋮"),
                (0.85, "A[i,k] × B[k, j=N−1]")]
    for y, txt in use_rows:
        ax2.add_patch(patches.Rectangle((4.7, y - 0.27), 3.5, 0.55,
                                         facecolor="#fff2cc", edgecolor="black"))
        ax2.text(6.45, y, txt, ha="center", va="center", fontsize=9)
        ax2.annotate("", xy=(4.7, y), xytext=(2.4, 3.95),
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))

    ax2.text(5, 0.15,
             "Each loaded element used in N FLOPs (one per column of B).\n"
             "FLOPs = 2N³, bytes moved = 12N² (FP32). AI = N/6, grows linearly with N.",
             ha="center", va="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc",
                       edgecolor="#888"))
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "reuse_diagram.png", dpi=140, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: FP32 vs BF16 rooflines, with transformer kernels positioned
# ──────────────────────────────────────────────────────────────────────────────

def fp32_vs_bf16_roofline():
    """Two side-by-side rooflines (FP32 and BF16) for the same H100, with
    representative transformer kernels positioned in each regime."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

    BW = 3.35e12                       # H100 HBM3 — same in both
    PEAK_FP32 = 67e12                  # 67 TFLOP/s, no tensor cores
    PEAK_BF16 = 989e12                 # 989 TFLOP/s, BF16/FP16 tensor cores (dense)
    RIDGE_FP32 = PEAK_FP32 / BW        # 20
    RIDGE_BF16 = PEAK_BF16 / BW        # ~295

    ai_range = np.logspace(-2, 4, 500)

    # Representative kernels — (name, FP32 AI, kind)
    # "kind" controls colour: memory-bound (low AI) vs compute-bound (high AI)
    # In BF16 the AI doubles because bytes per element halves.
    kernels = [
        ("residual add",                  0.083, "mem"),
        ("RMSNorm",                       0.4,   "mem"),
        ("softmax (decode)",              0.5,   "mem"),
        ("decode FFN matmul (seq=1)",     0.5,   "mem"),
        ("decode attention (KV=1024)",    0.5,   "mem"),
        ("prefill Q-proj (seq=1024)",     270,   "comp"),
        ("prefill FFN matmul (seq=1024)", 322,   "comp"),
    ]

    def draw(ax, peak, ridge, title, ai_multiplier):
        # Shaded regions
        ax.axvspan(1e-2, ridge, alpha=0.06, color="tab:blue")
        ax.axvspan(ridge, 1e4, alpha=0.06, color="tab:red")

        # Ceilings
        mem = BW * ai_range
        comp = np.full_like(ai_range, peak)
        roof = np.minimum(mem, comp)
        ax.loglog(ai_range, roof, "k-", linewidth=2.5, label="Roofline")
        ax.loglog(ai_range, mem, "b--", linewidth=1, alpha=0.5,
                  label=f"Memory ceiling: {BW/1e12:.2f} TB/s slope")
        ax.loglog(ai_range, comp, "r--", linewidth=1, alpha=0.5,
                  label=f"Compute ceiling: {peak/1e12:.0f} TFLOP/s")

        # Ridge point
        ax.plot([ridge], [peak], "ko", markersize=10, zorder=5)
        ax.annotate(f"Ridge\nAI = {ridge:.0f}",
                    xy=(ridge, peak),
                    xytext=(ridge * 6, peak * 0.18),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

        # Place kernels
        for name, ai_fp32, kind in kernels:
            ai = ai_fp32 * ai_multiplier
            perf = min(peak, BW * ai)
            color = "#2ca02c" if kind == "comp" else "#ff7f0e"
            ax.plot([ai], [perf], "o", color=color, markersize=9,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=4)
            ax.annotate(name, xy=(ai, perf),
                        xytext=(9, -4), textcoords="offset points",
                        fontsize=8, color=color)

        # Region labels
        ax.text(0.18, 4e9, "memory-bound",
                color="tab:blue", fontsize=10, ha="center", alpha=0.85)
        ax.text(2000, 4e9, "compute-bound",
                color="tab:red", fontsize=10, ha="center", alpha=0.85)

        ax.set_xlim(1e-2, 1e4)
        ax.set_ylim(1e9, 5e15)
        ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
        ax.set_title(title, fontsize=12, fontweight="bold")

    draw(ax1, PEAK_FP32, RIDGE_FP32,
         "FP32 (no tensor cores) — H100", ai_multiplier=1.0)
    draw(ax2, PEAK_BF16, RIDGE_BF16,
         "BF16 (tensor cores) — H100",     ai_multiplier=2.0)

    ax1.set_ylabel("Achievable performance (FLOP/s)", fontsize=11)

    fig.suptitle(
        "Same H100, FP32 vs BF16 — roof rises ~15×, kernels shift right (bytes halve)",
        fontsize=13, fontweight="bold")

    # Legend for kernel colours
    fig.text(0.5, 0.005,
             "● memory-bound kernels (decode, norms, residuals)     "
             "● compute-bound kernels (prefill matmuls)",
             ha="center", fontsize=10,
             color="dimgray")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(OUT / "fp32_vs_bf16_roofline.png", dpi=140, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    roofline()
    ai_scaling()
    memory_pyramid()
    reuse_diagram()
    fp32_vs_bf16_roofline()
    print(f"Wrote figures to {OUT}")
