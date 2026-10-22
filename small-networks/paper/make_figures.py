"""Figures of the manuscript, drawn from saved results only (paper/figures/*.pdf)."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
os.environ.setdefault("MPLCONFIGDIR", str(PAPER / "build/mpl"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

S1P = PROJECT / "studies/1-compiled-programs/posthoc/results"
S2R = PROJECT / "studies/2-trained-networks/results/final"
OUT = PAPER / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical order (light surface)
BUDGETS = [12, 24, 60, 120]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})


def study1():
    sim = json.loads((S1P / "panel-simulation.json").read_text())
    rows = {(r["panel"], r["budget"]): r for r in sim["rows"]}
    panels = [("ordinary", "Restricted singles"), ("full_single", "Full-domain singles"),
              ("heldout_single", "Held-out singles"), ("joint", "Joint"),
              ("faithfulness", "Complement resampling"), ("transport", "Transported divergence")]
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 3.3), sharex=True, sharey=True, layout="constrained")
    for ax, (key, title) in zip(axes.flat, panels):
        r = [rows[(key, b)] for b in BUDGETS]
        ax.fill_between(BUDGETS, [x["range95_low"] for x in r], [x["range95_high"] for x in r], color="#c3c2b7",
                        alpha=0.75, lw=0, label="central 95% of replays")
        ax.plot(BUDGETS, [(x["mean_ties_fail"] + x["mean_ties_pass"]) / 2 for x in r], color="#8f8e88", lw=1.0,
                label="expected")
        ax.plot(BUDGETS, [x["observed"] for x in r], ls="none", marker="o", ms=4.2, mfc=SERIES[0], mec="white",
                mew=0.8, label="frozen decision")
        ax.set_title(title, color=INK, pad=3)
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.minorticks_off()
        ax.set_ylim(-4, 100)
        ax.set_yticks([0, 40, 80])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    # Constrained layout places the shared axis labels next to the tick labels (no gap above the x label).
    fig.supxlabel("Budget (model calls per alignment)", fontsize=8.5)
    fig.supylabel("Invalid alignments accepted (of 160)", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    fig.savefig(OUT / "fig-compiled-panels.pdf")
    plt.close(fig)


def study2():
    if not (S2R / "analysis.json").exists():
        return
    a = json.loads((S2R / "analysis.json").read_text())
    rows = [r for r in csv.DictReader(open(S2R / "candidates.csv")) if r["candidate"] in ("das_r1", "das_r2", "das_r4")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.75), gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) single versus joint failure rate of every full-domain DAS alignment (one series: no legend)
    xs = [(float(r["p_full_single_A"]) + float(r["p_full_single_B"])) / 2 for r in rows]
    ys = [float(r["p_joint"]) for r in rows]
    ax1.scatter(xs, ys, s=8, color=INK2, alpha=0.45, lw=0)
    ax1.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=(0, (3, 2)))
    ax1.text(0.97, 0.90, "joint = single", color=INK, fontsize=7, rotation=45, ha="right", va="bottom")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Failure rate, full-domain singles")
    ax1.set_ylabel("Failure rate, joint queries")
    ax1.set_title("(a) Full-domain DAS alignments", color=INK, pad=3)
    ax1.grid(color=GRID, lw=0.6)
    ax1.set_axisbelow(True)
    # (b) expected detection at matched budgets, averaged over the same alignments
    det = a["detection"]["das"]
    panels = [("full_single", "full-domain singles", "o"), ("joint", "joint", "s"),
              ("minimal_single", "minimal-pair singles", "^"), ("restricted_single", "restricted singles", "D")]
    for (key, label, marker), color in zip(panels, SERIES):
        ys = [det[key][str(b)] for b in BUDGETS]
        ax2.plot(BUDGETS, ys, color=color, lw=1.6, marker=marker, ms=4, mec="white", mew=0.6, label=label)
    ceiling = 1 - json.loads((PROJECT / "studies/2-trained-networks/posthoc/results/detection-sensitivity.json")
                             .read_text())["overall"]["fail_no_panel_query"] / len(rows)
    ax2.axhline(ceiling, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax2.text(12, ceiling + 0.02, "ceiling: alignments that fail some query", color=INK2, fontsize=7)
    ax2.legend(loc="lower right", fontsize=7, handlelength=2.2)
    ax2.set_xscale("log")
    ax2.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    ax2.minorticks_off()
    ax2.set_xlim(10, 140)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Budget (model calls per alignment)")
    ax2.set_ylabel("Expected share detected")
    ax2.set_title("(b) Detection at matched budgets", color=INK, pad=3)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig-trained.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    study1()
    study2()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
