"""
Paper figures from the real-data study.
  fig_frontier.pdf   accuracy vs cost frontiers (Qwen, Mistral; one split, recomputed)
  fig_confidence.pdf E2: gain over the subset cascade vs confidence AUROC (from results_real)
  fig_labels.pdf     E3: AUCC vs number of labelled questions (from results_real)
Palette: validated default categorical slots 1-6, fixed per method; every line also has
its own marker + line style and a direct label (three slots are < 3:1 contrast on white).
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import policies as P
import real_eval as E
from real_data import load_cascade

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results_real")
FIG = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {  # method -> (label, color, marker, linestyle)   color follows the method everywhere
    "Subset cascade (<=3 models)": ("Subset cascade", "#2a78d6", "o", "-"),
    "Per-stage threshold":         ("Per-stage thr.", "#eb6834", "s", "--"),
    "VoC-Stop (paper)":            ("VoC-Stop", "#1baf7a", "^", "-."),
    "Learned deferral (GBM)":      ("Learned deferral", "#eda100", "D", ":"),
    "Per-stage+cat threshold":     ("Per-stage+cat thr.", "#e87ba4", "v", (0, (5, 1, 1, 1))),
    "Global threshold":            ("Global thr.", "#008300", "x", (0, (1, 1))),
}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
    "lines.linewidth": 1.4, "lines.markersize": 4, "pdf.fonttype": 42,
})
CASC_TITLE = {"qwen1.5": "Qwen1.5 (0.5B→72B, 7 models)", "llama2": "Llama-2 (7B/13B/70B)",
              "mistral": "Mistral (7B/8x7B/8x22B)"}


def _label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(3, dy), textcoords="offset points", fontsize=6.5,
                color=INK2, va="center")
    ax.plot([x], [y], marker="o", ms=2, color=color)


def frontier_panel(ax, name, seed=0):
    data, costs, names = load_cascade(name)
    P.set_costs(costs)
    tr, te = E.split(data, seed)
    st = P.FittedStats(tr); p_tr, p_te = st.p_hat(tr["conf"]), st.p_hat(te["conf"])
    LAM = P.CUMCOST[-1] * np.logspace(-2.5, 2.5, 30)
    big = costs[-1]
    curves = {
        "Global threshold": E.frontier(te, [E.stop_stage(p_te, E.fit_global_threshold(p_tr, tr["correct"], l)) for l in LAM]),
        "Per-stage threshold": E.frontier(te, [E.stop_stage(p_te, E.fit_stage_thresholds(p_tr, tr["correct"], l)) for l in LAM]),
        "VoC-Stop (paper)": E.frontier(te, [E._voc_stop_stage(P.VoCStopLA(tr, st), te, l) for l in LAM[::1]]),
    }
    sub = [E.subset_cascade(tr, te, p_tr, p_te, costs, l) for l in LAM[::2]]
    curves["Subset cascade (<=3 models)"] = (np.array([s[0] for s in sub]), np.array([s[1] for s in sub]))
    for k in ["Global threshold", "Per-stage threshold", "VoC-Stop (paper)", "Subset cascade (<=3 models)"]:
        c, a = E.pareto_envelope(*curves[k])
        lab, col, mk, ls = STYLE[k]
        ax.step(100 * c / big, 100 * a, where="post", color=col, ls=ls, lw=1.4, label=lab)
        ax.plot(100 * c / big, 100 * a, ls="none", marker=mk, ms=3, color=col, mfc="white", mew=0.9)
    sm = te["correct"].mean(0)
    ax.plot(100 * costs / big, 100 * sm, ls="none", marker="o", ms=4, color=INK2, zorder=5,
            label="Single models")
    short = {"qwen1.5": ["0.5B", "1.8B", "4B", "7B", "14B", "32B", "72B"],
             "mistral": ["7B", "8x7B", "8x22B"], "llama2": ["7B", "13B", "70B"]}[name]
    for x, y, n in zip(100 * costs / big, 100 * sm, short):
        left = x > 4                               # tiny models sit on the y-axis: label to the right
        ax.annotate(n, (x, y), xytext=(-4 if left else 5, 5 if left else -3), textcoords="offset points",
                    fontsize=6.5, color=INK, ha="right" if left else "left", fontweight="bold")
    ax.axhline(99 * sm[-1], color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.annotate("99% of largest model", (148, 99 * sm[-1]), xytext=(0, -8), textcoords="offset points",
                fontsize=6, color=INK2, ha="right")
    ax.set_xlim(0, 150); ax.set_title(CASC_TITLE[name], fontsize=8, color=INK)
    ax.set_xlabel("cost per query (% of largest model alone)")


def fig_frontier():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    frontier_panel(axs[0], "qwen1.5"); frontier_panel(axs[1], "mistral")
    axs[0].set_ylim(35, 80); axs[1].set_ylim(62, 80)
    axs[0].set_ylabel("accuracy (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIG, "fig_frontier.pdf")); fig.savefig(os.path.join(FIG, "fig_frontier.png"), dpi=200)


def fig_confidence():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=False)
    show = ["Per-stage threshold", "Per-stage+cat threshold", "Learned deferral (GBM)", "VoC-Stop (paper)"]
    for ax, name in zip(axs, ["qwen1.5", "llama2", "mistral"]):
        rows = json.load(open(os.path.join(RES, f"e2_{name}.json")))["rows"]
        rows = [r for r in rows if r["signal"].startswith(("degrade", "sharpen")) or r["signal"] == "real maxprob"]
        rows.sort(key=lambda r: r["auroc"])
        x = np.array([r["auroc"] for r in rows])
        base = np.array([np.mean(r["aucc"]["Subset cascade (<=3 models)"]) for r in rows])
        ax.axhline(0, color=STYLE["Subset cascade (<=3 models)"][1], lw=1.4)
        ax.annotate("subset cascade (= 0)", (x[-1], 0), xytext=(0, -9), textcoords="offset points",
                    fontsize=6.5, color=INK2, ha="right")
        for k in show:
            lab, col, mk, ls = STYLE[k]
            y = np.array([np.mean(r["aucc"][k]) for r in rows]) - base
            ax.plot(x, y, color=col, ls=ls, marker=mk, mfc="white", mew=0.9, label=lab)
        real = [r["auroc"] for r in rows if r["signal"] == "real maxprob"][0]
        ax.axvline(real, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.annotate("real signal", (real, ax.get_ylim()[1]), xytext=(2, -8), textcoords="offset points",
                    fontsize=6, color=INK2)
        ax.set_title(CASC_TITLE[name], fontsize=8, color=INK)
        ax.set_xlabel("confidence AUROC (mean over models)")
    axs[0].set_ylabel("AUCC minus subset cascade")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG, "fig_confidence.pdf")); fig.savefig(os.path.join(FIG, "fig_confidence.png"), dpi=200)


def fig_labels():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2))
    ns = [100, 300, 1000, 3000]
    show = ["Subset cascade (<=3 models)", "Per-stage threshold", "Per-stage+cat threshold",
            "Learned deferral (GBM)", "VoC-Stop (paper)"]
    for ax, name in zip(axs, ["qwen1.5", "llama2", "mistral"]):
        full = json.load(open(os.path.join(RES, f"e1_{name}_all.json")))
        nfull = full["n"] // 2
        for k in show:
            lab, col, mk, ls = STYLE[k]
            y = [np.mean(json.load(open(os.path.join(RES, f"e3n{n}_{name}_all.json")))["summary"][k]["aucc"])
                 for n in ns] + [np.mean(full["summary"][k]["aucc"])]
            ax.plot(ns + [nfull], y, color=col, ls=ls, marker=mk, mfc="white", mew=0.9, label=lab)
        ax.set_xscale("log"); ax.set_xticks(ns + [nfull]); ax.set_xticklabels(["100", "300", "1k", "3k", "all"])
        ax.minorticks_off()
        ax.set_title(CASC_TITLE[name], fontsize=8, color=INK); ax.set_xlabel("labelled questions for tuning")
    axs[0].set_ylabel("AUCC"); axs[0].set_ylim(0.52, 0.74)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG, "fig_labels.pdf")); fig.savefig(os.path.join(FIG, "fig_labels.png"), dpi=200)


if __name__ == "__main__":
    fig_confidence(); fig_labels(); fig_frontier()
    print("figures ->", os.path.abspath(FIG))
