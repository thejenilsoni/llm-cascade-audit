"""Print the paper's tables from results_real/*.json (E1, E3, E4, E5; E2 is printed by e2_confidence.py)."""
from __future__ import annotations
import json, os
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results_real")
CASC = ["qwen1.5", "llama2", "mistral"]
CORE = ["Global threshold", "Per-stage threshold", "Per-stage+cat threshold", "VoC-Stop (paper)",
        "Learned deferral (GBM)", "Subset cascade (<=3 models)", "Single models (no cascade)", "Oracle"]
SHORT = {"Global threshold": "Global", "Per-stage threshold": "PerStage", "Per-stage+cat threshold": "PS+cat",
         "VoC-Stop (paper)": "VoC", "Learned deferral (GBM)": "GBM", "Subset cascade (<=3 models)": "Subset",
         "Single models (no cascade)": "Single", "Oracle": "Oracle",
         "abl: raw conf, global thr": "raw+global", "abl: VoC myopic (1-step)": "VoC-myopic",
         "abl: VoC lookahead, no cat": "VoC-nocat"}


def load(tag, c, tasks="all"):
    with open(os.path.join(RES, f"{tag}_{c}_{tasks}.json")) as fh:
        return json.load(fh)["summary"]


def cell(s, k):
    a = np.array(s[k]["aucc"]); return f"{a.mean():.4f}"


def c99(s, k):
    xs = s[k]["cost_match99"]
    return "n/a" if any(x is None for x in xs) else f"{np.mean(xs):.0f}%"


def table(title, tag, methods, tasks="all"):
    print(f"\n### {title}")
    print(f"{'method':14s}" + "".join(f"{c:>20s}" for c in CASC))
    sums = {c: load(tag, c, tasks) for c in CASC}
    best = {c: max((k for k in methods if k in sums[c] and k != "Oracle"),
                   key=lambda k: np.mean(sums[c][k]["aucc"])) for c in CASC}
    for k in methods:
        row = f"{SHORT[k]:14s}"
        for c in CASC:
            if k in sums[c]:
                row += f"{cell(sums[c], k) + ('*' if best[c] == k else ' '):>12s} {c99(sums[c], k):>6s} "
            else:
                row += f"{'-':>20s}"
        print(row)
    print("(AUCC, * = best non-oracle; then cost to reach 99% of the largest model's accuracy)")


def e3():
    print("\n### E3 label budget: AUCC by number of labelled questions (full = ~8,240)")
    ns = [100, 300, 1000, 3000]
    for c in CASC:
        print(f"  {c}")
        print(f"  {'method':12s}" + "".join(f"{n:>9d}" for n in ns) + f"{'full':>9s}")
        full = load("e1", c)
        for k in CORE[1:6]:
            vals = [np.mean(load(f"e3n{n}", c)[k]["aucc"]) for n in ns] + [np.mean(full[k]["aucc"])]
            print(f"  {SHORT[k]:12s}" + "".join(f"{v:9.4f}" for v in vals))


if __name__ == "__main__":
    table("E1 main comparison (5 splits)", "e1", CORE)
    e3()
    table("E4 task shift: tune on MMLU -> test on ARC+Winogrande (5 train subsamples)", "e4shift", CORE)
    table("E5 ablation of VoC-Stop's ingredients (5 splits)", "e5abl",
          ["abl: raw conf, global thr", "Global threshold", "Per-stage threshold", "abl: VoC myopic (1-step)",
           "abl: VoC lookahead, no cat", "VoC-Stop (paper)", "Per-stage+cat threshold", "Learned deferral (GBM)"])
