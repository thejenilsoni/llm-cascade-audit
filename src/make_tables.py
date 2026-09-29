"""Write the paper's LaTeX tables straight from results_real/*.json (no hand-copied numbers)."""
from __future__ import annotations
import json, os
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results_real")
OUT = os.path.join(HERE, "..", "paper", "tables")
os.makedirs(OUT, exist_ok=True)
CASC = ["qwen1.5", "llama2", "mistral"]
NAME = {"Global threshold": "Global threshold", "Per-stage threshold": "Per-stage threshold",
        "Per-stage+cat threshold": "Per-stage + category thr.", "VoC-Stop (paper)": "VoC-Stop",
        "Learned deferral (GBM)": "Learned deferral (GBM)", "Subset cascade (<=3 models)": "Subset cascade ($\\le$3)",
        "Single models (no cascade)": "Single models (no cascade)", "Oracle": "Oracle (upper bound)",
        "abl: raw conf, global thr": "raw confidence, global thr.", "abl: VoC myopic (1-step)": "VoC, one-step (myopic)",
        "abl: VoC lookahead, no cat": "VoC, lookahead, no category"}
CORE = ["Global threshold", "Per-stage threshold", "Per-stage+cat threshold", "VoC-Stop (paper)",
        "Learned deferral (GBM)", "Subset cascade (<=3 models)", "Single models (no cascade)", "Oracle"]


def load(tag, c):
    return json.load(open(os.path.join(RES, f"{tag}_{c}_all.json")))["summary"]


def c99(xs):
    return "--" if any(x is None for x in xs) else f"{np.mean(xs):.0f}\\%"


def results_table(tag, methods, caption, label, sd=True):
    S = {c: load(tag, c) for c in CASC}
    best = {c: max((m for m in methods if m != "Oracle" and m in S[c]), key=lambda m: np.mean(S[c][m]["aucc"]))
            for c in CASC}
    L = ["\\begin{table*}[t]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\begin{tabular}{l" + "rr" * 3 + "}\\toprule",
         " & \\multicolumn{2}{c}{Qwen1.5 (7 models)} & \\multicolumn{2}{c}{Llama-2 (3)} & \\multicolumn{2}{c}{Mistral (3)}\\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "Method & AUCC & Cost@99 & AUCC & Cost@99 & AUCC & Cost@99\\\\\\midrule"]
    for m in methods:
        if m == "Oracle":
            L.append("\\midrule")
        row = NAME[m]
        for c in CASC:
            a = np.array(S[c][m]["aucc"])
            v = f"{a.mean():.3f}" + (f"\\,{{\\scriptsize$\\pm${a.std():.3f}}}" if sd else "")
            if best[c] == m:
                v = "\\textbf{" + v + "}"
            row += f" & {v} & {c99(S[c][m]['cost_match99'])}"
        L.append(row + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table*}"]
    return "\n".join(L)


def conf_table():
    cols = ["Per-stage threshold", "Per-stage+cat threshold", "Learned deferral (GBM)", "VoC-Stop (paper)",
            "Subset cascade (<=3 models)"]
    head = ["Per-stage", "PS+cat", "GBM", "VoC-Stop", "Subset"]
    order = ["degrade 2.0", "degrade 1.0", "real maxprob", "sharpen 0.5", "sharpen 1.0", "sharpen 2.0"]
    pretty = {"degrade 2.0": "noised ($\\sigma{=}2$)", "degrade 1.0": "noised ($\\sigma{=}1$)",
              "real maxprob": "\\textit{real (max-prob)}", "sharpen 0.5": "sharpened ($\\beta{=}0.5$)",
              "sharpen 1.0": "sharpened ($\\beta{=}1$)", "sharpen 2.0": "sharpened ($\\beta{=}2$)"}
    L = ["\\begin{table*}[t]\\centering\\small",
         "\\caption{E2: AUCC as confidence quality varies (3 splits). Sharpened signals use the label and "
         "are \\emph{not} deployable; they show what a better confidence estimator would buy. Best non-oracle in bold.}",
         "\\label{tab:conf}", "\\begin{tabular}{llrrrrrr}\\toprule",
         " & Signal & AUROC & " + " & ".join(head) + "\\\\\\midrule"]
    for c, cname in zip(CASC, ["Qwen", "Llama-2", "Mistral"]):
        rows = {r["signal"]: r for r in json.load(open(os.path.join(RES, f"e2_{c}.json")))["rows"]}
        for i, sgl in enumerate(order):
            r = rows[sgl]; vals = [np.mean(r["aucc"][m]) for m in cols]; b = int(np.argmax(vals))
            cells = [("\\textbf{%.3f}" if j == b else "%.3f") % v for j, v in enumerate(vals)]
            L.append(f"{cname if i == 0 else ''} & {pretty[sgl]} & {r['auroc']:.2f} & " + " & ".join(cells) + "\\\\")
        if c != CASC[-1]:
            L.append("\\midrule")
    L += ["\\bottomrule\\end{tabular}", "\\end{table*}"]
    return "\n".join(L)


if __name__ == "__main__":
    tabs = {
        "tab_main.tex": results_table("e1", CORE,
            "E1: main comparison on real data (16{,}481 questions; mean $\\pm$ s.d.\\ over 5 random splits; every "
            "method tuned on the training half only). AUCC: area under the accuracy--cost curve (higher is better). "
            "Cost@99: cost to reach 99\\% of the largest model's accuracy, as \\% of calling that model alone "
            "(-- = not reached in every split). Best non-oracle AUCC in bold.", "tab:main"),
        "tab_shift.tex": results_table("e4shift", CORE,
            "E4: task shift. Rules tuned on MMLU, evaluated on ARC-Challenge + WinoGrande (category information "
            "removed; 5 training subsamples).", "tab:shift", sd=False),
        "tab_ablation.tex": results_table("e5abl", ["abl: raw conf, global thr", "Global threshold",
            "Per-stage threshold", "abl: VoC myopic (1-step)", "abl: VoC lookahead, no cat", "VoC-Stop (paper)",
            "Per-stage+cat threshold"],
            "E5: ablation, adding one ingredient at a time (5 splits).", "tab:ablation", sd=False),
        "tab_conf.tex": conf_table(),
    }
    for f, t in tabs.items():
        open(os.path.join(OUT, f), "w", encoding="utf-8").write(t + "\n")
    print("tables ->", os.path.abspath(OUT))
    # key numbers for the text
    for c in CASC:
        rows = {r["signal"]: r for r in json.load(open(os.path.join(RES, f"e2_{c}.json")))["rows"]}
        print(c, "real AUROC %.3f" % rows["real maxprob"]["auroc"],
              "| cost99 real subset", c99(rows["real maxprob"]["cost99"]["Subset cascade (<=3 models)"]),
              "| sharpen1 best cost99", {m[:8]: c99(rows["sharpen 1.0"]["cost99"][m]) for m in CORE[1:6]})
