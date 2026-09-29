"""
E2: does confidence-signal quality change which stopping rule wins?

E2a (controlled): transform the real max-prob confidence in logit space
    degrade:  logit(c) + N(0, sigma^2)            (sigma = 1, 2)
    sharpen:  logit(c) + beta * (2*correct - 1)   (beta = 0.5, 1, 2)
Sharpening uses the label, so it is NOT a deployable signal: it stands in for
"a better confidence estimator with AUROC x" and tells us what a better
estimator would buy.  The mean AUROC of each level is reported.
E2b (real): alternative signals from the same logs (margin, entropy, agree).

Usage: python e2_confidence.py qwen1.5 [n_splits]
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

import real_eval as E
from real_data import load_cascade

METHODS = ["Global threshold", "Per-stage threshold", "Per-stage+cat threshold", "VoC-Stop (paper)",
           "Learned deferral (GBM)", "Subset cascade (<=3 models)", "Single models (no cascade)", "Oracle"]


def transform(data, kind, amount, seed=123):
    rng = np.random.default_rng(seed)
    z = np.log(np.clip(data["conf"], 1e-6, 1 - 1e-6) / np.clip(1 - data["conf"], 1e-6, 1))
    if kind == "degrade":
        z = z + rng.normal(0, amount, z.shape)
    elif kind == "sharpen":
        z = z + amount * (2 * data["correct"] - 1)
    return {**data, "conf": 1 / (1 + np.exp(-z))}


def mean_auroc(data):
    c, p = data["correct"], data["conf"]
    return float(np.mean([roc_auc_score(c[:, j], p[:, j]) for j in range(c.shape[1])]))


def main(name, n_splits=3):
    base, costs, names = load_cascade(name, "all", "maxprob")
    variants = [("degrade", 2.0), ("degrade", 1.0), ("real maxprob", 0), ("sharpen", 0.5),
                ("sharpen", 1.0), ("sharpen", 2.0)]
    variants_real = ["margin", "entropy", "agree"]
    rows = []
    for kind, amt in variants:
        d = base if kind.startswith("real") else transform(base, kind, amt)
        label = kind if kind.startswith("real") else f"{kind} {amt}"
        rows.append((label, mean_auroc(d), E.run(name, "all", n_splits, (d, costs, names), verbose=False)))
        print(f"  done: {label}", flush=True)
    for ct in variants_real:
        d, _, _ = load_cascade(name, "all", ct)
        rows.append((f"real {ct}", mean_auroc(d), E.run(name, "all", n_splits, (d, costs, names), verbose=False)))
        print(f"  done: real {ct}", flush=True)

    short = ["Global", "PerStage", "PS+cat", "VoC", "GBM", "Subset", "Single", "Oracle"]
    print(f"\n=== E2 {name}: mean AUCC over {n_splits} splits (best non-oracle in *)")
    print(f"{'signal':18s} {'AUROC':>6s} " + " ".join(f"{s:>9s}" for s in short))
    out = []
    for label, au, res in rows:
        vals = [np.mean(res[m]["aucc"]) for m in METHODS]
        best = int(np.argmax(vals[:-1]))
        cells = [f"{v:.4f}{'*' if i == best else ' '}" for i, v in enumerate(vals)]
        print(f"{label:18s} {au:6.3f} " + " ".join(f"{c:>9s}" for c in cells))
        c99 = res["Subset cascade (<=3 models)"]["cost_match99"]
        out.append(dict(signal=label, auroc=au, aucc={m: res[m]["aucc"] for m in METHODS},
                        cost99={m: res[m]["cost_match99"] for m in METHODS}))
    print("\ncost to 99% of largest model's accuracy (% of its cost; n/a = never reached):")
    print(f"{'signal':18s} " + " ".join(f"{s:>9s}" for s in short[:-1]))
    for o in out:
        cells = []
        for m in METHODS[:-1]:
            xs = o["cost99"][m]
            cells.append("n/a" if any(x is None for x in xs) else f"{np.mean(xs):.0f}%")
        print(f"{o['signal']:18s} " + " ".join(f"{c:>9s}" for c in cells))
    with open(os.path.join(E.OUT, f"e2_{name}.json"), "w") as fh:
        json.dump(dict(cascade=name, n_splits=n_splits, rows=out), fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen1.5", int(sys.argv[2]) if len(sys.argv) > 2 else 3)
