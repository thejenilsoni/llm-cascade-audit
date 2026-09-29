"""
Run the full cascade evaluation: sweep each policy's control parameter to trace
its accuracy-vs-cost frontier, compute headline metrics with bootstrap CIs,
and dump results for plotting and for the paper's tables.
"""
from __future__ import annotations
import json, os
import numpy as np

import data_gen
import policies as P
from calibration_table import MODEL_NAMES, MODEL_ACC, MODEL_COST, N_MODELS

OUT = os.path.join(os.path.dirname(__file__), "..", "results_sim")   # created on first write
COST_MAX = MODEL_COST[-1]          # always-GPT-4 cost = normalization unit


# ---------- frontier helpers -------------------------------------------------
def pareto_envelope(cost, acc):
    """Upper-left envelope: sort by cost, keep running max accuracy."""
    order = np.argsort(cost)
    c, a = cost[order], acc[order]
    best = np.maximum.accumulate(a)
    # collapse to unique non-dominated steps
    keep = np.ones(len(c), dtype=bool)
    keep[1:] = best[1:] > best[:-1] + 1e-12
    keep[0] = True
    return c[keep], best[keep]


def aucc(cost, acc, c_lo, c_hi):
    """Area under accuracy-cost envelope over [c_lo, c_hi], normalized to [0,1]
    on the cost axis. Accuracy is the best achievable at <= a given cost."""
    c, a = pareto_envelope(np.asarray(cost), np.asarray(acc))
    # build a step function acc(cost) = best achievable at <= cost
    grid = np.linspace(c_lo, c_hi, 400)
    vals = np.empty_like(grid)
    for i, g in enumerate(grid):
        ok = c <= g + 1e-15
        vals[i] = a[ok].max() if ok.any() else a[0]
    return np.trapezoid(vals, (grid - c_lo) / (c_hi - c_lo))


def cost_at_accuracy(cost, acc, target):
    """Min cost on the frontier achieving >= target accuracy (None if infeasible)."""
    c, a = pareto_envelope(np.asarray(cost), np.asarray(acc))
    ok = a >= target - 1e-12
    return float(c[ok].min()) if ok.any() else None


# ---------- run one policy's sweep, returning per-item matrices --------------
def sweep(fn, params):
    """Returns dict with means (cost,acc per param) and per-item matrices
    [n_params, n_items] for bootstrapping."""
    accs, costs, CI, OI = [], [], [], []
    for prm in params:
        mean_acc, mean_cost, ci, oi = fn(prm)
        accs.append(mean_acc); costs.append(mean_cost)
        CI.append(ci); OI.append(oi)
    return dict(params=np.asarray(params), acc=np.array(accs), cost=np.array(costs),
                item_correct=np.array(CI), item_cost=np.array(OI))


def run_setting(setting, data, costs_vec, targets, anchor_names, baseline_label):
    """Generic frontier evaluation shared by the cascade and debate settings.
    `baseline_label` names the scalar-threshold policy (Conf/Consensus)."""
    os.makedirs(OUT, exist_ok=True)
    P.set_costs(costs_vec)
    train, test = data_gen.train_test_split(data)
    stats = P.FittedStats(train)
    dp = P.VoCStopDP(train, stats)
    la = P.VoCStopLA(train, stats)
    rng = np.random.default_rng(7)

    TAU = np.linspace(0.001, 0.999, 80)
    LAM = np.logspace(-3.5, 1.5, 100)

    sweeps = {
        baseline_label:             sweep(lambda t: P.conf_threshold(test, t), TAU),
        "Calibrated-Threshold":     sweep(lambda t: P.calibrated_threshold(test, stats, t), TAU),
        "VoC-Stop (myopic, abl.)":  sweep(lambda l: P.voc_stop(test, stats, l), LAM),
        "VoC-Stop (DP, abl.)":      sweep(lambda l: dp.run(test, l), LAM),
        "VoC-Stop (ours)":          sweep(lambda l: la.run(test, l), LAM),
        "Oracle-Stop (upper bnd)":  sweep(lambda l: P.oracle_stop(test, l), LAM),
    }
    # fixed-stage anchors (single model / fixed-K rounds)
    cum = np.cumsum(costs_vec)
    singles = {"cost": [], "acc": [], "name": []}
    for m in range(test["correct"].shape[1]):
        a, c, _, _ = P.single_model(test, m)
        singles["cost"].append(c); singles["acc"].append(a)
        singles["name"].append(anchor_names[m] if m < len(anchor_names) else f"stage{m}")

    c_lo = float(cum[0]); c_hi = float(cum[-1])
    B = 300
    n_items = test["correct"].shape[0]
    summary = {}
    boot_idx = [rng.integers(0, n_items, n_items) for _ in range(B)]

    for name, sw in sweeps.items():
        IC, ICST = sw["item_correct"], sw["item_cost"]   # [n_params, n_items]
        # point estimates
        a_pt = sw["acc"]; c_pt = sw["cost"]
        au = aucc(c_pt, a_pt, c_lo, c_hi)
        iso = {k: cost_at_accuracy(c_pt, a_pt, t) for k, t in targets.items()}
        # bootstrap
        au_b = []
        iso_b = {k: [] for k in targets}
        for bi in boot_idx:
            a_b = IC[:, bi].mean(1); c_b = ICST[:, bi].mean(1)
            au_b.append(aucc(c_b, a_b, c_lo, c_hi))
            for k, t in targets.items():
                v = cost_at_accuracy(c_b, a_b, t)
                if v is not None:
                    iso_b[k].append(v)
        def ci(x):
            x = np.asarray(x); return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        summary[name] = dict(
            aucc=float(au), aucc_ci=ci(au_b),
            iso_cost={k: (None if iso[k] is None else float(iso[k])) for k in targets},
            iso_cost_ci={k: (ci(iso_b[k]) if iso_b[k] else None) for k in targets},
        )

    # cost-saving of VoC-Stop vs the threshold baselines, at iso-accuracy
    savings = {}
    for ref in [baseline_label, "Calibrated-Threshold"]:
        savings[ref] = {}
        for k in targets:
            v = summary["VoC-Stop (ours)"]["iso_cost"][k]
            r = summary[ref]["iso_cost"][k]
            if v and r:
                savings[ref][k] = dict(pct=100.0 * (1 - v / r), voc=v, ref=r)

    # save everything for plotting
    np.savez(os.path.join(OUT, f"sweeps_{setting}.npz"),
             **{f"{n}__{key}": sw[key] for n, sw in sweeps.items()
                for key in ("params", "acc", "cost")},
             singles_cost=np.array(singles["cost"]), singles_acc=np.array(singles["acc"]),
             c_lo=c_lo, c_hi=c_hi)
    with open(os.path.join(OUT, f"summary_{setting}.json"), "w") as f:
        json.dump(dict(setting=setting, summary=summary, savings=savings, singles=singles,
                       targets=targets, c_lo=c_lo, c_hi=c_hi, baseline_label=baseline_label,
                       n_test=int(n_items)), f, indent=2)
    np.savez(os.path.join(OUT, f"diag_{setting}.npz"),
             test_correct=test["correct"], test_conf=test["conf"],
             test_cat=test["cat"], p_hat=stats.p_hat(test["conf"]),
             u=stats.u, v=stats.v)

    # ---- console report -----------------------------------------------------
    print(f"\n=== {setting.upper()} EVALUATION (n_test={n_items}) ===")
    print(f"{'Policy':28s} {'AUCC':>8s}  {'95% CI':>18s}")
    for name, s in summary.items():
        lo, hi = s["aucc_ci"]
        print(f"{name:28s} {s['aucc']:.4f}  [{lo:.4f}, {hi:.4f}]")
    print("\n--- Cost to reach iso-accuracy ($/query, lower=better) ---")
    hdr = f"{'Policy':28s}" + "".join([f"{k:>22s}" for k in targets])
    print(hdr)
    for name, s in summary.items():
        row = f"{name:28s}"
        for k in targets:
            val = s["iso_cost"][k]
            row += f"{('%.5f'%val if val else 'n/a'):>22s}"
        print(row)
    print("\n--- VoC-Stop cost savings at iso-accuracy ---")
    for ref, d in savings.items():
        for k, info in d.items():
            print(f"  vs {ref:24s} @ {k:22s}: {info['pct']:5.1f}% "
                  f"({info['voc']:.5f} vs {info['ref']:.5f})")
    print(f"\nSaved -> results/summary_{setting}.json, sweeps_{setting}.npz, diag_{setting}.npz")
    return summary, savings


def main():
    """Flagship cascade setting (grounded in published RouterBench marginals)."""
    data = data_gen.generate()
    targets = {"match-GPT3.5 (0.685)": 0.685, "match-Claude (0.740)": 0.740,
               "high (0.780)": 0.780}
    run_setting("cascade", data, MODEL_COST, targets, MODEL_NAMES,
                baseline_label="Conf-Threshold (raw)")


if __name__ == "__main__":
    main()
