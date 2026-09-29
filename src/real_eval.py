"""
E1: main comparison of stopping rules on REAL cascade data.

Every method sees the same inputs (calibrated confidences so far + category) and
is tuned on the same train split; all scores are on the held-out split, repeated
over several random splits.  Methods:

  Global threshold      one calibrated-confidence threshold for all stages (old paper's baseline)
  Per-stage threshold   one threshold per stage, tuned per lambda (FrugalGPT-style)
  Per-stage+cat thresh  a threshold per (stage, category), tuned per lambda
  VoC-Stop              the paper's lookahead rule (policies.VoCStopLA)
  Learned deferral      gradient-boosted P(model t correct | features at stage s) + lookahead
  Single models         call one model only (no cascade) -- the "is a cascade even worth it?" check
  Oracle                clairvoyant per-question stop (upper bound)

Usage: python real_eval.py qwen1.5 [tasks] [n_splits]
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import policies as P
from experiments import aucc, cost_at_accuracy, pareto_envelope
from real_data import load_cascade

OUT = os.path.join(os.path.dirname(__file__), "..", "results_real")
os.makedirs(OUT, exist_ok=True)
GRID = np.r_[np.linspace(0.0, 1.0, 41), 1.01]      # 1.01 = "never stop at this stage"


def split(data, seed):
    n = data["correct"].shape[0]
    idx = np.random.default_rng(seed).permutation(n); k = n // 2
    pick = lambda ix: {key: (v[ix] if hasattr(v, "shape") and v.shape[:1] == (n,) else v)
                       for key, v in data.items()}
    return pick(idx[:k]), pick(idx[k:])


# ---------------------------------------------------------------- threshold rules
def stop_stage(p, taus):
    n, M = p.shape; stop = np.full(n, M - 1); dec = np.zeros(n, bool)
    for s in range(M - 1):
        take = ~dec & (p[:, s] >= taus[s]); stop[take] = s; dec |= take
    return stop


def fit_stage_thresholds(p, c, lam, passes=2, cum=None, return_obj=False):
    cum = P.CUMCOST if cum is None else cum
    M = p.shape[1]; rows = np.arange(len(p))
    def obj(t):
        st = stop_stage(p, t); return c[rows, st].mean() - cum[st].mean() / lam
    # start from the best GLOBAL threshold, so per-stage is never worse than global on train
    taus = np.full(M - 1, GRID[int(np.argmax([obj(np.full(M - 1, g)) for g in GRID]))])
    for _ in range(passes):
        for s in range(M - 1):
            vals = [obj(np.r_[taus[:s], g, taus[s + 1:]]) for g in GRID]
            taus[s] = GRID[int(np.argmax(vals))]
    return (taus, obj(taus)) if return_obj else taus


def fit_global_threshold(p, c, lam):
    """Single threshold for all stages, chosen on TRAIN for this lambda."""
    M = p.shape[1]; rows = np.arange(len(p))
    vals = [c[rows, stop_stage(p, np.full(M - 1, g))].mean()
            - P.CUMCOST[stop_stage(p, np.full(M - 1, g))].mean() / lam for g in GRID]
    return np.full(M - 1, GRID[int(np.argmax(vals))])


def subset_cascade(tr, te, p_tr, p_te, costs, lam, max_size=3):
    """Choose WHICH models to cascade (<= max_size, cost-ordered) and their per-stage
    thresholds on TRAIN; return per-question (cost, correct) on TEST.  Only models
    actually run are paid for, so skipping sizes is allowed."""
    from itertools import combinations
    M = len(costs); best = (-np.inf, None, None)
    for k in range(1, max_size + 1):
        for S in combinations(range(M), k):
            S = list(S); cum = np.cumsum(costs[S])
            if k == 1:
                taus, ob = np.array([]), tr["correct"][:, S[0]].mean() - cum[0] / lam
            else:
                taus, ob = fit_stage_thresholds(p_tr[:, S], tr["correct"][:, S], lam, cum=cum, return_obj=True)
            if ob > best[0]:
                best = (ob, S, taus)
    _, S, taus = best; cum = np.cumsum(costs[S])
    st = stop_stage(p_te[:, S], taus) if len(S) > 1 else np.zeros(len(p_te), int)
    return cum[st].mean(), te["correct"][np.arange(len(p_te)), np.array(S)[st]].mean(), S


# ---------------------------------------------------------------- learned deferral
class LearnedDeferral:
    """P(model t correct | calibrated confs up to s, category) with gradient boosting,
    then the same lookahead decision as VoC-Stop (continue iff best reachable
    stage's predicted accuracy minus extra cost/lambda beats the current one)."""
    def __init__(self, tr, p_tr):
        self.M = p_tr.shape[1]; self.ncat = len(tr["cat_names"]); self.clf = {}
        for s in range(self.M - 1):
            X = self._feat(p_tr, tr["cat"], s)
            for t in range(s, self.M):
                y = tr["correct"][:, t]
                self.clf[(s, t)] = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                                  max_leaf_nodes=15, random_state=0).fit(X, y)

    def _feat(self, p, cat, s):
        return np.c_[p[:, :s + 1], np.eye(self.ncat)[cat]]

    def run(self, te, p_te, lam):
        n = len(p_te); stop = np.full(n, self.M - 1); alive = np.ones(n, bool)
        for s in range(self.M - 1):
            X = self._feat(p_te, te["cat"], s)
            pr = {t: self.clf[(s, t)].predict_proba(X)[:, 1] for t in range(s, self.M)}
            cont = np.max([pr[t] - (P.CUMCOST[t] - P.CUMCOST[s]) / lam for t in range(s + 1, self.M)], 0)
            halt = alive & (cont <= pr[s]); stop[halt] = s; alive &= ~halt
        return stop


# ---------------------------------------------------------------- evaluation
def frontier(te, stops):
    rows = np.arange(te["correct"].shape[0])
    return (np.array([P.CUMCOST[s].mean() for s in stops]),
            np.array([te["correct"][rows, s].mean() for s in stops]))


def run(name, tasks="all", n_splits=5, bundle=None, tag="e1", verbose=True,
        n_train=None, shift=False, ablation=False):
    """bundle=(data, costs, names) lets callers pass transformed data (E2).
    n_train : subsample the train half to this many labelled questions (E3).
    shift   : train on MMLU, test on ARC+Winogrande, category info removed (E4).
    ablation: add VoC-Stop ingredient ablations (E5)."""
    data, costs, names = bundle if bundle is not None else load_cascade(name, tasks)
    if shift:                                     # categories are meaningless across the shift
        data = {**data, "cat": np.zeros_like(data["cat"])}
    P.set_costs(costs)
    lo, hi = P.CUMCOST[0], P.CUMCOST[-1]
    LAM = hi * np.logspace(-2.5, 2.5, 30)
    acc_big = data["correct"][:, -1].mean()
    res = {}
    t0 = time.time()
    for seed in range(n_splits):
        if shift:
            mm = np.char.startswith(data["task"].astype(str), "hendrycksTest")
            pick = lambda ix: {k: (v[ix] if hasattr(v, "shape") and v.shape[:1] == mm.shape else v)
                               for k, v in data.items()}
            rng = np.random.default_rng(seed)             # vary the train subsample across seeds
            tr_ix = np.flatnonzero(mm); tr_ix = rng.choice(tr_ix, int(0.8 * len(tr_ix)), replace=False)
            tr, te = pick(tr_ix), pick(np.flatnonzero(~mm))
        else:
            tr, te = split(data, seed)
        if n_train is not None:
            ix = np.random.default_rng(1000 + seed).choice(len(tr["cat"]), n_train, replace=False)
            tr = {k: (v[ix] if hasattr(v, "shape") and v.shape[:1] == tr["cat"].shape else v)
                  for k, v in tr.items()}
        st = P.FittedStats(tr); p_tr, p_te = st.p_hat(tr["conf"]), st.p_hat(te["conf"])
        fr = {}
        fr["Global threshold"] = frontier(te, [stop_stage(p_te, fit_global_threshold(p_tr, tr["correct"], l))
                                               for l in LAM])
        fr["Per-stage threshold"] = frontier(te, [stop_stage(p_te, fit_stage_thresholds(p_tr, tr["correct"], l))
                                                  for l in LAM])
        stops = []
        for l in LAM:
            s_ = np.zeros(len(p_te), int)
            for k in np.unique(tr["cat"]):
                mtr, mte = tr["cat"] == k, te["cat"] == k
                s_[mte] = stop_stage(p_te[mte], fit_stage_thresholds(p_tr[mtr], tr["correct"][mtr], l))
            stops.append(s_)
        fr["Per-stage+cat threshold"] = frontier(te, stops)
        la = P.VoCStopLA(tr, st)
        fr["VoC-Stop (paper)"] = frontier(te, [_voc_stop_stage(la, te, l) for l in LAM])
        ld = LearnedDeferral(tr, p_tr)
        fr["Learned deferral (GBM)"] = frontier(te, [ld.run(te, p_te, l) for l in LAM])
        sub = [subset_cascade(tr, te, p_tr, p_te, costs, l) for l in LAM[::2]]
        fr["Subset cascade (<=3 models)"] = (np.array([x[0] for x in sub]), np.array([x[1] for x in sub]))
        if seed == 0 and verbose:
            print("   subsets chosen (low->high budget):", [tuple(x[2]) for x in sub], flush=True)
        if ablation:
            raw = te["conf"]; raw_tr = tr["conf"]
            fr["abl: raw conf, global thr"] = frontier(te, [stop_stage(raw, fit_global_threshold(raw_tr, tr["correct"], l))
                                                           for l in LAM])
            fr["abl: VoC myopic (1-step)"] = frontier(te, [_voc_myopic_stage(st, te, p_te, l) for l in LAM])
            nocat = lambda d: {**d, "cat": np.zeros_like(d["cat"])}
            la0 = P.VoCStopLA(nocat(tr), st)
            fr["abl: VoC lookahead, no cat"] = frontier(te, [_voc_stop_stage(la0, nocat(te), l) for l in LAM])
        rows = np.arange(te["correct"].shape[0])
        fr["Oracle"] = frontier(te, [(te["correct"] * l - P.CUMCOST[None, :]).argmax(1) for l in LAM])
        # single models: cost = that model ALONE (no cascade), not cumulative
        sm_c, sm_a = costs.copy(), te["correct"].mean(0)
        for k, (c, a) in fr.items():
            r = res.setdefault(k, {"aucc": [], "cost_match99": [], "cost_match100": []})
            r["aucc"].append(aucc(c, a, lo, hi))
            for key, q in (("cost_match99", 0.99), ("cost_match100", 1.0)):
                v = cost_at_accuracy(c, a, q * sm_a[-1])
                r[key].append(None if v is None else 100 * v / costs[-1])
        r = res.setdefault("Single models (no cascade)", {"aucc": [], "cost_match99": [], "cost_match100": []})
        r["aucc"].append(aucc(sm_c, sm_a, lo, hi))
        r["cost_match99"].append(100.0 if sm_a.max() >= 0.99 * sm_a[-1] else None)
        r["cost_match100"].append(100.0)
        if verbose:
            print(f"  split {seed} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- report
    if not verbose:                               # E2 calls: return numbers, write no files
        return {k: dict(aucc=r["aucc"], cost_match99=r["cost_match99"]) for k, r in res.items()}
    print(f"\n=== {name} / {tasks}: n={data['correct'].shape[0]}, models={names}")
    print(f"    single-model acc: {np.round(data['correct'].mean(0), 3)}   costs: {costs}")
    print(f"{'method':28s} {'AUCC mean±sd':>16s}  {'dAUCC vs per-stage':>20s}  "
          f"{'cost to 99% of largest':>24s}  {'cost to 100%':>14s}")
    base = np.array(res["Per-stage threshold"]["aucc"])
    summary = {}
    for k, r in res.items():
        a = np.array(r["aucc"]); d = a - base
        f = lambda xs: "n/a" if any(x is None for x in xs) else f"{np.mean(xs):5.1f}% ±{np.std(xs):4.1f}"
        print(f"{k:28s} {a.mean():.4f} ±{a.std():.4f}  {d.mean():+.4f} ({(d > 0).sum()}/{len(d)} wins)  "
              f"{f(r['cost_match99']):>24s}  {f(r['cost_match100']):>14s}")
        summary[k] = dict(aucc=r["aucc"], d_vs_perstage=d.tolist(),
                          cost_match99=r["cost_match99"], cost_match100=r["cost_match100"])
    with open(os.path.join(OUT, f"{tag}_{name}_{tasks}.json"), "w") as fh:
        json.dump(dict(cascade=name, tasks=tasks, models=names, costs=costs.tolist(),
                       n=int(data["correct"].shape[0]), n_splits=n_splits, summary=summary), fh, indent=1)
    return summary


def _voc_myopic_stage(st, te, p, lam):
    """Per-question stop stage of the paper's myopic VoC-1 rule (policies.voc_stop)."""
    cat = te["cat"]; n, M = p.shape; stop = np.full(n, M - 1); alive = np.ones(n, bool)
    for s in range(M - 1):
        gain = (1 - p[:, s]) * st.u[s, cat] - p[:, s] * st.v[s, cat]
        halt = alive & ~(gain * lam > P.MARGINAL_COST[s]); stop[halt] = s; alive &= ~halt
    return stop


def _voc_stop_stage(la, te, lam):
    """Per-question stop stage of VoC-Stop (policies.VoCStopLA.run returns only aggregates)."""
    p = la.stats.p_hat(te["conf"]); cat = te["cat"]; n = len(p)
    stop = np.full(n, la.M - 1); alive = np.ones(n, bool)
    for s in range(la.M - 1):
        cont = np.full(n, -np.inf)
        for t in range(s + 1, la.M):
            cont = np.maximum(cont, la._gpred(s, t, cat, p[:, s]) - (P.CUMCOST[t] - P.CUMCOST[s]) / lam)
        halt = alive & (cont <= p[:, s]); stop[halt] = s; alive &= ~halt
    return stop


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "qwen1.5",
        sys.argv[2] if len(sys.argv) > 2 else "all",
        int(sys.argv[3]) if len(sys.argv) > 3 else 5)
