"""
Section 6 check: inside the original simulator, how much of VoC-Stop's saving
survives when the global-threshold baseline is replaced by per-stage thresholds?

Prints the cost to reach the GPT-3.5 accuracy operating point for
  calibrated global threshold (the original baseline),
  per-stage thresholds (tuned on train by coordinate ascent),
  VoC-Stop,
and VoC-Stop's saving against each baseline.   Usage: python sim_baseline_check.py
"""
from __future__ import annotations
import numpy as np

import calibration_table as CT
import data_gen
import policies as P
from experiments import cost_at_accuracy

GRID = np.linspace(0.0, 1.0, 41)
LAM = np.logspace(-3.5, 1.5, 100)


def stop_stage(p, taus):
    n, M = p.shape; stop = np.full(n, M - 1); dec = np.zeros(n, bool)
    for s in range(M - 1):
        take = ~dec & (p[:, s] >= taus[s]); stop[take] = s; dec |= take
    return stop


def fit_stage_thresholds(p, c, lam, passes=3):
    M = p.shape[1]; taus = np.full(M - 1, 0.5); rows = np.arange(len(p))
    def obj(t):
        st = stop_stage(p, t); return c[rows, st].mean() - P.CUMCOST[st].mean() / lam
    for _ in range(passes):
        for s in range(M - 1):
            taus[s] = max(GRID, key=lambda g: obj(np.r_[taus[:s], g, taus[s + 1:]]))
    return taus


def frontier(results):
    return np.array([r[1] for r in results]), np.array([r[0] for r in results])


if __name__ == "__main__":
    P.set_costs(CT.MODEL_COST)
    tr, te = data_gen.train_test_split(data_gen.generate())
    st = P.FittedStats(tr); la = P.VoCStopLA(tr, st)
    p_tr, p_te = st.p_hat(tr["conf"]), st.p_hat(te["conf"])
    target = float(te["correct"].mean(0)[3])          # GPT-3.5 accuracy operating point
    cost = {
        "calibrated global threshold": frontier([P.calibrated_threshold(te, st, t)
                                                 for t in np.linspace(0.001, 0.999, 80)]),
        "per-stage thresholds": frontier([P._evaluate(te, stop_stage(p_te, fit_stage_thresholds(p_tr, tr["correct"], l)))
                                          for l in LAM[::3]]),
        "VoC-Stop": frontier([la.run(te, l) for l in LAM]),
    }
    at = {k: cost_at_accuracy(c, a, target) for k, (c, a) in cost.items()}
    print(f"cost ($/query) to reach accuracy {target:.3f}:")
    for k, v in at.items():
        print(f"  {k:28s} {v:.5f}")
    for base in ("calibrated global threshold", "per-stage thresholds"):
        print(f"VoC-Stop saving vs {base}: {100 * (1 - at['VoC-Stop'] / at[base]):.1f}%")
