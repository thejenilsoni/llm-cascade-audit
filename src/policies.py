"""
Stopping / routing policies for an LLM cascade.

A cascade queries models in cost order m=0..M-1. After observing model m's
answer (and a confidence signal), we decide STOP (accept m's answer) or
ESCALATE (pay for m+1). Cost = sum of costs of all models queried; accuracy =
correctness of the last model queried.

Policies implemented
--------------------
  single_model(m)                 fixed model, no escalation        (anchor pts)
  conf_threshold(tau)             escalate while RAW conf < tau     (FrugalGPT-ish)
  calibrated_threshold(tau)       escalate while CALIBRATED p < tau (ablation)
  voc_stop(lmbda)                 OUR rule: escalate iff instance-conditional
                                  expected accuracy gain * lambda > marginal cost
  oracle_stop(lmbda)              same rule with TRUE gains (upper bound)

The VoC-Stop continuation value (escalating from stage s to s+1) is

    G_s(x) = (1 - p_hat_s) * u[s,c]  -  p_hat_s * v[s,c]

  p_hat_s : CALIBRATED P(model s correct) for this item  (isotonic-recalibrated)
  u[s,c]  : P(model s+1 correct | model s wrong,  stage s, category c)  "recovery"
  v[s,c]  : P(model s+1 wrong   | model s right,  stage s, category c)  "breakage"

and we escalate iff  G_s(x) * lambda > marginal_cost(s -> s+1).
This is a value-of-computation / optimal-stopping index rule; conditioning
G on (category, calibrated confidence) is what a scalar threshold cannot do.
"""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from calibration_table import MODEL_COST

# Stage costs are swappable so the SAME policies serve both settings:
#   cascade  -> per-model query cost (MODEL_COST)
#   debate   -> per-round cost (linear: r rounds cost r * c_round)
# `costs[s]` = incremental price incurred at stage s. Total cost to STOP at s is
# CUMCOST[s] = sum_{k<=s} costs[k]; the cost ADDED by escalating s -> s+1 is
# MARGINAL_COST[s] = CUMCOST[s+1] - CUMCOST[s] = costs[s+1].
CUMCOST = np.cumsum(MODEL_COST)                # cumulative cost if we stop at s
MARGINAL_COST = np.diff(CUMCOST)              # cost of escalating stage s -> s+1


def set_costs(costs):
    """Swap the global stage-cost vector (cascade vs debate)."""
    global MARGINAL_COST, CUMCOST
    costs = np.asarray(costs, dtype=float)
    CUMCOST = np.cumsum(costs)
    MARGINAL_COST = np.diff(CUMCOST)


# ----------------------------------------------------------------------------
# Fitting the calibrators and value tables on TRAIN data
# ----------------------------------------------------------------------------
class FittedStats:
    def __init__(self, train):
        c = train["correct"]; conf = train["conf"]; cat = train["cat"]
        M = c.shape[1]; n_cat = int(cat.max()) + 1
        self.M = M; self.n_cat = n_cat
        # per-stage isotonic recalibration: raw conf -> P(correct)
        self.iso = []
        for s in range(M):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(conf[:, s], c[:, s])
            self.iso.append(ir)
        # per (stage, category) recovery u and breakage v rates
        self.u = np.zeros((M - 1, n_cat))
        self.v = np.zeros((M - 1, n_cat))
        for s in range(M - 1):
            for k in range(n_cat):
                mask = cat == k
                cs, cs1 = c[mask, s], c[mask, s + 1]
                wrong = cs == 0; right = cs == 1
                self.u[s, k] = cs1[wrong].mean() if wrong.any() else 0.0
                self.v[s, k] = (1 - cs1[right]).mean() if right.any() else 0.0

    def p_hat(self, conf):
        out = np.empty_like(conf, dtype=float)
        for s in range(self.M):
            out[:, s] = self.iso[s].predict(conf[:, s])
        return out


# ----------------------------------------------------------------------------
# Cascade evaluation given a per-item stop stage
# ----------------------------------------------------------------------------
def _evaluate(data, stop_stage):
    """stop_stage[i] in 0..M-1 (causal). Returns (mean_acc, mean_cost,
    per-item correct, per-item cost)."""
    idx = np.arange(data["correct"].shape[0])
    correct = data["correct"][idx, stop_stage]
    cost = CUMCOST[stop_stage]
    return correct.mean(), cost.mean(), correct, cost


# --- policies return a per-item stop stage (causal) --------------------------
def single_model(data, m):
    n = data["correct"].shape[0]
    return _evaluate(data, np.full(n, m, dtype=int))


def conf_threshold(data, tau):
    """Escalate while RAW confidence < tau (stop at first conf>=tau)."""
    conf = data["conf"]
    n, M = conf.shape
    stop = np.full(n, M - 1, dtype=int)
    decided = np.zeros(n, dtype=bool)
    for s in range(M - 1):
        take = (~decided) & (conf[:, s] >= tau)
        stop[take] = s; decided |= take
    return _evaluate(data, stop)


def calibrated_threshold(data, stats, tau):
    """Ablation: escalate while CALIBRATED p_hat < tau."""
    p = stats.p_hat(data["conf"])
    n, M = p.shape
    stop = np.full(n, M - 1, dtype=int)
    decided = np.zeros(n, dtype=bool)
    for s in range(M - 1):
        take = (~decided) & (p[:, s] >= tau)
        stop[take] = s; decided |= take
    return _evaluate(data, stop)


def voc_stop(data, stats, lmbda):
    """OUR rule. Escalate s->s+1 iff conditional expected gain * lambda > cost."""
    p = stats.p_hat(data["conf"]); cat = data["cat"]
    n, M = p.shape
    stop = np.full(n, M - 1, dtype=int)
    alive = np.ones(n, dtype=bool)           # still at current stage, may escalate
    cur = np.zeros(n, dtype=int)             # current stage per item
    for s in range(M - 1):
        at_s = alive & (cur == s)
        if not at_s.any():
            continue
        u = stats.u[s, cat]; v = stats.v[s, cat]
        gain = (1 - p[:, s]) * u - p[:, s] * v
        escalate = at_s & (gain * lmbda > MARGINAL_COST[s])
        stop_here = at_s & ~escalate
        stop[stop_here] = s; alive[stop_here] = False
        cur[escalate] = s + 1
    return _evaluate(data, stop)


class VoCStopDP:
    """OUR main method: finite-horizon optimal stopping solved by empirical
    backward induction over calibrated-confidence states.

    State at stage s = (category c, bin of calibrated p_hat_s). Decisions:
        STOP  -> reward = accuracy = E[correct_s | state]
        CONT  -> pay marginal_cost(s), transition to a stage-(s+1) state drawn
                 from the empirical conditional distribution, then act optimally.
    We maximize  accuracy - (1/lambda) * cost ; sweeping lambda traces the
    frontier. Continuation value includes reaching far downstream models, so
    (unlike the myopic one-step rule) the policy will escalate genuinely hard
    items all the way to the strongest model when lambda is large.
    """
    def __init__(self, train, stats, n_bins=10):
        self.stats = stats; self.K = n_bins
        p = stats.p_hat(train["conf"]); self.cat = train["cat"]
        c = train["correct"]
        M = c.shape[1]; n_cat = int(self.cat.max()) + 1
        self.M = M; self.n_cat = n_cat
        # per-stage quantile bin edges on calibrated p_hat
        self.edges = []
        for s in range(M):
            q = np.quantile(p[:, s], np.linspace(0, 1, n_bins + 1))
            q[0], q[-1] = -np.inf, np.inf
            self.edges.append(q)
        self.binned = np.stack([np.clip(np.digitize(p[:, s], self.edges[s][1:-1]), 0, n_bins - 1)
                                for s in range(M)], axis=1)  # [n, M]
        # acc_stop[s,c,k] = mean correctness of train items in that cell
        self.acc_stop = np.full((M, n_cat, n_bins), np.nan)
        for s in range(M):
            for cc in range(n_cat):
                m = self.cat == cc
                for k in range(n_bins):
                    sel = m & (self.binned[:, s] == k)
                    if sel.any():
                        self.acc_stop[s, cc, k] = c[sel, s].mean()
        # fill empty cells with stage marginal
        for s in range(M):
            stage_mean = c[:, s].mean()
            self.acc_stop[s] = np.where(np.isnan(self.acc_stop[s]), stage_mean, self.acc_stop[s])

    def _decisions(self, lmbda):
        """Backward induction -> continue table [M-1, n_cat, K] (bool)."""
        K = self.K; nc = self.n_cat; M = self.M
        V = self.acc_stop[M - 1].copy()                  # value at last stage (forced stop)
        cont = np.zeros((M - 1, nc, K), dtype=bool)
        for s in range(M - 2, -1, -1):
            # expected V_{s+1} for each (c, bin_s) cell, via train transitions
            nextV_item = V[self.cat, self.binned[:, s + 1]]      # [n]
            cell = self.cat * K + self.binned[:, s]              # [n]
            sumV = np.bincount(cell, weights=nextV_item, minlength=nc * K)
            cnt = np.bincount(cell, minlength=nc * K)
            expV = np.divide(sumV, cnt, out=np.full(nc * K, V.mean()), where=cnt > 0).reshape(nc, K)
            value_cont = -MARGINAL_COST[s] / lmbda + expV
            value_stop = self.acc_stop[s]
            cont[s] = value_cont > value_stop
            V = np.maximum(value_stop, value_cont)
        return cont

    def run(self, test, lmbda):
        cont = self._decisions(lmbda)
        p = self.stats.p_hat(test["conf"]); cat = test["cat"]
        binned = np.stack([np.clip(np.digitize(p[:, s], self.edges[s][1:-1]), 0, self.K - 1)
                           for s in range(self.M)], axis=1)
        n = test["correct"].shape[0]
        stop = np.full(n, self.M - 1, dtype=int)
        alive = np.ones(n, dtype=bool); cur = np.zeros(n, dtype=int)
        for s in range(self.M - 1):
            at_s = alive & (cur == s)
            if not at_s.any():
                continue
            go = cont[s, cat, binned[:, s]]
            escalate = at_s & go
            stop_here = at_s & ~go
            stop[stop_here] = s; alive[stop_here] = False
            cur[escalate] = s + 1
        return _evaluate(test, stop)


class VoCStopLA:
    """OUR main method: continuous receding-horizon value-of-computation stopping.

    Combines the two ingredients that the ablations show each matter:
      * CONTINUOUS calibrated confidence (per-item resolution; the binned DP
        loses this), and
      * multi-step LOOKAHEAD to the *best reachable downstream* stage (the
        myopic one-step rule lacks this).

    For an item at stage s with calibrated confidence p_hat_s and category c:
        stop_value      = p_hat_s                       (E[acc | stop now])
        cont_value      = max_{t>s} [ g_{s,t,c}(p_hat_s) - (C[t]-C[s]) / lambda ]
        CONTINUE iff cont_value > stop_value             (then act one step, re-decide)
    where g_{s,t,c}(p) = E[correct_t | category c, p_hat_s = p], estimated on
    train by isotonic regression (monotone, label-efficient). lambda is the
    cost-accuracy exchange rate; sweeping it traces the frontier.
    """
    def __init__(self, train, stats):
        self.stats = stats
        p = stats.p_hat(train["conf"]); cat = train["cat"]; c = train["correct"]
        self.M = c.shape[1]; self.n_cat = int(cat.max()) + 1
        # g[s][t][cc]: isotonic map p_hat_s -> E[correct_t]   (t > s)
        self.g = {}
        for s in range(self.M):
            for t in range(s + 1, self.M):
                for cc in range(self.n_cat):
                    m = cat == cc
                    if m.sum() >= 20:
                        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                        ir.fit(p[m, s], c[m, t])
                        self.g[(s, t, cc)] = ir

    def _gpred(self, s, t, cc_arr, p_s):
        out = np.empty_like(p_s)
        for cc in range(self.n_cat):
            sel = cc_arr == cc
            if not sel.any():
                continue
            ir = self.g.get((s, t, cc))
            out[sel] = ir.predict(p_s[sel]) if ir is not None else p_s[sel]
        return out

    def run(self, test, lmbda):
        p = self.stats.p_hat(test["conf"]); cat = test["cat"]
        n = test["correct"].shape[0]
        stop = np.full(n, self.M - 1, dtype=int)
        alive = np.ones(n, dtype=bool); cur = np.zeros(n, dtype=int)
        for s in range(self.M - 1):
            at_s = alive & (cur == s)
            if not at_s.any():
                continue
            ps = p[:, s]
            stop_val = ps.copy()
            cont_val = np.full(n, -np.inf)
            for t in range(s + 1, self.M):
                gt = self._gpred(s, t, cat, ps)
                cont_val = np.maximum(cont_val, gt - (CUMCOST[t] - CUMCOST[s]) / lmbda)
            go = at_s & (cont_val > stop_val)
            stop_here = at_s & ~go
            stop[stop_here] = s; alive[stop_here] = False
            cur[go] = s + 1
        return _evaluate(test, stop)


def oracle_stop(data, lmbda):
    """TRUE per-item optimal stopping (clairvoyant upper bound): for each item,
    pick the stop stage s that maximizes  lambda * correct[s] - CUMCOST[s].
    This is the best achievable by ANY stopping policy and is a valid upper
    bound for both the cascade and debate settings (unlike a myopic one-step
    rule, whose value can fall below adaptive policies when gains arrive late)."""
    correct = data["correct"]
    obj = correct.astype(float) * lmbda - CUMCOST[None, :]   # [n, M]
    stop = obj.argmax(1)
    return _evaluate(data, stop)
