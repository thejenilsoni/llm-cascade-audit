"""
Item-Response-Theory (IRT) data generator.

We synthesize a controlled but realistic testbed of (item x model) outcomes:

  * Each item i has a latent difficulty b_i and belongs to a category.
  * Each model m has a latent ability theta_m, calibrated by bisection so that
    its *marginal* accuracy matches the published value in calibration_table.
  * P(correct | model m, item i) = sigmoid( gain_cat * (theta_m - b_i) ).

Because b_i is SHARED across models, correctness is positively correlated
across the cascade (hard items are hard for everyone) -- the structure that
makes cascades and stopping non-trivial. Category-dependent `gain` makes the
*value of escalation* instance-dependent, which is the phenomenon VoC-Stop
exploits.

Outputs a dict of numpy arrays:
  correct[i, m]  in {0,1}
  conf[i, m]     in (0,1)   raw (miscalibrated) confidence from model m on item i
  p_true[i, m]   in (0,1)   ground-truth P(correct) (oracle; for analysis only)
  cat[i]         category index
  diff[i]        item difficulty
"""
from __future__ import annotations
import numpy as np
from calibration_table import (
    MODEL_ACC, N_MODELS, CAT_NAMES, CAT_WEIGHTS, CAT_GAIN, CAT_DIFF,
    CONF_NOISE_SD, CONF_OVERCONF_GAIN, SEED,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _calibrate_abilities(rng, n_cal=200_000):
    """Find theta_m so marginal accuracy matches MODEL_ACC, given the category
    mixture and per-category gains. Bisection on a large Monte-Carlo sample."""
    cat_idx = rng.choice(len(CAT_NAMES), size=n_cal, p=CAT_WEIGHTS)
    b = rng.normal(0.0, 1.0, size=n_cal) + CAT_DIFF[cat_idx]
    gain = CAT_GAIN[cat_idx]
    thetas = np.zeros(N_MODELS)
    for m in range(N_MODELS):
        lo, hi = -6.0, 6.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            acc = _sigmoid(gain * (mid - b)).mean()
            if acc < MODEL_ACC[m]:
                lo = mid
            else:
                hi = mid
        thetas[m] = 0.5 * (lo + hi)
    return thetas


def generate(n_items=24_000, seed=SEED):
    rng = np.random.default_rng(seed)
    thetas = _calibrate_abilities(rng)

    cat = rng.choice(len(CAT_NAMES), size=n_items, p=CAT_WEIGHTS)
    diff = rng.normal(0.0, 1.0, size=n_items) + CAT_DIFF[cat]
    gain = CAT_GAIN[cat]

    p_true = np.empty((n_items, N_MODELS))
    for m in range(N_MODELS):
        p_true[:, m] = _sigmoid(gain * (thetas[m] - diff))

    correct = (rng.random((n_items, N_MODELS)) < p_true).astype(int)

    # Raw confidence: noisy, over-confident transform of true correctness prob.
    z = _logit(p_true) * CONF_OVERCONF_GAIN + rng.normal(0, CONF_NOISE_SD, p_true.shape)
    conf = _sigmoid(z)

    return dict(correct=correct, conf=conf, p_true=p_true,
                cat=cat, diff=diff, thetas=thetas,
                cat_names=CAT_NAMES)


def train_test_split(data, frac_train=0.5, seed=SEED):
    """Generic split: any array whose first axis matches the item count is
    split by row; everything else (e.g. cat_names, thetas) is passed through.
    Works for both the cascade and debate data dicts."""
    rng = np.random.default_rng(seed + 1)
    n = data["correct"].shape[0]
    idx = rng.permutation(n)
    k = int(frac_train * n)
    tr, te = idx[:k], idx[k:]
    out = {}
    for split, ix in (("train", tr), ("test", te)):
        d = {}
        for key, val in data.items():
            if hasattr(val, "shape") and getattr(val, "ndim", 0) >= 1 and val.shape[0] == n:
                d[key] = val[ix]
            else:
                d[key] = val
        out[split] = d
    return out["train"], out["test"]


if __name__ == "__main__":
    d = generate()
    print("items:", d["correct"].shape[0])
    print("marginal acc (realized vs target):")
    for m, name in enumerate(__import__("calibration_table").MODEL_NAMES):
        print(f"  {name:14s} {d['correct'][:,m].mean():.3f}  target {MODEL_ACC[m]:.3f}")
    # quick AUROC of raw confidence vs correctness for the cheapest model
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(d["correct"][:,0], d["conf"][:,0])
    print(f"conf AUROC (Mistral-7B): {auc:.3f}")
