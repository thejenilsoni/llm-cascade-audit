"""
Load REAL cascade data from the Open LLM Leaderboard v1 logs fetched by fetch_real.py.

Builds the same dict format the policies use:
    correct[i, m]  real correctness of model m on question i
    conf[i, m]     real confidence = max softmax over the option log-likelihoods
    cat[i]         category id (MMLU super-category, ARC, Winogrande)
Questions are aligned across models by (task, question-hash, occurrence index).
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data_real")

# cost proxy = parameters (billions) actually run per token; equal prompts across
# models make per-query cost proportional to this.  Mixtral uses ACTIVE params.
CASCADES = {
    "qwen1.5": [("Qwen__Qwen1.5-0.5B", 0.5), ("Qwen__Qwen1.5-1.8B", 1.8), ("Qwen__Qwen1.5-4B", 4.0),
                ("Qwen__Qwen1.5-7B", 7.0), ("Qwen__Qwen1.5-14B", 14.0), ("Qwen__Qwen1.5-32B", 32.0),
                ("Qwen__Qwen1.5-72B", 72.0)],
    "llama2":  [("meta-llama__Llama-2-7b-hf", 7.0), ("meta-llama__Llama-2-13b-hf", 13.0),
                ("meta-llama__Llama-2-70b-hf", 70.0)],
    "mistral": [("mistralai__Mistral-7B-v0.1", 7.3), ("mistralai__Mixtral-8x7B-v0.1", 12.9),
                ("mistralai__Mixtral-8x22B-v0.1", 39.0)],
}

# MMLU subject -> super-category (standard grouping from the MMLU release)
_STEM = """abstract_algebra anatomy astronomy college_biology college_chemistry college_computer_science
college_mathematics college_physics computer_security conceptual_physics electrical_engineering
elementary_mathematics high_school_biology high_school_chemistry high_school_computer_science
high_school_mathematics high_school_physics high_school_statistics machine_learning""".split()
_HUM = """formal_logic high_school_european_history high_school_us_history high_school_world_history
international_law jurisprudence logical_fallacies moral_disputes moral_scenarios philosophy prehistory
professional_law world_religions""".split()
_SOC = """econometrics high_school_geography high_school_government_and_politics high_school_macroeconomics
high_school_microeconomics high_school_psychology human_sexuality professional_psychology public_relations
security_studies sociology us_foreign_policy""".split()
CAT_NAMES = ["mmlu-stem", "mmlu-humanities", "mmlu-social", "mmlu-other", "arc", "winogrande"]


def task_category(task: str) -> int:
    if task == "arc:challenge":
        return 4
    if task == "winogrande":
        return 5
    subj = task.replace("hendrycksTest-", "")
    if subj in _STEM: return 0
    if subj in _HUM:  return 1
    if subj in _SOC:  return 2
    return 3


def _probs(ll):
    ll = np.asarray(ll, dtype=float); e = np.exp(ll - ll.max()); return e / e.sum()


def _signals(ll):
    """Real confidence signals from one question's option log-likelihoods."""
    p = _probs(ll); q = np.sort(p)[::-1]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(p))
    return q[0], q[0] - q[1], 1.0 - ent, int(p.argmax())


def load_cascade(name: str, tasks: str = "all", conf_type: str = "maxprob"):
    """tasks: 'all' | 'mmlu' | 'arc' | 'winogrande'.
    conf_type: 'maxprob' (top option prob) | 'margin' (top1 - top2) | 'entropy' (1 - normalised
    entropy) | 'agree' (geometric mean of this model's top prob and the probability the PREVIOUS
    model in the cascade gave to this model's answer; stage 0 falls back to maxprob)."""
    models, costs = zip(*CASCADES[name])
    frames = []
    for m in models:
        d = pd.read_parquet(os.path.join(DATA, f"{m}.parquet"))
        d["occ"] = d.groupby(["task", "example_hash"]).cumcount()   # disambiguate duplicate questions
        sig = np.array([_signals(x) for x in d["logliks"]])
        d["maxprob"], d["margin"], d["entropy"], d["pred"] = sig[:, 0], sig[:, 1], sig[:, 2], sig[:, 3].astype(int)
        d["probs"] = d["logliks"].map(_probs)
        frames.append(d.set_index(["task", "example_hash", "occ"])[["acc", "maxprob", "margin", "entropy", "pred", "probs"]]
                      .rename(columns=lambda c: f"{c}|{m}"))
    df = pd.concat(frames, axis=1, join="inner").reset_index()     # questions every model answered
    if tasks == "mmlu":
        df = df[df.task.str.startswith("hendrycksTest")]
    elif tasks == "arc":
        df = df[df.task == "arc:challenge"]
    elif tasks == "winogrande":
        df = df[df.task == "winogrande"]
    correct = np.stack([df[f"acc|{m}"].to_numpy() for m in models], 1).astype(int)
    if conf_type == "agree":
        cols = [df[f"maxprob|{models[0]}"].to_numpy()]
        for a, b in zip(models[:-1], models[1:]):
            prev_on_mine = np.array([pa[k] for pa, k in zip(df[f"probs|{a}"], df[f"pred|{b}"])])
            cols.append(np.sqrt(df[f"maxprob|{b}"].to_numpy() * prev_on_mine))
        conf = np.stack(cols, 1)
    else:
        conf = np.stack([df[f"{conf_type}|{m}"].to_numpy() for m in models], 1)
    cat = df.task.map(task_category).to_numpy()
    return dict(correct=correct, conf=conf, cat=cat, task=df.task.to_numpy(),
                cat_names=CAT_NAMES), np.asarray(costs, float), [m.split("__")[1] for m in models]


if __name__ == "__main__":
    import sys
    from sklearn.metrics import roc_auc_score
    for name in sys.argv[1:] or ["qwen1.5"]:
        data, costs, names = load_cascade(name)
        print(f"\n{name}: {data['correct'].shape[0]} aligned questions, {len(names)} models")
        for j, n in enumerate(names):
            print(f"  {n:22s} acc={data['correct'][:, j].mean():.3f}  "
                  f"conf AUROC={roc_auc_score(data['correct'][:, j], data['conf'][:, j]):.3f}  cost={costs[j]}")
        c = data["correct"]
        print("  corr(correct) small vs large:", np.round(np.corrcoef(c[:, 0], c[:, -1])[0, 1], 3),
              "| P(large right | small wrong):", np.round(c[c[:, 0] == 0, -1].mean(), 3),
              "| P(large wrong | small right):", np.round(1 - c[c[:, 0] == 1, -1].mean(), 3))
