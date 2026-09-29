"""
Fetch REAL per-question, per-model log-likelihoods from the public (ungated)
Open LLM Leaderboard v1 archive (HF org `open-llm-leaderboard-old`).

For every (model, task) we keep: question text (for cross-model alignment),
correctness (`metrics.acc`) and the per-option log-likelihood vector
(`predictions`) from which a real confidence signal is derived:
    conf = max softmax(option log-likelihoods)

Output: ../data_real/<model>.parquet  with columns
    task, example_hash, acc, logliks (list[float])

Usage:  python fetch_real.py            (~8-12 MB per model, parallel)
"""
from __future__ import annotations
import hashlib, io, os, re, sys, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
from huggingface_hub import HfApi

ORG = "open-llm-leaderboard-old"
MODELS = [
    # Qwen1.5 base family
    "Qwen__Qwen1.5-0.5B", "Qwen__Qwen1.5-1.8B", "Qwen__Qwen1.5-4B", "Qwen__Qwen1.5-7B",
    "Qwen__Qwen1.5-14B", "Qwen__Qwen1.5-32B", "Qwen__Qwen1.5-72B",
    # Llama-2 base family
    "meta-llama__Llama-2-7b-hf", "meta-llama__Llama-2-13b-hf", "meta-llama__Llama-2-70b-hf",
    # Mistral family
    "mistralai__Mistral-7B-v0.1", "mistralai__Mixtral-8x7B-v0.1", "mistralai__Mixtral-8x22B-v0.1",
]
TASK_RE = re.compile(r"details_harness\|(hendrycksTest-[a-z_]+|arc:challenge|winogrande)\|")
OUT = os.path.join(os.path.dirname(__file__), "..", "data_real")
os.makedirs(OUT, exist_ok=True)
api = HfApi()


def _get(repo, path):
    u = f"https://huggingface.co/datasets/{repo}/resolve/main/" + urllib.parse.quote(path)
    for attempt in range(8):                      # survive flaky connections: back off up to ~4 min
        try:
            r = requests.get(u, timeout=120); r.raise_for_status(); return r.content
        except Exception as e:  # noqa
            err = e; time.sleep(min(60, 2 ** attempt))
    raise err


def fetch_model(model):
    dst = os.path.join(OUT, f"{model}.parquet")
    if os.path.exists(dst):
        return model, "cached"
    repo = f"{ORG}/details_{model}"
    files = [(s.rfilename, s.size or 0) for s in api.dataset_info(repo, files_metadata=True).siblings]
    # task -> file of the most complete run: largest file, ties broken by latest timestamp.
    # ("latest" alone is wrong: some repos end with a partial re-run, e.g. Llama-2-7b.)
    latest, best = {}, {}
    for f, size in files:
        m = TASK_RE.search(f)
        if m and f.endswith(".parquet"):
            t = m.group(1)
            if t not in best or (size, f) > best[t]:
                best[t] = (size, f); latest[t] = f
    tasks = sorted(latest)
    with ThreadPoolExecutor(12) as ex:           # latency-bound: fetch a model's files in parallel
        blobs = list(ex.map(lambda t: _get(repo, latest[t]), tasks))
    rows = []
    for task, blob in zip(tasks, blobs):
        df = pd.read_parquet(io.BytesIO(blob))
        # 2024 logs nest correctness in `metrics`; 2023 logs have a top-level `acc` column
        accs = df["metrics"].map(lambda m: m["acc"]) if "metrics" in df else df["acc"]
        for ex_, acc, pr in zip(df["example"], accs, df["predictions"]):
            # some 2023 runs store acc / predictions as text ("True", "[-1.2, -3.4]")
            acc = {"True": 1.0, "False": 0.0}.get(acc, acc)
            if isinstance(pr, str):
                pr = [float(x) for x in re.split(r"[\s,]+", pr.strip("[] \n")) if x]
            rows.append((task, hashlib.md5(ex_.encode()).hexdigest(), float(acc),
                         [float(x) for x in pr]))
    out = pd.DataFrame(rows, columns=["task", "example_hash", "acc", "logliks"])
    out.to_parquet(dst)
    return model, f"{len(latest)} tasks, {len(out)} rows"


if __name__ == "__main__":
    models = sys.argv[1:] or MODELS
    for m in models:                              # one model at a time; files are parallel
        print(f"{m:36s} {fetch_model(m)[1]}", flush=True)
