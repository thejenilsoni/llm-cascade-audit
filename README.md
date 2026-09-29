# Do LLM Cascades Need Smart Stopping? A Real-Data Audit of Deferral Rules

**Jenil Soni** · Anybody Can AI

An LLM *cascade* sends a query to a cheap model first and escalates to larger
models only when needed; its *deferral rule* decides when to stop. This
repository audits seven deferral rules on **real per-question log-likelihoods of
13 open models** (Qwen1.5 0.5B–72B, Llama-2 7B–70B, Mistral/Mixtral) on 16,481
questions from MMLU, ARC-Challenge and WinoGrande, taken from the public
[Open LLM Leaderboard](https://huggingface.co/open-llm-leaderboard-old) logs.

📄 **Paper:** [`paper/main.pdf`](paper/main.pdf)

## Findings

1. **Per-stage thresholds are the big win.** Moving from one global threshold to
   per-stage thresholds gains 0.8–4.3 AUCC points, more than any more
   sophisticated rule adds on top.
2. **Choosing which models to chain matters most.** A subset cascade (e.g.
   32B → 72B) is best in two of three families, degrades least with few labels
   and survives task shift, reaching 99% of the largest model's accuracy at 74%
   (Qwen) and 81% (Mistral) of its cost.
3. **Sophisticated rules need better confidence.** Adaptive rules overtake simple
   model selection only once confidence AUROC exceeds ~0.9; today's real signals
   are 0.74–0.81.
4. **A value-of-computation lookahead rule (VoC-Stop), which looked strong in our
   earlier simulation, is never the best rule on real data**, and its category
   conditioning hurts in two of three families.

## Reproduce

No GPU, no API keys. About 40 minutes on a desktop PC; the data download is ~15 MB.

**Windows, one click:** double-click `start.bat`. It creates a private Python
environment, downloads the logs, runs every experiment (resumable) and packs
the results into `results_<timestamp>.zip`.

**Any OS:**

```bash
pip install -r requirements.txt
cd src
python run_study.py          # fetch data + experiments E1-E5 -> results_real/
python summarize_real.py     # print the tables
python make_tables.py        # regenerate paper/tables/*.tex
python plots_real.py         # regenerate paper/figures/*
python sim_baseline_check.py # Section 6: the simulator baseline check
```

Compile the paper with `pdflatex main.tex` (or `tectonic main.tex`) in `paper/`.

## What's where

| Path | Contents |
|---|---|
| `src/fetch_real.py` | downloads and compacts the public leaderboard logs |
| `src/real_data.py` | aligns questions across models; confidence signals |
| `src/real_eval.py` | the deferral rules and the evaluation protocol (E1, E3, E4, E5) |
| `src/e2_confidence.py` | confidence-quality sweep (E2) |
| `src/run_study.py` | runs everything unattended; resumes after interruption |
| `src/policies.py`, `data_gen.py`, `calibration_table.py`, `experiments.py` | original simulator and shared rule code |
| `results_real/` | every result file used in the paper |
| `paper/` | LaTeX source, compiled PDF, figures and tables |

## Protocol in one paragraph

Each run splits questions 50/50 into train and test. Confidence calibration and
every rule's parameters are fitted on train only, and all numbers are on test,
averaged over 5 random splits (3 for E2/E3). Cost is proportional to active
parameters. AUCC is the normalised area under the accuracy–cost frontier;
Cost@99 is the cost to reach 99% of the largest model's accuracy, as a
percentage of calling that model alone.

## Citation

```bibtex
@misc{soni2026cascades,
  title  = {Do LLM Cascades Need Smart Stopping? A Real-Data Audit of Deferral Rules},
  author = {Soni, Jenil},
  year   = {2026},
  url    = {https://github.com/thejenilsoni/llm-cascade-audit}
}
```

## License

MIT (see `LICENSE`). The underlying evaluation logs are published by the Open
LLM Leaderboard on the Hugging Face Hub under their own terms.
