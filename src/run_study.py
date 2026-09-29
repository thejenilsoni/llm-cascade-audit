"""
Run the whole real-data study unattended (used by start.bat in the portable zip).

  * each step runs in its own process; a failed step is retried, then skipped
  * finished steps are recorded in results_real/.done/ -> re-running resumes
  * everything is logged to logs/study.log
  * the PC is kept awake while this runs (Windows only; released on exit)
  * at the end all results + logs are packed into results_<timestamp>.zip
"""
from __future__ import annotations
import datetime, os, subprocess, sys, time, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_real"); DONE = os.path.join(RES, ".done")
LOGS = os.path.join(ROOT, "logs")
for d in (RES, DONE, LOGS):
    os.makedirs(d, exist_ok=True)
LOG = open(os.path.join(LOGS, "study.log"), "a", encoding="utf-8")
CASCADES = ["qwen1.5", "llama2", "mistral"]


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


def keep_awake(on=True):
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def py(code):
    return [sys.executable, "-c", f"import real_eval as E; {code}"]


def steps():
    s = [("fetch_data", [sys.executable, "fetch_real.py"])]
    for c in CASCADES:
        s.append((f"e1_{c}", py(f"E.run('{c}', 'all', 5, tag='e1')")))
    s.append(("e1_qwen1.5_mmlu", py("E.run('qwen1.5', 'mmlu', 5, tag='e1')")))
    for c in CASCADES:
        s.append((f"e2_{c}", [sys.executable, "e2_confidence.py", c, "3"]))
    for c in CASCADES:
        for n in (100, 300, 1000, 3000):
            s.append((f"e3_{c}_n{n}", py(f"E.run('{c}', 'all', 3, tag='e3n{n}', n_train={n})")))
    for c in CASCADES:
        s.append((f"e4_{c}", py(f"E.run('{c}', 'all', 5, tag='e4shift', shift=True)")))
    for c in CASCADES:
        s.append((f"e5_{c}", py(f"E.run('{c}', 'all', 5, tag='e5abl', ablation=True)")))
    return s


def run_step(name, cmd, retries=2):
    marker = os.path.join(DONE, name)
    if os.path.exists(marker):
        log(f"skip  {name} (already done)"); return True
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    for attempt in range(1, retries + 2):
        log(f"start {name} (attempt {attempt})"); t0 = time.time()
        with open(os.path.join(LOGS, f"{name}.log"), "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=env)
        if rc == 0:
            open(marker, "w").close()
            log(f"done  {name} in {(time.time() - t0) / 60:.1f} min"); return True
        log(f"FAIL  {name} (exit {rc}) - see logs/{name}.log")
        time.sleep(30)
    return False


def pack():
    out = os.path.join(ROOT, f"results_{datetime.datetime.now():%Y%m%d_%H%M}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in (RES, LOGS):
            for dp, _, files in os.walk(folder):
                for f in files:
                    p = os.path.join(dp, f); z.write(p, os.path.relpath(p, ROOT))
    return out


if __name__ == "__main__":
    keep_awake(True)
    log("=== study started ===")
    failed = [n for n, c in steps() if not run_step(n, c)]
    out = pack()
    log(f"=== finished. failed steps: {failed or 'none'}")
    log(f"=== bring back this file: {out}")
    keep_awake(False)
