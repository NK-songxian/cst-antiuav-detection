"""Drive the 4-group FP study: wave1 G1+G2 parallel, wave2 G3+G4 parallel, then retry failures.

subprocess.Popen.wait() is used instead of bash `wait`, which returned early in the
sandboxed environment and caused all four trainings to start simultaneously.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Ryan/cst_antiuav")
STUDY = ROOT / "runs/fp_study"
STUDY.mkdir(parents=True, exist_ok=True)
LOG = open(STUDY / "driver.log", "a")


def log(msg):
    LOG.write(f"[driver] {msg}\n")
    LOG.flush()
    print(f"[driver] {msg}", flush=True)


def launch(group):
    log(f"{group} start")
    f = open(STUDY / f"{group}.log", "w")
    p = subprocess.Popen(
        [sys.executable, "scripts/train_fp_study.py", "--group", group],
        cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
    )
    return p, f


def wave(groups):
    procs = {g: launch(g) for g in groups}
    for g, (p, f) in procs.items():
        p.wait()
        f.close()
        log(f"{g} exit {p.returncode}")
    return {g: p.returncode for g, (p, _) in procs.items()}


def done(group):
    return (STUDY / group / "weights" / "best.pt").exists()


log("wave 1: G1 + G2")
wave(["G1", "G2"])
log("wave 2: G3 + G4")
wave(["G3", "G4"])
for g in ["G1", "G2", "G3", "G4"]:
    if not done(g):
        log(f"retry {g} sequentially")
        p, f = launch(g)
        p.wait()
        f.close()
        log(f"{g} retry exit {p.returncode}")
ok = [g for g in ["G1", "G2", "G3", "G4"] if done(g)]
log(f"summary: finished={ok}")
if len(ok) == 4:
    (STUDY / "FP_STUDY_DONE").touch()
    log("FP_STUDY_DONE")
else:
    log("INCOMPLETE")
