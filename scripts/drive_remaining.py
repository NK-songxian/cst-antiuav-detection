"""Serial chain for remaining FP-study groups, fully detached via setsid.

Skips groups whose best.pt already exists. Runs one training at a time to stay
within GPU memory alongside the already-running G3 process.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Ryan/cst_antiuav")
STUDY = ROOT / "runs/fp_study"
GROUPS = ["G1", "G2", "G4"]


def log(msg):
    with open(STUDY / "chain.log", "a") as f:
        f.write(f"[chain] {msg} {time.strftime('%F %T')}\n")


def finished(group):
    return (STUDY / group / "weights" / "best.pt").exists()


for g in GROUPS:
    if finished(g):
        log(f"{g} already done, skip")
        continue
    log(f"{g} start")
    with open(STUDY / f"{g}.log", "a") as f:
        p = subprocess.run(
            [sys.executable, "scripts/train_fp_study.py", "--group", g],
            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
        )
    log(f"{g} exit {p.returncode}")
    if p.returncode != 0:
        # wait for GPU headroom then retry once
        time.sleep(120)
        with open(STUDY / f"{g}.log", "a") as f:
            p2 = subprocess.run(
                [sys.executable, "scripts/train_fp_study.py", "--group", g],
                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
            )
        log(f"{g} retry exit {p2.returncode}")

done = [g for g in GROUPS if finished(g)]
log(f"chain summary: {done} + G3={'done' if finished('G3') else 'running/pending'}")
if all(finished(g) for g in GROUPS) and finished("G3"):
    (STUDY / "FP_STUDY_DONE").touch()
    log("FP_STUDY_DONE")
