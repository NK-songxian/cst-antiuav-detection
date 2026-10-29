"""Dual-process baseline queue for CST-AntiUAV.

Config:  epochs=100  patience=20  batch=256  imgsz=640  workers=16  device=0
         defaults on everything else (YOLO's built-in geometric augs on).
Queue:  YOLOv8n → YOLO11n → YOLO11s
Strategy: wave 1 runs YOLOv8n + YOLO11n in parallel (2 × ~37GB ≈ 74GB on 96GB safe),
          wave 2 runs YOLO11s alone.  Each training has 1 auto-retry on failure.
Output dir: runs/queue_{name}/  (name = yolo8n / yolo11n / yolo11s)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Ryan/cst_antiuav")
STUDY = ROOT / "runs/queue_logs"
STUDY.mkdir(parents=True, exist_ok=True)
QUEUE = [
    dict(name="yolo8n", model=str(ROOT / "yolov8n.pt")),
    dict(name="yolo11n", model=str(ROOT / "yolo11n.pt")),
    dict(name="yolo11s", model=str(ROOT / "yolo11s.pt")),
]


def log(msg):
    ts = time.strftime("%F %T")
    with open(STUDY / "queue.log", "a") as f:
        f.write(f"[queue {ts}] {msg}\n")
    print(f"[queue] {msg}", flush=True)


def finished(name):
    return (ROOT / "runs" / name / "weights" / "best.pt").exists()


def run_once(job):
    name, model = job["name"], job["model"]
    log(f"{name} start")
    out = open(STUDY / f"{name}.log", "w")
    p = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "train_one.py"),
         "--model", model, "--name", name],
        cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
    )
    p.wait()
    out.close()
    log(f"{name} exit {p.returncode}")
    return p.returncode


def do_job(job):
    if finished(job["name"]):
        log(f"{job['name']} already done, skip")
        return True
    rc = run_once(job)
    if rc != 0:
        log(f"{job['name']} failed ({rc}); wait 120s then retry")
        time.sleep(120)
        rc = run_once(job)
    ok = finished(job["name"])
    log(f"{job['name']} result={'OK' if ok else 'FAILED'}")
    return ok


# wave 1: YOLOv8n + YOLO11n parallel
w1 = QUEUE[:2]
log(f"wave 1: {[j['name'] for j in w1]} parallel")
procs = {}
for job in w1:
    if finished(job["name"]):
        log(f"{job['name']} already done, skip")
        continue
    out = open(STUDY / f"{job['name']}.log", "w")
    p = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "train_one.py"),
         "--model", job["model"], "--name", job["name"]],
        cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
    )
    procs[job["name"]] = (p, out)

for name, (p, out) in procs.items():
    p.wait()
    out.close()
    log(f"wave1 {name} exit {p.returncode}")

# retry wave1 failures sequentially (wave1 parallel already done, no GPU competition)
for job in w1:
    if finished(job["name"]):
        continue
    log(f"wave1 {job['name']} retry sequentially")
    time.sleep(60)
    do_job(job)

# wave 2: YOLO11s alone
time.sleep(60)
log("wave 2: yolo11s alone")
do_job(QUEUE[2])

done = [j["name"] for j in QUEUE if finished(j["name"])]
log(f"summary: finished={done}")
if len(done) == len(QUEUE):
    (STUDY / "QUEUE_DONE").touch()
    log("QUEUE_DONE")
else:
    log(f"INCOMPLETE ({len(done)}/{len(QUEUE)})")
