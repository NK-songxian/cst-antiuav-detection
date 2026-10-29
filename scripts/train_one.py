"""Single baseline training. Invoked by run_queue.py as subprocess."""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("/root/autodl-tmp/Ryan/cst_antiuav")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--name", required=True)
args = ap.parse_args()

result = YOLO(args.model).train(
    data=str(ROOT / "data/yolo/cst.yaml"),
    epochs=100, patience=20, batch=256, imgsz=640, workers=16,
    device="0", seed=0, augment=True, project=str(ROOT / "runs"),
    name=args.name, exist_ok=True, close_mosaic=10,
    pretrained=True, deterministic=True, verbose=True,
)
sys.exit(0)
