"""FP study: 4-group controlled training for matched-pair auxiliary loss.

Groups (all share identical training config, geometric augmentation fully off):
  G1  baseline retrain (no aux loss)      -- fair reference under no-geom-aug config
  G2  naive FP suppression                -- aux on 31 reliable FP cells only
  G3  matched protection                  -- G2 + protect 31 matched true UAVs (pairs.json)
  G4  shuffle control                     -- G2 + protect 31 random true UAVs (seed=0)

aux = bs * lam * mean( softplus(z_n @ FP cells) + softplus(margin - z_p @ protected cells) )
FP/GT cells: box center rasterized at all 3 scales (stride 8/16/32).

Geometry note: images are 640x512; with all geometric augs off the train transform is a
deterministic letterbox to 640x640 (r=1.0, content shifted down by 64px, border=114).
So cx' = cx, cy' = cy + 64. Verified against batch bboxes in --smoke.
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "runs/fp_pool/pool_tiered.json"
PAIRS = ROOT / "runs/fp_pool/pairs.json"
TRAIN_LABELS = ROOT / "data/yolo/train/labels"
IMG = 640
PAD_Y = 64.0  # letterbox shift: 640x512 -> 640x640
STRIDES = (8.0, 16.0, 32.0)
LAM = 0.1
MARGIN = 1.0

AUX_LOG = []  # (step, n_cells, aux_raw) for smoke diagnostics


def fp_spec():
    tiered = json.load(open(POOL))
    fps = tiered["reliable"]
    assert len(fps) == 31, f"expect 31 reliable FPs, got {len(fps)}"
    return {e["stem"]: [tuple(e["box"])] for e in fps}


def prot_spec_matched():
    pairs = json.load(open(PAIRS))["pairs"]
    assert len(pairs) == 31, f"expect 31 pairs, got {len(pairs)}"
    m = {}
    for p in pairs:
        m.setdefault(p["pos"]["stem"], []).append(tuple(p["pos"]["box"]))
    return m


def prot_spec_random(fp_map, n=31):
    """Random GT boxes (xyxy px in original 640x512 coords) from train labels, seed=0."""
    rnd = random.Random(0)
    fp_stems = set(fp_map)
    cands = []
    for lf in sorted(TRAIN_LABELS.glob("*.txt")):
        stem = lf.stem
        if stem in fp_stems:
            continue
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            _, cx, cy, w, h = map(float, p)
            # normalized xywh over (640, 512) -> xyxy px
            cands.append((stem, ((cx - w / 2) * 640, (cy - h / 2) * 512,
                                 (cx + w / 2) * 640, (cy + h / 2) * 512)))
    picks = rnd.sample(cands, n)
    m = {}
    for stem, box in picks:
        m.setdefault(stem, []).append(box)
    return m


class MatchAuxLoss(nn.Module):
    def __init__(self, base, fp_map, prot_map, lam=LAM, margin=MARGIN):
        super().__init__()
        self.base = base
        self.fp_map = fp_map
        self.prot_map = prot_map
        self.lam = lam
        self.margin = margin

    def forward(self, preds, batch):
        loss, items = self.base(preds, batch)
        p = preds[1] if isinstance(preds, tuple) else preds
        if "one2many" in p:  # end-to-end heads wrap the dict
            p = p["one2many"]
        scores = p["scores"]  # [bs, nc, A] raw logits, levels 8->16->32, row-major within level
        bs = batch["img"].shape[0]
        total, count = self._collect(scores, batch)
        if count > 0:
            loss = loss + bs * self.lam * total / count
            AUX_LOG.append((len(AUX_LOG), count, float(total.detach() / count)))
        return loss, items

    def _collect(self, scores, batch):
        hs = [IMG // int(s) for s in STRIDES]  # 80, 40, 20
        offs, acc = [], 0
        for h in hs:
            offs.append(acc)
            acc += h * h
        total = scores.new_zeros(())
        count = 0
        for b, f in enumerate(batch["im_file"]):
            stem = Path(f).stem
            for sign, box_map in ((1.0, self.fp_map), (-1.0, self.prot_map)):
                for box in box_map.get(stem, ()):
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0 + PAD_Y
                    for i, s in enumerate(STRIDES):
                        h = hs[i]
                        gi = min(max(int(cx // s), 0), h - 1)
                        gj = min(max(int(cy // s), 0), h - 1)
                        z = scores[b, 0, offs[i] + gj * h + gi]
                        term = nn.functional.softplus(z) if sign > 0 else nn.functional.softplus(self.margin - z)
                        total = total + term
                        count += 1
        return total, count


def verify_transform(prot_map, batch):
    """Check letterbox formula against authoritative batch bboxes (normalized xywh @ 640x640)."""
    ok, checked = 0, 0
    for b, f in enumerate(batch["im_file"]):
        stem = Path(f).stem
        if stem not in prot_map or len(batch["bboxes"][batch["batch_idx"] == b]) == 0:
            continue
        bxs = batch["bboxes"][batch["batch_idx"] == b]
        for box in prot_map[stem]:
            gi = min(max(int(((box[0] + box[2]) / 2) // 8), 0), 79)
            gj = min(max(int(((box[1] + box[3]) / 2 + PAD_Y) // 8), 0), 79)
            best = min(
                range(len(bxs)),
                key=lambda i: abs(bxs[i][0] * 640 - (box[0] + box[2]) / 2) + abs(bxs[i][1] * 640 - (box[1] + box[3]) / 2 - PAD_Y),
            )
            gx = int(bxs[best][0] * 640 // 8)
            gy = int(bxs[best][1] * 640 // 8)
            checked += 1
            ok += int((gi, gj) == (gx, gy))
    return ok, checked


class AuxDetectionModel(DetectionModel):
    """DetectionModel whose criterion wraps v8DetectionLoss with MatchAuxLoss.

    Module-level class so checkpoints stay picklable; fp_map/prot_map are plain
    instance attributes (dicts of str->list[tuple]), also picklable.
    """

    fp_map = {}
    prot_map = {}

    def init_criterion(self):
        from ultralytics.utils.loss import v8DetectionLoss

        base = v8DetectionLoss(self)
        if self.fp_map or self.prot_map:
            return MatchAuxLoss(base, self.fp_map, self.prot_map)
        return base


class FPStudyTrainer(DetectionTrainer):
    fp_map = {}
    prot_map = {}

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if self.fp_map or self.prot_map:
            model.__class__ = AuxDetectionModel
            model.fp_map = self.fp_map
            model.prot_map = self.prot_map
        return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["G1", "G2", "G3", "G4"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    fp_map, prot_map = {}, {}
    if args.group in ("G2", "G3", "G4"):
        fp_map = fp_spec()
    if args.group == "G3":
        prot_map = prot_spec_matched()
    if args.group == "G4":
        prot_map = prot_spec_random(fp_map)
    n_fp = sum(len(v) for v in fp_map.values())
    n_prot = sum(len(v) for v in prot_map.values())
    print(f"[fp_study] group={args.group} fp_boxes={n_fp} prot_boxes={n_prot} "
          f"fp_stems={len(fp_map)} prot_stems={len(prot_map)}", flush=True)

    epochs = 1 if args.smoke else args.epochs
    trainer = FPStudyTrainer(
        overrides=dict(
            model=str(ROOT / "yolo11n.pt"),
            data=str(ROOT / "data/yolo/cst.yaml"),
            epochs=epochs,
            patience=20 if not args.smoke else 100,
            batch=args.batch,
            imgsz=IMG,
            workers=args.workers,
            # geometric augmentation fully off (deterministic letterbox only); HSV kept
            mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
            scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
            fliplr=0.0, flipud=0.0, erasing=0.0,
            seed=0,
            project=str(ROOT / "runs/fp_study"),
            name=f"{args.group}_smoke" if args.smoke else args.group,
            exist_ok=True,
        ),
    )
    trainer.fp_map = fp_map
    trainer.prot_map = prot_map
    if args.smoke and prot_map:
        # intercept one batch to verify the letterbox formula
        orig_get = trainer.get_dataloader

        def get_dataloader(*a, **kw):
            dl = orig_get(*a, **kw)
            if getattr(dl, "verified", False):
                return dl
            dl.verified = True
            batch = next(iter(dl))
            ok, checked = verify_transform(prot_map, batch)
            print(f"[fp_study][smoke] letterbox cell check: {ok}/{checked} match", flush=True)
            assert checked == 0 or ok == checked, "letterbox transform mismatch!"
            return dl

        trainer.get_dataloader = get_dataloader
    trainer.train()
    if args.smoke:
        n = min(10, len(AUX_LOG))
        vals = [v for _, _, v in AUX_LOG[-n:]] or []
        print(f"[fp_study][smoke] aux term mean(last {n} batches): "
              f"{sum(vals) / max(len(vals), 1):.4f} (n_batches={len(AUX_LOG)})", flush=True)


if __name__ == "__main__":
    main()
