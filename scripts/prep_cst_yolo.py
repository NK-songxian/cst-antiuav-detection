"""Convert CST Anti-UAV dataset to YOLO detection format.

Verified layout (10/16, full dataset zip):
  CST-AntiUAV/{train,val,test}/{seq}/
      000001.jpg ...          640x512 RGB (grayscale content, 3 channels)
      gt.txt                  one line per frame: x,y,w,h (top-left, pixel)
      IR_label.json           keys: exist, gt, OC, OV, SV, TC, DBC, CDB, FM, SIZE
                              all lists of length n_frames (frame-level attrs!)
      exist.txt               one int per frame

Output:
  data/yolo/{split}/images/*.jpg   (hardlinked)
  data/yolo/{split}/labels/*.txt   YOLO: "0 cx cy w h" normalized; empty = background
  data/yolo/attribute_index.json   per-frame 8-dim attrs + exist (for AAT & stratified eval)
  data/yolo/cst.yaml               ultralytics dataset config
"""
import argparse
import json
import os
from pathlib import Path

ATTR_KEYS = ["OC", "OV", "SV", "TC", "DBC", "CDB", "FM", "SIZE"]
IMG_W, IMG_H = 640.0, 512.0


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cst_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame_stride", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.cst_root)
    out = Path(args.out)
    index = {}
    stats = {"seqs": 0, "frames_kept": 0, "pos": 0, "bg": 0, "skipped_exist0_dummy_box": 0}
    oob = {"x_plus_w_gt_W": 0, "y_plus_h_gt_H": 0, "x_gt_W": 0, "total": 0}
    attr_counts = {k: 0 for k in ATTR_KEYS}

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[warn] missing split {split}")
            continue
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        index[split] = {}
        seqs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for seq_dir in seqs:
            seq = seq_dir.name
            lab = json.loads((seq_dir / "IR_label.json").read_text())
            exist = lab["exist"]
            gt = lab["gt"]
            attrs = {k: lab[k] for k in ATTR_KEYS if k in lab}
            n = len(exist)
            frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() == ".jpg")
            if len(frames) != n:
                print(f"[warn] {split}/{seq}: {len(frames)} imgs vs {n} anno entries; use min")
            kept = 0
            for i in range(0, min(n, len(frames)), args.frame_stride):
                e = exist[i]
                b = gt[i]
                x, y, w, h = (b + [0.0] * 4)[:4] if isinstance(b, list) else (0, 0, 0, 0)
                oob["total"] += 1
                if x > IMG_W:
                    oob["x_gt_W"] += 1
                if x + w > IMG_W + 2:
                    oob["x_plus_w_gt_W"] += 1
                if y + h > IMG_H + 2:
                    oob["y_plus_h_gt_H"] += 1

                stem = f"{seq}_f{i:06d}"
                dst_img = out / split / "images" / f"{stem}.jpg"
                if not dst_img.exists():
                    try:
                        os.link(frames[i], dst_img)
                    except OSError:
                        import shutil
                        shutil.copy2(frames[i], dst_img)

                lines = []
                if e == 1 and w > 1 and h > 1:
                    cx = clip((x + w / 2) / IMG_W, 0, 1)
                    cy = clip((y + h / 2) / IMG_H, 0, 1)
                    nw = clip(w / IMG_W, 0, 1)
                    nh = clip(h / IMG_H, 0, 1)
                    lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    stats["pos"] += 1
                else:
                    stats["bg"] += 1
                    if e == 0:
                        stats["skipped_exist0_dummy_box"] += 1
                (out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))

                attr = {k: int(attrs[k][i]) for k in attrs}
                for k, v in attr.items():
                    attr_counts[k] += int(v == 1)
                index[split][stem] = {"seq": seq, "frame": i, "exist": int(e), "attr": attr}
                kept += 1
            stats["seqs"] += 1
            stats["frames_kept"] += kept
            print(f"[{split}] {seq}: {n} frames -> {kept} kept")
        print(f"[{split}] done: {len(seqs)} seqs")

    (out / "attribute_index.json").write_text(json.dumps(index))
    yaml = (f"path: {out.resolve()}\n"
            f"train: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: ['uav']\n")
    (out / "cst.yaml").write_text(yaml)
    print("STATS:", json.dumps(stats))
    print("OOB(convention check):", json.dumps(oob))
    print("ATTR_POS_COUNTS:", json.dumps(attr_counts))


if __name__ == "__main__":
    main()
