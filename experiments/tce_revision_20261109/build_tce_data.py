"""Build TCE (Temporal Channel Encoding) revision datasets on top of the frozen
gap10 (far=10, near=5) triplet protocol.

Images: pixel-identical PNGs to the aat_temporal gap10 training data (BGR file
order = past-to-current, grayscale merge), rebuilt from data/yolo JPEGs because
the original directory was archived away. train/val triplets reuse the frozen
aat history_*.json (verified equal to build_history); test triplets are built
with the same verified function.

Labels (evaluation splits keep the original current-frame GT protocol):
  train T2 (nc=1): current boxes + ghost boxes, all class 0.
  train T3 (nc=3): current->2, near ghost->1, far ghost->0.
  val/test T2: original labels untouched (directory symlink).
  val/test T3: original boxes remapped to class 2, no ghosts.
Ghost rule (frozen): a past-channel box is a ghost iff max IoU vs every current
box < 0.5; far ghosts additionally require IoU < 0.5 vs kept near ghosts (near
channel wins). The 86.3%/1.97 paper diagnostic corresponds to the unfiltered
all-past-boxes variant (see probe_ghost.py).
"""
import concurrent.futures
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/tce_revision_20261109'
AAT = ROOT / 'experiments/aat_temporal_100e_20261017'
NEAR, FAR = 5, 10
IOU_T = 0.5


def build_history(index, near, far):
    groups = defaultdict(dict)
    for stem, rec in index.items():
        groups[rec['seq']][rec['frame']] = stem
    result = {}
    for stem, rec in index.items():
        lookup = groups[rec['seq']]
        frame = rec['frame']
        p_near = lookup.get(frame - near, stem)
        p_far = lookup.get(frame - far, p_near)
        result[stem] = [p_far, p_near, stem]
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_boxes(stem, split):
    """Raw (cx, cy, w, h) normalized floats; empty when no label file."""
    p = ROOT / 'data/yolo' / split / 'labels' / (stem + '.txt')
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append(tuple(map(float, parts[1:5])))
    return out


def to_xyxy(box):
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area - inter + 1e-12) if area > 0 else 0.0


def ghost_split(triplet, split):
    """Return (current_raw, near_ghost_raw, far_ghost_raw) under the frozen rule."""
    far_s, near_s, cur_s = triplet
    cur = raw_boxes(cur_s, split)
    cur_x = [to_xyxy(b) for b in cur]
    near_g = []
    for b in raw_boxes(near_s, split):
        bx = to_xyxy(b)
        if all(iou(bx, c) < IOU_T for c in cur_x):
            near_g.append(b)
    near_gx = [to_xyxy(b) for b in near_g]
    far_g = []
    for b in raw_boxes(far_s, split):
        bx = to_xyxy(b)
        if all(iou(bx, c) < IOU_T for c in cur_x) and all(iou(bx, g) < IOU_T for g in near_gx):
            far_g.append(b)
    return cur, near_g, far_g


def fmt(cls, box):
    return f'{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}'


def make_image(triplet, split, dest_dir):
    stem = triplet[2]
    source = ROOT / 'data/yolo' / split / 'images'
    ims = {s: cv2.imread(str(source / (s + '.jpg')), cv2.IMREAD_GRAYSCALE) for s in set(triplet)}
    assert all(im is not None for im in ims.values()), stem
    merged = cv2.merge([ims[s] for s in triplet])
    dest = dest_dir / (stem + '.png')
    assert cv2.imwrite(str(dest), merged, [cv2.IMWRITE_PNG_COMPRESSION, 1]), dest
    return stem, sha(dest)


def main():
    cv2.setNumThreads(1)
    index = json.loads((ROOT / 'data/yolo/attribute_index.json').read_text())
    out = STUDY / 'data'
    manifest = dict(status='running', created_at=datetime.now(timezone.utc).isoformat(),
                    near=NEAR, far=FAR, ghost_iou_threshold=IOU_T, splits={})

    for split in ('train', 'val', 'test'):
        if split in ('train', 'val'):
            history = json.loads((AAT / f'history_{split}.json').read_text())
            assert history == build_history(index[split], NEAR, FAR)
        else:
            history = build_history(index[split], NEAR, FAR)
        (STUDY / f'history_{split}.json').write_text(json.dumps(history))

        gap_img = out / 'gap10' / split / 'images'
        gap_img.mkdir(parents=True, exist_ok=True)
        t2_lab = out / 't2' / split / 'labels'
        t3_lab = out / 't3' / split / 'labels'
        t2_lab.parent.mkdir(parents=True, exist_ok=True)
        t2_lab.mkdir(parents=True, exist_ok=True)
        t3_lab.parent.mkdir(parents=True, exist_ok=True)
        t3_lab.mkdir(parents=True, exist_ok=True)
        stems = sorted(index[split])

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(make_image, history[s], split, gap_img) for s in stems]
            hashes = dict(f.result() for f in futures)
        assert len(hashes) == len(stems)

        g_frames = g_total = all_past = 0
        cls_counts = {0: 0, 1: 0, 2: 0}
        if split == 'train':
            for stem in stems:
                cur, near_g, far_g = ghost_split(history[stem], split)
                g = len(near_g) + len(far_g)
                if g:
                    g_frames += 1
                    g_total += g
                all_past += len(raw_boxes(history[stem][0], split)) + len(raw_boxes(history[stem][1], split))
                lines2 = [fmt(0, b) for b in cur] + [fmt(0, b) for b in near_g] + [fmt(0, b) for b in far_g]
                (t2_lab / (stem + '.txt')).write_text('\n'.join(lines2) + ('\n' if lines2 else ''))
                lines3 = [fmt(2, b) for b in cur] + [fmt(1, b) for b in near_g] + [fmt(0, b) for b in far_g]
                (t3_lab / (stem + '.txt')).write_text('\n'.join(lines3) + ('\n' if lines3 else ''))
                for line in lines3:
                    cls_counts[int(line.split()[0])] += 1
            t2_lab.mkdir(parents=True, exist_ok=True)
        else:
            for stem in stems:
                cur = raw_boxes(stem, split)
                lines3 = [fmt(2, b) for b in cur]
                (t3_lab / (stem + '.txt')).write_text('\n'.join(lines3) + ('\n' if lines3 else ''))
            # T2 val/test labels: original labels, class 0 unchanged.
            if t2_lab.is_symlink():
                pass
            elif t2_lab.exists():
                shutil.rmtree(t2_lab)
            t2_lab.parent.mkdir(parents=True, exist_ok=True)
            t2_lab.symlink_to(ROOT / 'data/yolo' / split / 'labels', target_is_directory=True)

        for arm in ('t2', 't3'):
            img_link = out / arm / split / 'images'
            if not img_link.exists():
                img_link.parent.mkdir(parents=True, exist_ok=True)
                img_link.symlink_to(gap_img, target_is_directory=True)

        print(f'[{split}] images={len(stems)} '
              f'ghost_frames={g_frames / len(stems):.3f} ghost_total={g_total} '
              f'all_past_boxes={all_past} t3_class_counts={cls_counts}', flush=True)

        txt = '\n'.join(str(gap_img / (s + '.png')) for s in stems) + '\n'
        (STUDY / f'{split}.txt').write_text(txt)
        hash_list = '\n'.join(hashes[s] for s in stems) + '\n'
        (STUDY / f'{split}_image_hashes.txt').write_text(hash_list)
        manifest['splits'][split] = dict(
            frames=len(stems),
            ghost_frame_ratio=(g_frames / len(stems)) if split == 'train' else None,
            ghost_total=g_total if split == 'train' else None,
            all_past_boxes=all_past if split == 'train' else None,
            t3_class_counts=cls_counts if split == 'train' else None,
            image_hash_list_sha256=hashlib.sha256(hash_list.encode()).hexdigest(),
            history_sha256=sha(STUDY / f'history_{split}.json'),
            list_sha256=sha(STUDY / f'{split}.txt'))

    # Dataset yamls: per-arm frame lists so the /images -> /labels derivation
    # resolves inside each arm's tree (labels live in data/<arm>/<split>/labels).
    lists = {}
    for split in ('train', 'val', 'test'):
        stems = sorted(index[split])
        for arm in ('t2', 't3'):
            txt = '\n'.join(str(out / arm / split / 'images' / (s + '.png')) for s in stems) + '\n'
            (STUDY / f'{arm}_{split}.txt').write_text(txt)
            lists[arm + '_' + split] = hashlib.sha256(txt.encode()).hexdigest()
    for arm, nc, names in (('t2', 1, ['- uav']),
                           ('t3', 3, ['- uav_far', '- uav_near', '- uav_current'])):
        (STUDY / f'cst_tce_{arm}.yaml').write_text(
            f"path: {out / arm}\n"
            f"train: {STUDY / f'{arm}_train.txt'}\n"
            f"val: {STUDY / f'{arm}_val.txt'}\n"
            f"test: {STUDY / f'{arm}_test.txt'}\n"
            f"nc: {nc}\n" + 'names:\n' + '\n'.join(names) + '\n')
    manifest['list_sha256s'] = lists

    # Probe: pixel equality of saved channels vs source frames.
    rng = np.random.default_rng(0)
    for split in ('train', 'val', 'test'):
        stems = sorted(index[split])
        history = json.loads((STUDY / f'history_{split}.json').read_text())
        for stem in rng.choice(stems, size=30, replace=False):
            s = str(stem)
            saved = cv2.imread(str(out / 'gap10' / split / 'images' / (s + '.png')))
            assert saved is not None, s
            for ch, prev in enumerate(history[s]):
                raw = cv2.imread(str(ROOT / 'data/yolo' / split / 'images' / (prev + '.jpg')),
                                 cv2.IMREAD_GRAYSCALE)
                assert np.array_equal(saved[:, :, ch], raw), (s, ch)
        print(f'[probe:{split}] 30 frames channel-equality PASS', flush=True)

    manifest['status'] = 'PASS'
    manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
    (STUDY / 'data_ready.json').write_text(json.dumps(manifest, indent=2))
    print('TCE_DATA_DONE', flush=True)


if __name__ == '__main__':
    main()
