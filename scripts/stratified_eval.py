#!/usr/bin/env python3
"""8属性分层评测：baseline vs AAT（COCOeval 按属性子集过滤）
每模型只预测一次 val，缓存检测；再按 attribute_index 属性旗标过滤 GT/DT 分层评测。
"""
import json, gc
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import torch

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
IDX = json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['val']
VAL_IMG = ROOT/'data/yolo/val/images'
ATTRS = ['OC','OV','SV','TC','DBC','CDB','SIZE','FM']

# 1) GT
images, anns = [], []
ann_id = 0
for stem, rec in IDX.items():
    lbl = VAL_IMG.parent/'labels'/f'{stem}.txt'
    if not lbl.exists():
        continue
    lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
    if not lines:
        continue
    images.append({'id': stem, 'file_name': f'{stem}.jpg'})
    for l in lines:
        cx, cy, w, h = (float(v) * s for v, s in zip(l[1:5], (640, 512, 640, 512)))
        anns.append({'id': ann_id, 'image_id': stem, 'category_id': 1,
                     'bbox': [cx-w/2, cy-h/2, w, h], 'area': w*h, 'iscrowd': 0})
        ann_id += 1
print(f'val GT: {len(images)} imgs / {len(anns)} anns', flush=True)

def coco_eval(imgs, anns_f, dets):
    gt = {'images': imgs, 'annotations': anns_f, 'categories': [{'id': 1, 'name': 'uav'}]}
    coco_gt = COCO(); coco_gt.dataset = gt; coco_gt.createIndex()
    dts = [d for d in dets if d['image_id'] in {i['id'] for i in imgs}]
    if not dts:
        return None
    coco_dt = coco_gt.loadRes(dts)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0], ev.stats[1]

MODELS = {'baseline': ROOT/'runs/baseline_yolo11n/weights/best.pt',
          'AAT_v1':  ROOT/'runs/AAT_v1_yolo11n/weights/best.pt'}
paths = sorted(VAL_IMG.glob('*.jpg'))
out = {'n_frames': {'ALL': len(images)}, 'results': {}}

for name, w in MODELS.items():
    print(f'\n######## {name} predicting ########', flush=True)
    model = YOLO(str(w))
    dets = []
    for i in range(0, len(paths), 512):
        results = model.predict(paths[i:i+512], conf=0.001, iou=0.7, imgsz=640,
                                batch=16, stream=True, verbose=False)
        for r in results:
            stem = Path(r.path).stem
            for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                x1, y1, x2, y2 = b
                dets.append({'image_id': stem, 'category_id': 1,
                             'bbox': [x1, y1, x2-x1, y2-y1], 'score': s})
        gc.collect(); torch.cuda.empty_cache()
    print(f'{name}: {len(dets)} dets cached', flush=True)

    res = {'ALL': coco_eval(images, anns, dets)}
    for attr in ATTRS:
        n = sum(1 for r in IDX.values() if r['attr'].get(attr) == 1)
        out['n_frames'][attr] = n
        if n >= 20:
            sub = {s for s, r in IDX.items() if r['attr'].get(attr) == 1}
            imgs = [i for i in images if i['id'] in sub]
            anns_f = [a for a in anns if a['image_id'] in sub]
            res[attr] = coco_eval(imgs, anns_f, dets)
    out['results'][name] = res
    del dets; gc.collect(); torch.cuda.empty_cache()

print('\n===== 分层 mAP50 / mAP50-95 对比 =====')
print(f"{'attr':<6}{'frames':>7} | {'base50':>7}{'aat50':>7}{'Δ':>8} | {'base95':>7}{'aat95':>7}{'Δ':>8}")
for k in ['ALL'] + ATTRS:
    b = out['results']['baseline'].get(k)
    a = out['results']['AAT_v1'].get(k)
    if not b or not a:
        continue
    print(f"{k:<6}{out['n_frames'][k]:>7} | {b[0]:>7.4f}{a[0]:>7.4f}{(a[0]-b[0])*100:>+7.1f}pt | {b[1]:>7.4f}{a[1]:>7.4f}{(a[1]-b[1])*100:>+7.1f}pt")

(ROOT/'runs/stratified_eval.json').write_text(json.dumps(out, indent=1))
print('\nsaved -> runs/stratified_eval.json')
