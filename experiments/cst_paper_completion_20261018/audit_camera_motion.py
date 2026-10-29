"""Sample all val/test sequences. Estimate apparent background transform, not camera ground truth."""
from pathlib import Path
from collections import defaultdict
import json,csv,hashlib
import cv2,numpy as np
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
OUT=Path(__file__).parent
def main():
    cv2.setNumThreads(1);index=json.loads((ROOT/'data/yolo/attribute_index.json').read_text());records=[]
    cfg=ROOT/'experiments/aat_temporal_100e_20261017'
    orb=cv2.ORB_create(nfeatures=1200);bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    for split in ['val','test']:
        idx=index[split];byseq=defaultdict(list);lookup=defaultdict(dict)
        for stem,r in idx.items():byseq[r['seq']].append(stem);lookup[r['seq']][r['frame']]=stem
        paths={Path(p).stem:p for p in (cfg/f'temporal_{split}.txt').read_text().splitlines()} if split=='val' else {Path(p).stem:p for p in (cfg/'completed_review_20261017/temporal_test.txt').read_text().splitlines()}
        for seq,stems in sorted(byseq.items()):
            choices=sorted([s for s in stems if idx[s]['frame']-5 in lookup[seq]],key=lambda s:idx[s]['frame'])
            for k in sorted(set(np.linspace(0,len(choices)-1,min(8,len(choices)),dtype=int).tolist())):
                stem=choices[k];prev=lookup[seq][idx[stem]['frame']-5]
                image=cv2.imread(paths[stem]);a=image[:,:,1].copy();b=image[:,:,2].copy();h,w=a.shape
                def mask(s):
                    m=np.full((h,w),255,np.uint8);txt=(ROOT/'data/yolo'/split/'labels'/(s+'.txt')).read_text().strip()
                    for line in txt.splitlines():
                        _,cx,cy,bw,bh=map(float,line.split());x1=max(0,int((cx-bw/2)*w)-8);x2=min(w,int((cx+bw/2)*w)+9)
                        y1=max(0,int((cy-bh/2)*h)-8);y2=min(h,int((cy+bh/2)*h)+9);m[y1:y2,x1:x2]=0
                    return m
                k1,d1=orb.detectAndCompute(a,mask(prev));k2,d2=orb.detectAndCompute(b,mask(stem))
                row=dict(split=split,seq=seq,stem=stem,previous=prev,source_gap=5,orb_matches=0,orb_inliers=0,
                    confidence_qualified=False,center_shift_px=None,rotation_degrees=None,scale=None,static_like=None)
                if d1 is not None and d2 is not None and len(d1)>=2 and len(d2)>=2:
                    pairs=bf.knnMatch(d1,d2,k=2);good=[m for pp in pairs if len(pp)==2 for m,n in [pp] if m.distance<.75*n.distance]
                    row['orb_matches']=len(good)
                    if len(good)>=12:
                        src=np.float32([k1[m.queryIdx].pt for m in good]);dst=np.float32([k2[m.trainIdx].pt for m in good])
                        affine,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=2000,confidence=.99)
                        count=int(inliers.sum()) if inliers is not None else 0;row['orb_inliers']=count
                        if affine is not None and count>=12 and count/len(good)>=.5:
                            center=np.array([w/2,h/2]);shift=float(np.linalg.norm(affine[:,:2]@center+affine[:,2]-center))
                            rotation=float(np.degrees(np.arctan2(affine[1,0],affine[0,0])));scale=float(np.hypot(affine[0,0],affine[1,0]))
                            row.update(confidence_qualified=True,center_shift_px=shift,rotation_degrees=rotation,scale=scale,
                                static_like=shift<=.5 and abs(rotation)<=.1 and abs(scale-1)<=.001)
                window=cv2.createHanningWindow((w,h),cv2.CV_32F)
                (dx,dy),response=cv2.phaseCorrelate(a.astype(np.float32),b.astype(np.float32),window)
                row.update(phase_shift_px=float(np.hypot(dx,dy)),phase_response=float(response))
                records.append(row)
        print('SPLIT_COMPLETE',split,len([r for r in records if r['split']==split]),flush=True)
    report=dict(scope='8 deterministic neighboring t-5/current pairs per val/test sequence where available; post-results descriptive analysis.',
        method='ORB on GT-excluded background; partial-affine RANSAC; >=12 inliers and >=50% inlier ratio. Phase correlation is supplemental and does not establish rigid camera stillness.',
        static_like_threshold=dict(center_shift_px=.5,rotation_deg=.1,scale_difference=.001),
        limitation='Estimated apparent background motion, not calibrated camera motion. Low-texture scenes can be unmeasurable; no extrapolation to unsampled frames.',splits={})
    for split in ['val','test']:
        rr=[r for r in records if r['split']==split];valid=[r for r in rr if r['confidence_qualified']]
        report['splits'][split]=dict(sequences=len({r['seq'] for r in rr}),pairs=len(rr),qualified_pairs=len(valid),
          qualified_coverage=len(valid)/len(rr),qualified_static_like_fraction=sum(r['static_like'] for r in valid)/len(valid) if valid else None,
          qualified_center_shift_median_px=float(np.median([r['center_shift_px'] for r in valid])) if valid else None,
          qualified_center_shift_p90_px=float(np.percentile([r['center_shift_px'] for r in valid],90)) if valid else None,
          phase_response_ge02_pairs=sum(r['phase_response']>=.2 for r in rr))
    with (OUT/'camera_motion_pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report['csv_sha256']=hashlib.sha256((OUT/'camera_motion_pairs.csv').read_bytes()).hexdigest()
    (OUT/'camera_motion_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
