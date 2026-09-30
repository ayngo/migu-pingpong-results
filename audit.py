"""Read archives without modifying them; produce reproducible training inventory."""
import argparse, collections, csv, hashlib, json, math, zipfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--participant',required=True);ap.add_argument('--out',default='results');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    counts=collections.Counter(); resolutions=collections.Counter(); labels=collections.Counter();issues=[]; rows=[]; sizes=[]
    with zipfile.ZipFile(a.train) as z:
        names=set(z.namelist())
        for name in sorted(names):
            if '/annotations/' not in name or not name.endswith('.json'):continue
            d=json.loads(z.read(name));w,h=d['frame_width'],d['frame_height'];resolutions[f'{w}x{h}']+=1
            counts['videos']+=1;counts['declared_frames']+=d['num_frames'];counts['frame_records']+=len(d['frames'])
            if 'pp_train_data/'+d['video_path'] not in names:issues.append([name,'missing_video'])
            ids=[f['frame_id'] for f in d['frames']]
            if sorted(ids)!=list(range(d['num_frames'])):issues.append([name,'non_contiguous_frame_ids'])
            group=name.split('/')[-2]
            # Directory groups are a provisional proxy; not verified match identities.
            split='val' if group in {'03','11','13','14','20'} else 'train'
            for f in d['frames']:
                balls=[]
                for ann in f['annotations']:
                    labels[ann['label']]+=len(ann['bbox_xyxy'])
                    if ann['label']=='ball':balls.extend(ann['bbox_xyxy'])
                counts['frames_with_ball' if balls else 'frames_without_ball']+=1
                if len(balls)>1:counts['multiple_ball_frames']+=1
                for box in balls:
                    x1,y1,x2,y2=box
                    valid=all(math.isfinite(v) for v in box) and 0<=x1<x2<=w and 0<=y1<y2<=h
                    if not valid:issues.append([d['video_id'],f['frame_id'],'invalid_ball_bbox',box]);continue
                    sizes.append(((x2-x1)*512/w,(y2-y1)*288/h))
                    rows.append(dict(video_id=d['video_id'],video_path=d['video_path'],group=group,split=split,frame_id=f['frame_id'],width=w,height=h,x=(x1+x2)/2,y=(y1+y2)/2))
        counts['annotation_files']=counts['videos'];counts['video_files']=sum(n.endswith('.mp4') for n in names)
    with (out/'ball_centers.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    model_dir=out/'model';model_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(a.participant) as z:
        b=z.read('participant/pingpang/participant/weights/ball.onnx');(model_dir/'ball.onnx').write_bytes(b)
        code=[n for n in z.namelist() if n.endswith(('.py','.pt','.pth','.ckpt')) and '/pingpang/' in n]
    stats={'counts':dict(counts),'resolutions':dict(resolutions),'labels':dict(labels),'issues':issues,'ball_onnx_sha256':hashlib.sha256(b).hexdigest(),'ball_onnx_bytes':len(b),'pingpang_code_and_checkpoint_files':code,'split_note':'Provisional split by source directory, NOT verified match-level split; absent ball annotation is UNKNOWN, not a confirmed negative.','split_groups':{s:sorted({r['group'] for r in rows if r['split']==s}) for s in ('train','val')},'scaled_ball_median_wh': [sorted(v[i] for v in sizes)[len(sizes)//2] for i in (0,1)]}
    (out/'audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
