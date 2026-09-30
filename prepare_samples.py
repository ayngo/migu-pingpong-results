"""Create a small positive-only 3-frame smoke dataset; never label missing balls negative."""
import argparse,csv,json,zipfile,tempfile
from pathlib import Path
import cv2
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument('--train-zip',required=True);p.add_argument('--centers',required=True);p.add_argument('--out',required=True);p.add_argument('--limit-per-split',type=int,default=32);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with open(a.centers,encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    wanted={};counts={'train':0,'val':0}
    for r in rows:
        fid=int(r['frame_id']);s=r['split']
        if fid<1 or counts[s]>=a.limit_per_split:continue
        wanted.setdefault(r['video_path'],[]).append(r);counts[s]+=1
    manifest=[]
    with zipfile.ZipFile(a.train_zip) as z,tempfile.TemporaryDirectory() as tmp:
        for vp,items in wanted.items():
            local=Path(tmp)/'video.mp4';local.write_bytes(z.read('pp_train_data/'+vp))
            targets={int(r['frame_id']):r for r in items};last=max(targets)+1
            cap=cv2.VideoCapture(str(local));buf=[];fid=0
            try:
                while fid<=last:
                    ok,bgr=cap.read()
                    if not ok:break
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);buf.append(cv2.resize(rgb,(512,288)));buf=buf[-3:]
                    mid=fid-1
                    if len(buf)==3 and mid in targets:
                        r=targets[mid];x=float(r['x'])*512/int(r['width']);y=float(r['y'])*288/int(r['height'])
                        yy,xx=np.mgrid[:288,:512];heat=np.exp(-((xx-x)**2+(yy-y)**2)/(2*2.0**2)).astype('float32')
                        name=f"{r['split']}_{r['video_id']}_{mid:05d}.npz"
                        np.savez_compressed(out/name,rgb=np.stack(buf),target=heat,center_xy=np.array([x,y],dtype='float32'))
                        manifest.append({'file':name,'video_id':r['video_id'],'frame_id':mid,'split':r['split']})
                    fid+=1
            finally:cap.release()
    (out/'manifest.json').write_text(json.dumps({'samples':manifest,'note':'Positive-only smoke test data, NOT a full training set or unbiased validation; targets are middle-frame centers, sigma=2 at 512x288.'},indent=2),encoding='utf-8')
    assert {s:sum(r['split']==s for r in manifest) for s in counts}==counts,'Some requested samples were not decoded'
    print(json.dumps(counts))
if __name__=='__main__':main()
