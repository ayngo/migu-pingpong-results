"""Check temporal sample tensors and measure provided model on real inputs."""
import json,time
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
root=Path(__file__).resolve().parent/'results'
manifest=json.loads((root/'smoke_samples/manifest.json').read_text())['samples']
opt=ort.SessionOptions();opt.intra_op_num_threads=4
session=ort.InferenceSession(str(root/'model/ball.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
mean=np.array([.485,.456,.406],dtype='float32')[None,None,None,:]
std=np.array([.229,.224,.225],dtype='float32')[None,None,None,:]
records=[];tiles=[]
# Small deterministic subset; this is a smoke test, not a representative metric.
chosen=[next(r for r in manifest if r['split']==s) for s in ('train','val')]
for r in chosen:
    d=np.load(root/'smoke_samples'/r['file']);rgb=d['rgb'];target=d['target'];center=d['center_xy']
    assert rgb.shape==(3,288,512,3) and rgb.dtype==np.uint8
    assert target.shape==(288,512) and np.isfinite(target).all()
    inp=np.ascontiguousarray(((rgb.astype('float32')/255-mean)/std).transpose(0,3,1,2).reshape(1,9,288,512))
    t=time.perf_counter();heat=session.run(None,{session.get_inputs()[0].name:inp})[0][0,1];sec=time.perf_counter()-t
    assert np.isfinite(heat).all()
    ys,xs=np.nonzero(heat>.7);pred=None
    if len(xs):
        weights=heat[ys,xs].astype('float64');pred=[float(np.average(xs,weights=weights)),float(np.average(ys,weights=weights))]
    records.append({'sample':r,'forward_sec':sec,'heat_max':float(heat.max()),'pred_center_512x288':pred,'gt_center_512x288':center.tolist(),'note':'Real sample smoke test only; no F1 or generalization claim.'})
    tile=cv2.cvtColor(rgb[1],cv2.COLOR_RGB2BGR);cv2.circle(tile,tuple(np.rint(center).astype(int)),7,(0,255,0),2)
    if pred:cv2.circle(tile,tuple(np.rint(pred).astype(int)),7,(0,0,255),2)
    cv2.putText(tile,r['split']+' '+r['video_id']+' green=GT red=prediction',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    tiles.append(tile)
cv2.imwrite(str(root/'sample_preview.jpg'),np.concatenate(tiles,axis=0))
(root/'real_sample_check.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
