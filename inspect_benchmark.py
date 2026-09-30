"""CPU diagnostic only. Not a competition FPS measurement."""
import argparse,json,time,collections,platform
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--runs',type=int,default=10);a=p.parse_args()
    m=onnx.load(a.model);onnx.checker.check_model(m)
    def desc(v):return {'name':v.name,'shape':[d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]}
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    sess=ort.InferenceSession(a.model,sess_options=opts,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(42);x=rng.random((1,9,288,512),dtype=np.float32)
    for _ in range(2):sess.run(None,{sess.get_inputs()[0].name:x})
    ts=[]
    for _ in range(a.runs):
        t=time.perf_counter();y=sess.run(None,{sess.get_inputs()[0].name:x});ts.append(time.perf_counter()-t)
    result={'kind':'CPU synthetic forward only; NOT official end-to-end FPS','platform':platform.platform(),'onnxruntime':ort.__version__,'threads':4,'inputs':[desc(v) for v in m.graph.input],'outputs':[desc(v) for v in m.graph.output],'opset':[{ 'domain':v.domain,'version':v.version} for v in m.opset_import],'operators':dict(collections.Counter(n.op_type for n in m.graph.node)),'initializer_elements':sum(int(np.prod(t.dims)) for t in m.graph.initializer),'onnx_check':'passed','runs':a.runs,'median_ms':float(np.median(ts)*1000),'p95_ms':float(np.percentile(ts,95)*1000),'output_finite':all(bool(np.isfinite(v).all()) for v in y),'output_range':[[float(v.min()),float(v.max())] for v in y]}
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
