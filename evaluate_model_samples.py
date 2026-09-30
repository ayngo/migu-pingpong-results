"""Positive-frame location smoke test for the shipped ONNX model.

This is deliberately not an F1/competition evaluation: it samples annotated
positive frames only and uses a provisional directory-based split.
"""
import argparse
import csv
import json
import random
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-zip', required=True)
    ap.add_argument('--centers', default='results/ball_centers.csv')
    ap.add_argument('--model', default='results/model/ball.onnx')
    ap.add_argument('--out', default='results/model_positive_smoke.json')
    ap.add_argument('--per-split', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20260926)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    rng = random.Random(args.seed)

    by_split_video = defaultdict(lambda: defaultdict(list))
    with open(args.centers, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if 1 <= int(row['frame_id']):
                by_split_video[row['split']][row['video_id']].append(row)

    chosen = []
    for split in ('train', 'val'):
        vids = sorted(by_split_video[split])
        rng.shuffle(vids)
        for vid in vids[:args.per_split]:
            rows = by_split_video[split][vid]
            rng.shuffle(rows)
            chosen.append(rows[0])
    if not chosen:
        raise RuntimeError('No eligible annotated frames found')

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(root / args.model), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    mean = np.array([.485, .456, .406], np.float32)[None, None, None, :]
    std = np.array([.229, .224, .225], np.float32)[None, None, None, :]
    records = []
    with zipfile.ZipFile(args.train_zip) as archive, tempfile.TemporaryDirectory() as td:
        for row in chosen:
            video_member = 'pp_train_data/' + row['video_path']
            local_video = Path(td) / 'sample.mp4'
            local_video.write_bytes(archive.read(video_member))
            cap = cv2.VideoCapture(str(local_video))
            fid = int(row['frame_id'])
            frames = []
            try:
                for i in range(fid - 1, fid + 2):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, bgr = cap.read()
                    if not ok:
                        frames = []
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames.append(cv2.resize(rgb, (512, 288), interpolation=cv2.INTER_LINEAR))
            finally:
                cap.release()
            if len(frames) != 3:
                records.append({'video_id': row['video_id'], 'frame_id': fid,
                                'split': row['split'], 'status': 'decode_failed'})
                continue

            x = np.stack(frames).astype(np.float32) / 255.0
            x = ((x - mean) / std).transpose(0, 3, 1, 2).reshape(1, 9, 288, 512)
            heat = session.run(None, {input_name: np.ascontiguousarray(x)})[0][0, 1]
            ys, xs = np.nonzero(heat > .7)
            if not len(xs):
                records.append({'video_id': row['video_id'], 'frame_id': fid,
                                'split': row['split'], 'status': 'no_detection',
                                'heat_max': float(heat.max())})
                continue
            weights = heat[ys, xs].astype(np.float64)
            px = float(np.average(xs, weights=weights)) * int(row['width']) / 512
            py = float(np.average(ys, weights=weights)) * int(row['height']) / 288
            dx, dy = px - float(row['x']), py - float(row['y'])
            records.append({'video_id': row['video_id'], 'frame_id': fid,
                            'split': row['split'], 'status': 'detected',
                            'error_px': float(np.hypot(dx, dy)),
                            'within_50px': bool(np.hypot(dx, dy) <= 50),
                            'heat_max': float(heat.max())})

    summary = {'kind': 'positive-frame model smoke test; NOT official F1 or unbiased validation',
               'model': str(args.model), 'seed': args.seed,
               'requested_per_split': args.per_split, 'samples': len(records),
               'splits': {}}
    for split in ('train', 'val'):
        subset = [r for r in records if r.get('split') == split]
        detected = [r for r in subset if r.get('status') == 'detected']
        errors = [r['error_px'] for r in detected]
        summary['splits'][split] = {
            'samples': len(subset), 'detected': len(detected),
            'detection_rate_on_positive_frames': len(detected) / len(subset) if subset else None,
            'within_50px_count': sum(r['within_50px'] for r in detected),
            'median_error_px_when_detected': float(np.median(errors)) if errors else None,
            'p90_error_px_when_detected': float(np.percentile(errors, 90)) if errors else None,
        }
    summary['records'] = records
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary['splits'], ensure_ascii=False, indent=2))
    print(f'Full report: {out}')


if __name__ == '__main__':
    main()
