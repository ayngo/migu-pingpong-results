"""Run the shipped ONNX + shipped landing heuristic on public videos via CPU.

This copies no files out of participant.zip. The runner and heuristic are
ported verbatim from the inspected official solution; scoring uses one-to-one
matching with the published +/-1 frame and 50 px criteria.
"""
import argparse
import collections
import json
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)
IN_W, IN_H = 512, 288
SCORE_TH = .7
CONF_TH = .75
MIN_GAP = 10
FALL_AMP = 2.0
RISE_AMP = 1.0


def match_events(preds, gts):
    """Maximum-cardinality one-to-one TP matching under the task tolerances."""
    edges = []
    for gt in gts:
        choices = []
        for pi, pred in enumerate(preds):
            df = abs(int(pred['frame_id']) - int(gt['frame_id']))
            dist = float(np.hypot(pred['x'] - gt['x'], pred['y'] - gt['y']))
            if df <= 1 and dist <= 50:
                choices.append((df, dist, pi))
        edges.append([pi for _, _, pi in sorted(choices)])
    pred_to_gt = {}

    def augment(gi, seen):
        for pi in edges[gi]:
            if pi in seen:
                continue
            seen.add(pi)
            if pi not in pred_to_gt or augment(pred_to_gt[pi], seen):
                pred_to_gt[pi] = gi
                return True
        return False

    order = sorted(range(len(gts)), key=lambda i: len(edges[i]))
    for gi in order:
        augment(gi, set())
    return len(pred_to_gt)


def score(preds, gts):
    tp = match_events(preds, gts)
    fp, fn = len(preds) - tp, len(gts) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r,
            'f1': 2*p*r/(p+r) if p+r else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--participant-zip', required=True)
    ap.add_argument('--model', default='results/model/ball.onnx')
    ap.add_argument('--out', default='results/public_cpu_baseline.json')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-videos', type=int, default=0,
                    help='0 means all 15 videos; use 1 for pipeline smoke test')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), sess_options=opts,
                                   providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    blank = ((np.zeros((IN_H, IN_W, 3), np.float32) - MEAN) / STD).transpose(2, 0, 1)
    start_all = time.perf_counter()
    with zipfile.ZipFile(args.participant_zip) as archive:
        base = 'participant/pingpang/public_data/'
        manifest = json.loads(archive.read(base + 'manifest.json'))['videos']
        # The archive stores Chinese video paths as GBK while the manifest is
        # UTF-8. Python decodes non-UTF8 ZIP names as CP437; recover a lookup
        # key without extracting or renaming the original archive entries.
        video_infos = {}
        for info in archive.infolist():
            try:
                raw_name = info.filename.encode('cp437')
                try:
                    corrected = raw_name.decode('utf-8')
                except UnicodeDecodeError:
                    corrected = raw_name.decode('gbk')
            except (UnicodeEncodeError, UnicodeDecodeError):
                corrected = info.filename
            if corrected.startswith(base + 'videos/') and corrected.lower().endswith('.mp4'):
                video_infos[corrected] = info
        gt_by_video = collections.defaultdict(list)
        for line in archive.read(base + 'gt_public.jsonl').decode('utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get('valid'):
                    gt_by_video[row['video_id']].append(row)
        if args.max_videos:
            manifest = manifest[:args.max_videos]

        all_preds = []
        all_gts = []
        per_angle = collections.defaultdict(lambda: {'preds': [], 'gts': []})
        per_video = []
        with tempfile.TemporaryDirectory() as td:
            local_video = Path(td) / 'public.mp4'
            for vi, meta in enumerate(manifest, 1):
                member = base + meta['path']
                if member not in video_infos:
                    raise KeyError(f'ZIP entry not found after GBK path recovery: {member}')
                local_video.write_bytes(archive.read(video_infos[member]))
                cap = cv2.VideoCapture(str(local_video))
                frames = []
                try:
                    for fid in range(meta['n_frames']):
                        ok, bgr = cap.read()
                        if not ok:
                            raise RuntimeError(f"{meta['video_id']}: decode stopped at {fid}/{meta['n_frames']}")
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        x = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
                        x = x.astype(np.float32) / 255.0
                        x = (x - MEAN) / STD
                        frames.append(np.ascontiguousarray(x.transpose(2, 0, 1)))
                finally:
                    cap.release()

                # BallDetector appends the current frame to two blank/history
                # frames and reads output channel 1 (the middle of each triplet).
                windows = []
                for fid, current in enumerate(frames):
                    triplet = [blank, blank, current] if fid == 0 else (
                        [blank, frames[0], current] if fid == 1 else
                        [frames[fid-2], frames[fid-1], current])
                    windows.append(np.concatenate(triplet, axis=0))
                heats = []
                for lo in range(0, len(windows), args.batch_size):
                    batch = np.ascontiguousarray(np.stack(windows[lo:lo+args.batch_size]))
                    heats.extend(session.run(None, {input_name: batch})[0][:, 1])

                # Faithful port of Solution.process_frame/_emit, including
                # confidence thresholds, skipped detections, gap and tail flush.
                trajectory = []
                last_emit = -10_000
                preds = []

                def emit(need_future):
                    nonlocal last_emit
                    out = []
                    for j in range(1, len(trajectory) - need_future):
                        fid, x, y, _conf = trajectory[j]
                        if fid - last_emit < MIN_GAP:
                            continue
                        prev_y, next_y = trajectory[j-1][2], trajectory[j+1][2]
                        if (y > prev_y and y >= next_y
                                and y-prev_y >= FALL_AMP
                                and y-next_y >= RISE_AMP):
                            out.append({'video_id': meta['video_id'],
                                        'frame_id': max(0, fid-1), 'x': x, 'y': y})
                            last_emit = fid
                    return out

                for fid, heat in enumerate(heats):
                    ys, xs = np.nonzero(heat > SCORE_TH)
                    if len(xs):
                        weights = heat[ys, xs].astype(np.float64)
                        conf = float(heat[ys, xs].max())
                        if conf >= CONF_TH:
                            x = float(np.average(xs, weights=weights)) * meta['width'] / IN_W
                            y = float(np.average(ys, weights=weights)) * meta['height'] / IN_H
                            trajectory.append((fid, x, y, conf))
                    preds.extend(emit(2))
                preds.extend(emit(1))
                gts = gt_by_video[meta['video_id']]
                all_preds.extend(preds)
                all_gts.extend(gts)
                per_angle[meta['angle']]['preds'].extend(preds)
                per_angle[meta['angle']]['gts'].extend(gts)
                per_video.append({'video_id': meta['video_id'], 'angle': meta['angle'],
                                  'frames': meta['n_frames'], 'predictions': len(preds),
                                  'valid_gt': len(gts), **score(preds, gts)})
                print(f"[{vi}/{len(manifest)}] {meta['video_id']} {meta['angle']} "
                      f"frames={meta['n_frames']} pred={len(preds)} gt={len(gts)}",
                      flush=True)

    report = {
        'kind': 'CPU ONNX + ported official solution; public validation diagnostic',
        'score_definition': 'one-to-one maximum matching; abs(frame delta)<=1 and Euclidean pixel distance<=50',
        'backend': 'onnxruntime CPUExecutionProvider',
        'model': str(model_path), 'batch_size': args.batch_size,
        'videos_evaluated': len(manifest), 'frames_evaluated': sum(v['frames'] for v in per_video),
        'elapsed_seconds': time.perf_counter() - start_all,
        'overall': score(all_preds, all_gts),
        'by_angle': {angle: score(v['preds'], v['gts']) for angle, v in sorted(per_angle.items())},
        'per_video': per_video,
        'limitations': [
            'CPU ONNX output may differ slightly from TensorRT output.',
            'Scoring implements the published tolerances locally; compare with official evaluator when available.',
            'This public validation set is for diagnostics/model selection, not the hidden test set.'
        ]
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'overall': report['overall'], 'by_angle': report['by_angle'],
                      'videos_evaluated': report['videos_evaluated'],
                      'frames_evaluated': report['frames_evaluated'],
                      'elapsed_seconds': report['elapsed_seconds'], 'report': str(out)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
