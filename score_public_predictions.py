"""Score a public-validation predictions.jsonl against packaged public GT."""
import argparse
import collections
import json
from pathlib import Path


def maximum_matches(preds, gts):
    edges = []
    for gt in gts:
        choices = []
        for pi, p in enumerate(preds):
            dt = abs(p['frame_id'] - gt['frame_id'])
            dx, dy = p['x'] - gt['x'], p['y'] - gt['y']
            dist2 = dx*dx + dy*dy
            if dt <= 1 and dist2 <= 50*50:
                choices.append((dt, dist2, pi))
        edges.append([pi for _, _, pi in sorted(choices)])
    pred_to_gt = {}

    def visit(gi, seen):
        for pi in edges[gi]:
            if pi in seen:
                continue
            seen.add(pi)
            if pi not in pred_to_gt or visit(pred_to_gt[pi], seen):
                pred_to_gt[pi] = gi
                return True
        return False

    for gi in sorted(range(len(gts)), key=lambda i: len(edges[i])):
        visit(gi, set())
    return len(pred_to_gt)


def metric(preds, gts):
    tp = maximum_matches(preds, gts)
    fp, fn = len(preds) - tp, len(gts) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall)
            if precision+recall else 0.0}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', default='public_score.json')
    args = ap.parse_args()

    manifest_doc = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    angle_by_id = {v['video_id']: v['angle'] for v in manifest_doc['videos']}
    gt_by_id = collections.defaultdict(list)
    for row in read_jsonl(args.gt):
        if row.get('valid'):
            gt_by_id[row['video_id']].append(row)
    pred_by_id = collections.defaultdict(list)
    for row in read_jsonl(args.predictions):
        vid = row['video_id']
        if vid not in angle_by_id:
            raise ValueError(f'Prediction has unknown video_id: {vid}')
        pred_by_id[vid].append({'video_id': vid, 'frame_id': int(row['frame_id']),
                                'x': float(row['x']), 'y': float(row['y'])})

    angle_preds, angle_gts = collections.defaultdict(list), collections.defaultdict(list)
    for vid, angle in angle_by_id.items():
        angle_preds[angle].extend(pred_by_id[vid])
        angle_gts[angle].extend(gt_by_id[vid])
    all_preds = [p for group in angle_preds.values() for p in group]
    all_gts = [g for group in angle_gts.values() for g in group]
    report = {
        'kind': 'local public-set scoring; verify matching implementation against official scorer when available',
        'criteria': 'one-to-one maximum matching; abs(frame delta)<=1 and Euclidean pixel distance<=50',
        'videos_in_manifest': len(angle_by_id),
        'predictions': len(all_preds),
        'valid_ground_truth': len(all_gts),
        'overall': metric(all_preds, all_gts),
        'by_angle': {a: metric(angle_preds[a], angle_gts[a])
                     for a in sorted(angle_by_id.values())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
