"""Independent audit of the current group-wise provisional train/val split."""
import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path

VAL_GROUPS = {'03', '11', '13', '14', '20'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-zip', required=True)
    ap.add_argument('--out', default='results/split_review.json')
    ap.add_argument('--hash-videos', action='store_true',
                    help='Hash all source videos to check exact duplicate files')
    args = ap.parse_args()
    groups = collections.defaultdict(lambda: {
        'videos': 0, 'frames': 0, 'ball_present_frames': 0,
        'ball_missing_frames': 0, 'resolution_counts': collections.Counter(),
    })
    splits = {s: {'groups': [], 'videos': 0, 'frames': 0,
                  'ball_present_frames': 0, 'ball_missing_frames': 0,
                  'resolutions': collections.defaultdict(lambda: {
                      'videos': 0, 'frames': 0, 'ball_present_frames': 0,
                      'ball_missing_frames': 0})}
              for s in ('train', 'val')}
    missing_ball = {'total': 0, 'with_player': 0, 'with_table': 0,
                    'with_neither_player_nor_table': 0, 'runs': 0,
                    'runs_over_5_frames': 0, 'max_run_frames': 0}
    hashes = collections.defaultdict(list)
    with zipfile.ZipFile(args.train_zip) as archive:
        names = set(archive.namelist())
        annotation_names = sorted(n for n in names
                                  if '/annotations/' in n and n.endswith('.json'))
        video_members = set()
        for name in annotation_names:
            ann = json.loads(archive.read(name))
            group = name.split('/')[-2]
            split = 'val' if group in VAL_GROUPS else 'train'
            res = f"{ann['frame_width']}x{ann['frame_height']}"
            g = groups[group]
            s = splits[split]
            r = s['resolutions'][res]
            g['videos'] += 1
            g['frames'] += len(ann['frames'])
            g['resolution_counts'][res] += 1
            s['videos'] += 1
            s['frames'] += len(ann['frames'])
            s['resolutions'][res]['videos'] += 1
            s['resolutions'][res]['frames'] += len(ann['frames'])
            absent_run = 0
            for frame in ann['frames']:
                labels = {a['label']: len(a['bbox_xyxy']) for a in frame['annotations']}
                has_ball = bool(labels.get('ball', 0))
                if has_ball:
                    g['ball_present_frames'] += 1
                    s['ball_present_frames'] += 1
                    r['ball_present_frames'] += 1
                    if absent_run:
                        missing_ball['runs'] += 1
                        missing_ball['runs_over_5_frames'] += absent_run > 5
                        missing_ball['max_run_frames'] = max(missing_ball['max_run_frames'], absent_run)
                        absent_run = 0
                else:
                    g['ball_missing_frames'] += 1
                    s['ball_missing_frames'] += 1
                    r['ball_missing_frames'] += 1
                    missing_ball['total'] += 1
                    missing_ball['with_player'] += bool(labels.get('player', 0))
                    missing_ball['with_table'] += bool(labels.get('table', 0))
                    missing_ball['with_neither_player_nor_table'] += not labels.get('player', 0) and not labels.get('table', 0)
                    absent_run += 1
            if absent_run:
                missing_ball['runs'] += 1
                missing_ball['runs_over_5_frames'] += absent_run > 5
                missing_ball['max_run_frames'] = max(missing_ball['max_run_frames'], absent_run)

            video_member = 'pp_train_data/' + ann['video_path']
            video_members.add(video_member)
        for split, s in splits.items():
            s['groups'] = sorted(VAL_GROUPS if split == 'val' else set(groups) - VAL_GROUPS)
            for res in s['resolutions'].values():
                res['video_frame_share_pct'] = round(100 * res['frames'] / s['frames'], 2)
        if args.hash_videos:
            for member in sorted(video_members):
                digest = hashlib.sha256(archive.read(member)).hexdigest()
                hashes[digest].append(member)

    totals = {'videos': sum(v['videos'] for v in groups.values()),
              'frames': sum(v['frames'] for v in groups.values()),
              'annotation_groups': len(groups)}
    duplicate_sets = [v for v in hashes.values() if len(v) > 1]
    report = {
        'split_type': 'group-wise provisional split; group IDs are annotation directory prefixes',
        'validation_groups': sorted(VAL_GROUPS),
        'selection_reason': 'Hold out complete directories to avoid splitting clips within a source folder; include at least one group from each observed resolution while retaining groups of each resolution in training; validation contains about one fifth of all frames.',
        'totals': totals,
        'groups': {k: {**v, 'resolution_counts': dict(v['resolution_counts'])}
                   for k, v in sorted(groups.items())},
        'splits': splits,
        'missing_ball_annotation_review': {
            **missing_ball,
            'interpretation': 'No ball box is not proven to mean confirmed ball absence. Most such frames still annotate players and/or table, and some missing-ball runs are long; treat them as unlabeled/unknown unless annotation policy confirms negatives.'},
        'exact_video_duplicate_check': {
            'performed': args.hash_videos,
            'unique_video_members': len(hashes) if args.hash_videos else None,
            'duplicate_sets': duplicate_sets if args.hash_videos else None},
        'limitations': [
            'The archives do not document what a directory prefix represents; match/session identity is not verified.',
            'Camera/viewpoint labels are not present in the inspected annotation JSON, so this report does not claim 0/45/90 coverage.',
            'This split is suitable for controlled provisional experiments, not a substitute for the official held-out evaluation set.'
        ]
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'totals': totals, 'splits': splits,
                      'missing_ball_annotation_review': report['missing_ball_annotation_review'],
                      'exact_video_duplicate_check': report['exact_video_duplicate_check'],
                      'limitations': report['limitations']}, ensure_ascii=False, indent=2))
    print(f'Report: {out}')


if __name__ == '__main__':
    main()
