import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/public_run_summary'))
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    stages = {
        'depth': 'artifacts/moge2/{sequence}/run.json',
        'masks': 'artifacts/inputs/{sequence}_masks_report.json',
        'mesh': 'artifacts/inputs/{sequence}_mesh_report.json',
        'landmarks': 'artifacts/pose_landmarks/{sequence}/run.json',
        'human': 'artifacts/public_human/{sequence}/run.json',
        'object': 'artifacts/public_objects/{sequence}/run.json',
        'assemble': 'artifacts/episodes/{sequence}_report.json',
    }
    for episode in range(30):
        sequence = f'episode_{episode:06d}'
        for stage, template in stages.items():
            path = Path(template.format(sequence=sequence))
            if episode == 0 and stage in ['human', 'object']:
                path = Path('artifacts/mhr_landmarks_full/run.json' if stage == 'human' else 'artifacts/object_pose_recovery/run.json')
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            observations = report.get('observations', [])
            records.append({'episode': episode, 'stage': stage, 'frames': report.get('frames', ''),
                            'seconds': report.get('seconds', ''), 'peak_allocated_mib': report.get('peak_allocated_mib', ''),
                            'mean_predicted_mask_iou': report.get('mean_silhouette_iou', ''),
                            'held_frames': sum(row.get('held_last_valid_pose', False) for row in observations),
                            'report': str(path)})
    with (args.output / 'stages.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    tiles = []
    for episode in range(30):
        sequence = f'episode_{episode:06d}'
        preview_directory = Path('artifacts/object_pose_recovery' if episode == 0 else f'artifacts/public_objects/{sequence}')
        previews = sorted(preview_directory.glob('*.jpg'))
        for position in [0, -1]:
            tile = np.zeros((204, 256, 3), np.uint8)
            if previews:
                image = cv2.imread(str(previews[position]))
                if image is not None:
                    tile[12:] = cv2.resize(image, (256, 192))
                label = f'{episode:02d} frame {previews[position].stem}'
            else:
                label = f'{episode:02d} pending'
            cv2.putText(tile, label, (4, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            tiles.append(tile)
    montage = np.concatenate([np.concatenate(tiles[index:index + 6], axis=1) for index in range(0, len(tiles), 6)])
    cv2.imwrite(str(args.output / 'object_tracking_first_last.jpg'), montage)
    summary = {'stage_reports': len(records), 'stage_seconds_sum': {}, 'stage_peak_allocated_mib': {},
               'object_iou_below_0_1': [row['episode'] for row in records if row['stage'] == 'object'
                                         and isinstance(row['mean_predicted_mask_iou'], (int, float))
                                         and row['mean_predicted_mask_iou'] < 0.1],
               'seconds': time.monotonic() - started,
               'note': 'Stage time sums are not wall time; overlapping GPU workers run concurrently. Episode 0 uses legacy human/object output directories. IoU compares predictions, not ground truth.'}
    for stage in stages:
        matching = [row for row in records if row['stage'] == stage]
        summary['stage_seconds_sum'][stage] = sum(row['seconds'] for row in matching if isinstance(row['seconds'], (int, float)))
        summary['stage_peak_allocated_mib'][stage] = max([row['peak_allocated_mib'] for row in matching if isinstance(row['peak_allocated_mib'], (int, float))], default=0)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
