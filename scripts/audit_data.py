import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v2d_challenge.dataset import Track1Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/official/track_1')
    parser.add_argument('--decode-all', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    dataset = Track1Dataset(args.root)
    sample = pd.read_parquet('external/submission_kit/v2d_submission_kit/data/track_1_sample_submission.parquet', columns=['row_id'])
    parts = sample['row_id'].str.split('_', expand=True)
    scored = sorted(parts[1].astype(int).unique().tolist())
    if scored != sorted(dataset.episodes):
        raise ValueError('Submission template and dataset episode rosters differ')
    report = []
    for episode in scored:
        index = dataset.index(episode)
        capture = cv2.VideoCapture(str(dataset.video_path(episode)))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rate = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if count != len(index) or (width, height) != (1536, 1152) or abs(rate - 30) > 0.001:
            raise ValueError(f'Episode {episode}: video metadata mismatch')
        rows = parts[parts[1].astype(int) == episode]
        scored_frames = sorted(set(rows[2].astype(int)) - {999999})
        if scored_frames and max(scored_frames) >= count:
            raise ValueError(f'Episode {episode}: scored frame outside video')
        decoded = None
        if args.decode_all or episode == scored[0]:
            decoded = 0
            for frame_id, rgb in dataset.frames(episode):
                if frame_id == 0:
                    cv2.imwrite(f'artifacts/episode_{episode:06d}_first.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                decoded += 1
        record = {'episode': episode, **dataset.metadata[episode], 'frames': count, 'decoded': decoded,
                  'width': width, 'height': height, 'fps': rate, 'scored_frames': scored_frames}
        report.append(record)
        print(f'Episode {episode:06d}: {count} frames, decoded={decoded}', flush=True)
    result = {'episodes': report, 'total_frames': sum(row['frames'] for row in report), 'seconds': time.monotonic() - started}
    Path('artifacts/data_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'episodes'}))


if __name__ == '__main__':
    main()
