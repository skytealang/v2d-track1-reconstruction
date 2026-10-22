import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import h5py
import mediapipe as mp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='artifacts/inputs/episode_000000.0.color.mp4')
    parser.add_argument('--output', type=Path, default=Path('artifacts/pose_landmarks/episode_000000'))
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--masks', type=Path)
    parser.add_argument('--sequence', default='episode_000000')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    capture = cv2.VideoCapture(args.video)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, args.max_frames) if args.max_frames else total
    image_points = np.full((limit, 33, 4), np.nan, np.float32)
    world_points = image_points.copy()
    pose_api = mp.solutions.pose
    mask_file = h5py.File(args.masks, 'r') if args.masks else None
    with pose_api.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as estimator:
        for frame in range(limit):
            success, bgr = capture.read()
            if not success:
                raise ValueError(f'Cannot decode frame {frame}')
            height, width = bgr.shape[:2]
            left, top, right, bottom = 0, 0, width, height
            if mask_file is not None:
                mask = mask_file[f'{args.sequence}/{frame:06d}-k0.person_mask.png'][:] > 0
                rows, columns = np.where(mask)
                if len(rows):
                    margin = int(max(columns.max() - columns.min(), rows.max() - rows.min()) * 0.2)
                    left, right = max(0, int(columns.min()) - margin), min(width, int(columns.max()) + margin + 1)
                    top, bottom = max(0, int(rows.min()) - margin), min(height, int(rows.max()) + margin + 1)
            result = estimator.process(cv2.cvtColor(bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                for point in result.pose_landmarks.landmark:
                    point.x = (point.x * (right - left) + left) / width
                    point.y = (point.y * (bottom - top) + top) / height
                    point.z *= (right - left) / width
                image_points[frame] = [[point.x, point.y, point.z, point.visibility] for point in result.pose_landmarks.landmark]
                world_points[frame] = [[point.x, point.y, point.z, point.visibility] for point in result.pose_world_landmarks.landmark]
            if frame in [0, 100, 300, 500, limit - 1]:
                mp.solutions.drawing_utils.draw_landmarks(bgr, result.pose_landmarks, pose_api.POSE_CONNECTIONS)
                cv2.imwrite(str(args.output / f'{frame:06d}.jpg'), bgr)
            if frame % 100 == 0:
                print(json.dumps({'frames': frame + 1, 'seconds': time.monotonic() - started}), flush=True)
    capture.release()
    if mask_file is not None:
        mask_file.close()
    np.savez_compressed(args.output / 'landmarks.npz', image=image_points, world=world_points)
    report = {'frames': limit, 'detected_frames': int(np.isfinite(image_points[:, 0, 0]).sum()),
              'seconds': time.monotonic() - started, 'gpu_memory_mib': 0, 'backend': 'MediaPipe Pose Heavy CPU 0.10.21',
              'limitations': 'Predicted 33 landmarks only; world landmarks are hip-centered estimates, not scene-calibrated MHR meshes. No GT accuracy measured.'}
    report['model_files'] = {str(path.relative_to(Path(mp.__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (Path(mp.__file__).parent / 'modules/pose_landmark').glob('*.tflite')}
    (args.output / 'run.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
