import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')
os.environ.setdefault('MAX_JOBS', '4')

import cv2
import h5py
import numpy as np
import torch

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
from v2d.common.datatypes import Image as V2dImage
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
from v2d.mesh.lib.mesh import Mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=16)
    parser.add_argument('--recover-iou', type=float, default=0.0)
    parser.add_argument('--recovery-interval', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('artifacts/object_pose_smoke'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    sequence = f'episode_{args.episode:06d}'
    inputs = Path('artifacts/inputs')
    mesh = Mesh.load(str(inputs / f'{sequence}_object.glb'), force_mesh=True)
    tracker = FoundationPoseTracker(mesh, 'checkpoints/cari4d/foundationpose', backend='nvlabs_pytorch')
    capture = cv2.VideoCapture(str(inputs / f'{sequence}.0.color.mp4'))
    poses = []
    observations = []
    last_recovery_frame = -args.recovery_interval
    with h5py.File(inputs / f'{sequence}_masks_k0.h5') as masks, torch.inference_mode():
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = min(count, args.max_frames) if args.max_frames else count
        for frame in range(limit):
            success, bgr = capture.read()
            if not success:
                raise ValueError(f'Failed to decode frame {frame}')
            with np.load(f'artifacts/moge2/{sequence}/{frame:06d}.npz') as cache:
                depth = cache['depth'].copy()
                depth[~cache['mask'] | ~np.isfinite(depth) | (depth <= 0)] = 0
                normalized = cache['intrinsics_normalized']
            height, width = depth.shape
            intrinsics = CameraIntrinsics(float(normalized[0, 0] * width), float(normalized[1, 1] * height),
                                          float(normalized[0, 2] * width), float(normalized[1, 2] * height), width, height)
            rgb = V2dImage(data=cv2.cvtColor(cv2.resize(bgr, (width, height)), cv2.COLOR_BGR2RGB))
            depth_image = DepthImage(depth=depth.astype(np.float32))
            observed = masks[f'{sequence}/{frame:06d}-k0.obj_rend_mask.png'][:]
            mask = Mask(mask=cv2.resize(observed, (width, height), interpolation=cv2.INTER_NEAREST) > 0)
            if frame == 0:
                pose = tracker.register(rgb, depth_image, mask, intrinsics, iteration=3)
            else:
                pose = tracker.track_one(rgb, depth_image, intrinsics, iteration=2)
            matrix = pose.to_matrix()
            held = False
            if not np.isfinite(matrix).all() or matrix[2, 3] <= 0:
                if poses:
                    matrix = poses[-1].copy()
                    held = True
                else:
                    matrix = np.load(inputs / f'{sequence}_initial_object_pose.npy')
                if not np.isfinite(matrix).all() or matrix[2, 3] <= 0:
                    raise ValueError(f'No valid observation-derived initialization at frame {frame}')
                pose = Transform3d.from_matrix(matrix)
                tracker.reset_to_pose(pose)
            poses.append(matrix)
            iou = tracker._mask_iou(mask, intrinsics, pose)
            if held:
                iou = 0.0
            recovered = False
            visible_depth = np.count_nonzero(mask.mask & (depth > 0))
            rejected_recovery = False
            if frame > 0 and iou < args.recover_iou and visible_depth >= 16 and frame - last_recovery_frame >= args.recovery_interval:
                last_recovery_frame = frame
                candidate = tracker.register(rgb, depth_image, mask, intrinsics, iteration=3)
                candidate_matrix = candidate.to_matrix()
                candidate_valid = np.isfinite(candidate_matrix).all() and candidate_matrix[2, 3] > 0
                candidate_iou = tracker._mask_iou(mask, intrinsics, candidate) if candidate_valid else -1.0
                if candidate_iou > iou:
                    pose, matrix, iou = candidate, candidate_matrix, candidate_iou
                    poses[-1] = matrix
                    recovered = True
                else:
                    tracker.reset_to_pose(pose)
                    rejected_recovery = True
            observations.append({'frame': frame, 'silhouette_iou': iou, 'recovered': recovered,
                                 'held_last_valid_pose': held, 'rejected_recovery': rejected_recovery,
                                 'valid_mask_depth_pixels': int(visible_depth)})
            if frame in [0, 100, 300, 500, limit - 1]:
                camera_points = mesh.to_trimesh().vertices @ matrix[:3, :3].T + matrix[:3, 3]
                projected = camera_points[:, :2] / camera_points[:, 2:] @ np.diag([intrinsics.fx, intrinsics.fy])
                projected += [intrinsics.cx, intrinsics.cy]
                preview = cv2.resize(bgr, (width, height))
                for point in np.round(projected[camera_points[:, 2] > 0]).astype(np.int32):
                    cv2.circle(preview, tuple(point), 1, (0, 0, 255), -1)
                cv2.imwrite(str(args.output / f'{frame:06d}.jpg'), preview)
            np.save(args.output / f'{frame:06d}.npy', matrix)
            if frame % 25 == 0 or frame == limit - 1:
                print(json.dumps({'frames': len(poses), 'seconds': time.monotonic() - started}), flush=True)
    capture.release()
    np.save(args.output / 'poses.npy', np.asarray(poses))
    report = {'frames': len(poses), 'complete_video': len(poses) == count, 'seconds': time.monotonic() - started,
              'episode': args.episode,
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
              'backend': tracker.runtime_info, 'register_iterations': 3, 'track_iterations': 2,
              'recover_iou': args.recover_iou,
              'recovery_interval': args.recovery_interval,
              'observations': observations, 'mean_silhouette_iou': float(np.mean([row['silhouette_iou'] for row in observations])),
              'limitations': 'Predicted monocular depth and reconstructed approximate mesh; no GT accuracy measured. Symmetric shapes have rotational ambiguity.'}
    (args.output / 'run.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'observations'}), flush=True)


if __name__ == '__main__':
    main()
