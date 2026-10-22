import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh


def project(vertices, pose, k):
    points = vertices @ pose[:3, :3].T + pose[:3, 3]
    valid = points[:, 2] > 1e-5
    uv = np.zeros((len(points), 2), np.float32)
    uv[valid, 0] = k[0, 0] * points[valid, 0] / points[valid, 2] + k[0, 2]
    uv[valid, 1] = k[1, 1] * points[valid, 1] / points[valid, 2] + k[1, 2]
    return uv, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--poses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sequence = f'episode_{args.episode:06d}'
    inputs = Path('artifacts/inputs')
    mesh = trimesh.load(inputs / f'{sequence}_object.glb', force='mesh')
    vertices = np.asarray(mesh.vertices, np.float32)
    poses = np.load(args.poses).astype(np.float32)
    refined = poses.copy()
    diagnostics = []
    with h5py.File(inputs / f'{sequence}_masks_k0.h5') as masks:
        capture = cv2.VideoCapture(str(inputs / f'{sequence}.0.color.mp4'))
        for frame, pose in enumerate(poses):
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError(f'cannot decode frame {frame}')
            with np.load(f'artifacts/moge2/{sequence}/{frame:06d}.npz') as cache:
                depth = cache['depth'].astype(np.float32)
                valid_depth = cache['mask'] & np.isfinite(depth) & (depth > 0)
                normalized = cache['intrinsics_normalized']
            h, w = depth.shape
            observed = masks[f'{sequence}/{frame:06d}-k0.obj_rend_mask.png'][:]
            mask = cv2.resize(observed, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            support = mask & valid_depth
            ys, xs = np.where(support)
            if len(xs) < 24:
                diagnostics.append({'frame': frame, 'valid_pixels': int(len(xs)), 'anchored': False})
                continue
            k = np.array([[normalized[0, 0] * w, 0, normalized[0, 2] * w],
                          [0, normalized[1, 1] * h, normalized[1, 2] * h], [0, 0, 1]], np.float32)
            # Trim occluding foreground/background depth outliers before forming the 3-D anchor.
            z = depth[ys, xs]
            lo, hi = np.quantile(z, [0.15, 0.85])
            keep = (z >= lo) & (z <= hi)
            ys, xs, z = ys[keep], xs[keep], z[keep]
            if len(z) < 12:
                diagnostics.append({'frame': frame, 'valid_pixels': int(len(z)), 'anchored': False})
                continue
            target = np.array([np.median((xs + 0.5 - k[0, 2]) / k[0, 0] * z),
                               np.median((ys + 0.5 - k[1, 2]) / k[1, 1] * z),
                               np.median(z)], np.float32)
            rotated = vertices @ pose[:3, :3].T
            # Robust mesh center avoids changing the FoundationPose orientation estimate.
            local_center = np.median(rotated, axis=0)
            candidate = pose.copy()
            candidate[:3, 3] = target - local_center
            refined[frame] = candidate
            diagnostics.append({'frame': frame, 'valid_pixels': int(len(z)), 'anchored': True,
                                'translation_delta_m': float(np.linalg.norm(candidate[:3, 3] - pose[:3, 3]))})
        capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, refined)
    report = {'schema': 'v2d.object_pose_depth_anchor.v1', 'episode': args.episode,
              'frames': len(refined), 'anchored_frames': int(sum(row['anchored'] for row in diagnostics)),
              'mean_translation_delta_m': float(np.mean([row.get('translation_delta_m', 0) for row in diagnostics])),
              'source': str(args.poses), 'diagnostics': diagnostics}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'diagnostics'}, indent=2))


if __name__ == '__main__':
    main()
