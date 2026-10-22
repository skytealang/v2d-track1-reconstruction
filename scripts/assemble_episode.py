import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'external/submission_kit/v2d_submission_kit'))

from v2d_challenge.temporal import optimize_translation
from v2dlb.mhr_submission import _rotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human', type=Path, default=Path('artifacts/mhr_landmarks_full/human_params.npz'))
    parser.add_argument('--object', type=Path, default=Path('artifacts/object_pose_recovery'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/episodes'))
    parser.add_argument('--smoothness', type=float, default=10.0)
    args = parser.parse_args()
    started = time.monotonic()
    with np.load(args.human) as source:
        human = {key: source[key] for key in ['pose', 'scales', 'shape']}
    poses = np.load(args.object / 'poses.npy')
    expected = next(row['frames'] for row in json.loads(Path('artifacts/data_audit.json').read_text())['episodes'] if row['episode'] == 0)
    required = {'pose': (expected, 136), 'scales': (68,), 'shape': (45,)}
    for key, shape in required.items():
        if human[key].shape != shape or not np.isfinite(human[key]).all():
            raise ValueError(f'Invalid {key}: expected {shape}')
    if poses.shape != (expected, 4, 4) or not np.isfinite(poses).all():
        raise ValueError('Object poses must cover the complete video')
    _rotations(poses[:, :3, :3])
    report = json.loads((args.object / 'run.json').read_text())
    confidence = np.asarray([row['silhouette_iou'] for row in report['observations']])
    if len(confidence) != expected:
        raise ValueError('Object confidence coverage mismatch')
    translation = optimize_translation(poses[:, :3, 3], confidence, smoothness=args.smoothness)
    mesh_source = Path('artifacts/inputs/episode_000000_object.glb')
    mesh = trimesh.load(mesh_source, force='mesh')
    if not mesh.is_watertight or mesh.area <= 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError('Invalid reconstructed object mesh')
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'episode_000000.npz', **human, object_rotation=poses[:, :3, :3],
                        object_translation=translation.astype(np.float32), object_scale=np.asarray(1.0, np.float32))
    shutil.copyfile(mesh_source, args.output / 'episode_000000_object.glb')
    result = {'episode': 0, 'frames': expected, 'schema_checked': True, 'official_rotation_check_passed': True,
              'object_smoothness': args.smoothness, 'mean_translation_change_m': float(np.linalg.norm(translation - poses[:, :3, 3], axis=-1).mean()),
              'raw_acceleration_rms_m_per_frame_squared': float(np.sqrt(np.square(np.diff(poses[:, :3, 3], n=2, axis=0)).mean())),
              'smoothed_acceleration_rms_m_per_frame_squared': float(np.sqrt(np.square(np.diff(translation, n=2, axis=0)).mean())),
              'seconds': time.monotonic() - started, 'gpu_memory_mib': 0,
              'limitations': 'First episode only. Smoothness is an internal statistic, not official ACC error. Mean-shape landmark MHR fallback has unvalidated accuracy. All 30 episodes and a real GitHub commit URL are required before packing/upload.'}
    (args.output / 'episode_000000_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
