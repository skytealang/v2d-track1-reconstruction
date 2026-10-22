import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external/submission_kit/v2d_submission_kit'))
from v2dlb.mhr_submission import _rotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=Path, default=Path('artifacts/episodes'))
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/prediction_audit.json'))
    args = parser.parse_args()
    dataset = json.loads(Path('artifacts/data_audit.json').read_text())['episodes']
    sample = pd.read_parquet(ROOT / 'external/submission_kit/v2d_submission_kit/data/track_1_sample_submission.parquet')
    expected_episodes = set(sample['row_id'].str.split('_').str[1].astype(int))
    records = []
    missing = []
    for metadata in dataset:
        episode = metadata['episode']
        if episode not in expected_episodes:
            continue
        sequence = f'episode_{episode:06d}'
        prediction = args.episodes / f'{sequence}.npz'
        mesh_path = args.episodes / f'{sequence}_object.glb'
        if not prediction.exists() or not mesh_path.exists():
            missing.append(episode)
            continue
        count = metadata['frames']
        with np.load(prediction) as data:
            for name, shape in {'pose': (count, 136), 'scales': (68,), 'shape': (45,),
                                'object_rotation': (count, 3, 3), 'object_translation': (count, 3), 'object_scale': ()}.items():
                if data[name].shape != shape or not np.isfinite(data[name]).all():
                    raise ValueError(f'{sequence}: invalid {name}')
            _rotations(data['object_rotation'])
            if float(data['object_scale']) <= 0:
                raise ValueError('Object scale must be positive')
            translation_acceleration = float(np.sqrt(np.square(np.diff(data['object_translation'], n=2, axis=0)).mean()))
        mesh = trimesh.load(mesh_path, force='mesh')
        if len(mesh.faces) < 4 or not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
            raise ValueError(f'{sequence}: invalid mesh')
        records.append({'episode': episode, 'frames': count, 'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
                        'watertight': bool(mesh.is_watertight), 'volume_m3': float(mesh.volume),
                        'object_translation_acceleration_rms_m_per_frame_squared': translation_acceleration})
    result = {'episodes_directory': str(args.episodes.resolve()),
              'complete_episodes': len(records), 'expected_episodes': len(expected_episodes),
              'frames': sum(record['frames'] for record in records), 'missing_episodes': missing,
              'finite_arrays_and_rotations_valid': True, 'records': records,
              'note': 'Structural validation and prediction statistics only. No hidden reference or official accuracy score.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2))
    if missing and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
