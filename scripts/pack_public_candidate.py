import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'external/submission_kit/v2d_submission_kit'
sys.path.insert(0, str(KIT))
from tools import pack_reconstruction


def write_candidate(rows, sample, commit, output):
    merged = sample.merge(rows, on='row_id', how='left', validate='one_to_one')
    if not np.isfinite(merged[['x', 'y', 'z']].to_numpy()).all():
        raise ValueError('Missing or non-finite reconstruction values')
    if len(merged) != len(sample) or merged['row_id'].duplicated().any():
        raise ValueError('Candidate rows do not match the sample')
    output.parent.mkdir(parents=True, exist_ok=True)
    merged[['row_id', 'x', 'y', 'z']].to_parquet(output, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=Path, default=ROOT / 'artifacts/episodes')
    parser.add_argument('--commit')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output or ROOT / 'artifacts/submissions' / ('track1.parquet' if args.commit else 'track1_candidate_not_for_upload.parquet')
    if not args.commit:
        pack_reconstruction.write = write_candidate
    sample_path = KIT / 'data/track_1_sample_submission.parquet'
    pack_reconstruction.pack_track1(SimpleNamespace(episodes=args.episodes, sample=sample_path,
                                                   commit=args.commit, out=output))
    table = pd.read_parquet(output)
    sample = pd.read_parquet(sample_path)
    if not table['row_id'].equals(sample['row_id']):
        raise ValueError('Packed row ordering differs from official template')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report = {'output': str(output), 'rows': len(table), 'episodes': int(table['row_id'].str.split('_').str[1].nunique()),
              'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'local_code_commit': revision, 'commit_url': args.commit, 'seconds': time.monotonic() - started,
              'official_packer_mesh_and_array_checks': True, 'exact_template_rows_and_order': True,
              'ready_for_uploader_dry_run': bool(args.commit), 'upload_verified': False,
              'note': 'Without --commit this is a local reconstruction candidate, deliberately omitting code_commit_url. It must not be uploaded. With --commit the unmodified official writer validates the GitHub URL; uploader dry-run is still required.'}
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
