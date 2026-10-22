import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import requests

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import get_token, snapshot_download


MODELS = [
    {'repo': 'nvidia/cari4d_commercial', 'revision': '1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9',
     'destination': 'checkpoints/cari4d/cari4d', 'patterns': ['2026-08-25-09-35-57/step200000.pth',
      '2026-08-25-09-35-57/resolved_config.yaml', '2026-08-25-09-35-57/manifest.json']},
    {'repo': 'facebook/sam-3d-body-dinov3', 'revision': '11aaa346c7204874a1cbafe3d39a979080b2c55a',
     'destination': 'checkpoints/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3'},
    {'repo': 'facebook/sam-3d-objects', 'destination': 'checkpoints/sam3d/hf-download'},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy')
    parser.add_argument('--download-approved', action='store_true')
    parser.add_argument('--include-objects', action='store_true')
    args = parser.parse_args()
    if args.proxy:
        os.environ['HTTP_PROXY'] = args.proxy
        os.environ['HTTPS_PROXY'] = args.proxy
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
    token = get_token()
    if not token:
        print(json.dumps({'token_available': False}))
        return
    records = []
    for specification in MODELS:
        repo = specification['repo']
        started = time.monotonic()
        record = {'repo': repo, 'authorized': False}
        try:
            response = requests.get(f'https://huggingface.co/api/models/{repo}/auth-check',
                                    headers={'Authorization': 'Bearer ' + token}, timeout=(10, 30))
            record.update({'status': response.status_code, 'authorized': response.status_code == 200})
            if response.status_code != 200:
                try:
                    reason = response.json().get('error', '')
                except ValueError:
                    reason = 'Non-JSON response'
                record['reason'] = re.sub(r'hf_[A-Za-z0-9]+', '[redacted]', reason.replace(token, '[redacted]'))
            elif args.download_approved and (repo != 'facebook/sam-3d-objects' or args.include_objects):
                print(json.dumps({'repo': repo, 'download_started': True}), flush=True)
                snapshot_download(repo_id=repo, revision=specification.get('revision'),
                                  allow_patterns=specification.get('patterns'), local_dir=specification['destination'],
                                  token=token, max_workers=2)
                record['download_complete'] = True
                if repo == 'nvidia/cari4d_commercial':
                    checkpoint = Path(specification['destination']) / '2026-08-25-09-35-57/step200000.pth'
                    checksum = hashlib.sha256()
                    with checkpoint.open('rb') as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            checksum.update(block)
                    if checksum.hexdigest() != '78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3':
                        raise ValueError('CARI4D checksum mismatch')
                    record['checkpoint_sha256'] = checksum.hexdigest()
        except Exception as error:
            record['error_type'] = type(error).__name__
        record['seconds'] = time.monotonic() - started
        records.append(record)
        print(json.dumps(record), flush=True)
    Path('artifacts/model_access.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
