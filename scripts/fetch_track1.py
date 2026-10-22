import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='https://huggingface.co')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('data/official'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    entries = json.loads(args.tree.read_text())
    files = [entry for entry in entries if entry['type'] == 'file']
    if any(not entry['path'].startswith('track_1/') or '..' in Path(entry['path']).parts for entry in files):
        raise ValueError('Only Track 1 paths are allowed')

    def fetch(entry):
        target = args.root / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get('lfs', {}).get('oid')
        started = time.monotonic()
        if target.exists() and target.stat().st_size == entry['size'] and (not expected or digest(target) == expected):
            return {'path': entry['path'], 'cached': True, 'sha256': digest(target)}
        url = f"{args.endpoint}/datasets/nvidia/video_to_data_challenge/resolve/{args.revision}/{entry['path']}?download=true"
        temporary = target.with_name(target.name + '.partial')
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'curl/7.81.0'})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                if temporary.stat().st_size != entry['size']:
                    raise ValueError('Downloaded size mismatch')
                checksum = digest(temporary)
                if expected and checksum != expected:
                    raise ValueError('LFS SHA256 mismatch')
                temporary.replace(target)
                result = {'path': entry['path'], 'sha256': checksum, 'seconds': round(time.monotonic() - started, 2)}
                print(json.dumps(result), flush=True)
                return result
            except Exception as error:
                if attempt == 2:
                    status = getattr(error, 'code', None)
                    raise RuntimeError(f"{entry['path']}: {type(error).__name__} status={status}: download failed") from None
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(fetch, files))
    manifest = {'repo': 'nvidia/video_to_data_challenge', 'revision': args.revision, 'endpoint': args.endpoint, 'files': records}
    Path('artifacts/download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Verified {len(records)} Track 1 files')


if __name__ == '__main__':
    main()
