import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    paths = list(Path('checkpoints').rglob('*.pt')) + list(Path('checkpoints').rglob('*.pth'))
    paths += list(Path('checkpoints').rglob('*.yml'))
    records = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': checksum(path)}
               for path in sorted(set(paths)) if path.is_file()]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
              'local_commit': revision.stdout.strip() if revision.returncode == 0 else None,
              'checkpoints': records, 'sources': json.loads(Path('configs/sources.json').read_text()),
              'policy': 'Track 1 RGB/metadata only; no Track 2 GT assets; no credentials recorded.'}
    Path('artifacts/provenance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'checkpoint_files': len(records), 'bytes': sum(row['bytes'] for row in records),
                      'output': 'artifacts/provenance.json'}))


if __name__ == '__main__':
    main()
