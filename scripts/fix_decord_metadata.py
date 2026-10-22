import importlib.metadata
import re
from pathlib import Path


def main():
    distribution = importlib.metadata.distribution('decord')
    candidate = next(entry for entry in distribution.files if str(entry).endswith('.dist-info/WHEEL'))
    path = Path(distribution.locate_file(candidate))
    content = path.read_text()
    expected = re.search(r'Tag: cp36-cp36m-(manylinux\S+)', content)
    if expected:
        import decord
        if decord.__version__ != '0.6.0':
            raise RuntimeError('Unexpected decord version')
        path.write_text(content.replace(expected.group(0), 'Tag: py3-none-' + expected.group(1)))
        print('Corrected decord 0.6.0 ctypes wheel metadata')
    else:
        print('No decord metadata correction required')


if __name__ == '__main__':
    main()
