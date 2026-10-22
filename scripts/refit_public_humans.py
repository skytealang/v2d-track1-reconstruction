import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from v2d_challenge.dataset import Track1Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, nargs='+', default=list(range(30)))
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 7])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/episodes_v2')
    parser.add_argument('--human-root', type=Path, default=ROOT / 'artifacts/public_human_v2')
    args = parser.parse_args()
    dataset = Track1Dataset(ROOT / 'data/official/track_1')
    status_path = ROOT / 'artifacts/public_human_v2_status.json'
    status = {'started_at_unix': time.time(), 'episodes': {}, 'output': str(args.output)}
    lock = threading.Lock()

    def worker(episodes, gpu):
        for episode in episodes:
            started = time.monotonic()
            sequence = f'episode_{episode:06d}'
            human = args.human_root / sequence
            logs = ROOT / 'logs/public_human_v2' / sequence
            logs.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
                                'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
            record = {'episode': episode, 'gpu': gpu, 'complete': False, 'commands': []}
            try:
                marker = human / 'run.json'
                reusable = False
                if marker.exists():
                    previous = json.loads(marker.read_text())
                    reusable = previous.get('frames') == dataset.episodes[episode]['length'] and 'root_initialization' in previous
                if not reusable:
                    command = [str(ROOT / '.envs/cari4d/bin/python'), '-u', 'scripts/fit_mhr_landmarks.py',
                               '--episode', str(episode), '--max-frames', '0', '--steps', '200', '--batch-size', '64',
                               '--fit-head', '--output', str(human)]
                    record['commands'].append(command)
                    print(json.dumps({'episode': episode, 'stage': 'human', 'gpu': gpu}), flush=True)
                    with (logs / 'human.log').open('w') as log:
                        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(marker.read_text())
                if report['max_initial_hip_residual_m'] > 1e-5:
                    raise ValueError('MHR hip initialization does not match observed root')
                objects = ROOT / ('artifacts/object_pose_recovery' if episode == 0 else f'artifacts/public_objects/{sequence}')
                command = [str(ROOT / '.envs/v2d/bin/python'), 'scripts/assemble_episode.py', '--episode', str(episode),
                           '--human', str(human / 'human_params.npz'), '--object', str(objects), '--output', str(args.output)]
                record['commands'].append(command)
                with (logs / 'assemble.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                record.update({'complete': True, 'fit_seconds': report['seconds'],
                               'peak_allocated_mib': report['peak_allocated_mib'],
                               'max_initial_hip_residual_m': report['max_initial_hip_residual_m']})
            except Exception as error:
                record['error'] = str(error)
            record['seconds'] = time.monotonic() - started
            with lock:
                status['episodes'][str(episode)] = record
                status_path.write_text(json.dumps(status, indent=2))
            print(json.dumps(record), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, args.episodes[index::len(args.gpus)], gpu) for index, gpu in enumerate(args.gpus)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    status['seconds'] = time.time() - status['started_at_unix']
    status_path.write_text(json.dumps(status, indent=2))
    if any(not record['complete'] for record in status['episodes'].values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
