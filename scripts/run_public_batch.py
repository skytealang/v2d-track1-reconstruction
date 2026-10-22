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


def prepare_prompts(dataset):
    specifications = json.loads((ROOT / 'configs/public_baseline_prompts.json').read_text())
    destination = ROOT / 'artifacts/public_prompts'
    destination.mkdir(parents=True, exist_ok=True)
    for episode in dataset.episodes:
        if episode == 0:
            continue
        specification = specifications['episodes'][str(episode)]
        prompts = [{'frame_index': 0, 'object_id': 0, 'box': [coordinate * 3 for coordinate in specification['human']]}]
        if 'object' in specification:
            prompts.append({'frame_index': 0, 'object_id': 1, 'box': [coordinate * 3 for coordinate in specification['object']]})
        else:
            prompts.append({'frame_index': 0, 'object_id': 1, 'points': [[coordinate * 3 for coordinate in point] for point in specification['points']],
                            'labels': specification['labels']})
        (destination / f'episode_{episode:06d}.json').write_text(json.dumps({'episode': episode, 'prompts': prompts,
                                                                         'provenance': specifications['provenance']}, indent=2))


def run_episode(episode, gpu, steps, smoke, status, lock):
    sequence = f'episode_{episode:06d}'
    log_root = ROOT / 'logs/public_batch' / sequence
    log_root.mkdir(parents=True, exist_ok=True)
    inputs = ROOT / 'artifacts/inputs'
    depth = ROOT / 'artifacts/moge2' / sequence
    landmarks = ROOT / 'artifacts/pose_landmarks' / sequence
    human = ROOT / 'artifacts/public_human' / sequence
    objects = ROOT / 'artifacts/public_objects' / sequence
    prompts = ROOT / 'artifacts/public_prompts' / f'{sequence}.json'
    native = str(ROOT / '.envs/cari4d/bin/python')
    lightweight = str(ROOT / '.envs/v2d/bin/python')
    pose_python = str(ROOT / '.envs/pose/bin/python')
    environment = os.environ.copy()
    environment.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                        'MKL_NUM_THREADS': '2', 'MAX_JOBS': '2', 'PYTHONUNBUFFERED': '1'})
    stages = [
        ('depth', lightweight, 'scripts/run_moge.py', ['--episode', str(episode)], depth / 'run.json'),
        ('masks', native, 'scripts/run_masks.py', ['--prompts', str(prompts), '--output', str(inputs)], inputs / f'{sequence}_masks_report.json'),
        ('mesh', native, 'scripts/reconstruct_visible_mesh.py', ['--episode', str(episode)], inputs / f'{sequence}_mesh_report.json'),
        ('landmarks', pose_python, 'scripts/run_pose_landmarks.py', ['--video', str(inputs / f'{sequence}.0.color.mp4'), '--output', str(landmarks),
         '--masks', str(inputs / f'{sequence}_masks_k0.h5'), '--sequence', sequence], landmarks / 'run.json'),
        ('human', native, 'scripts/fit_mhr_landmarks.py', ['--episode', str(episode), '--max-frames', '0', '--steps', str(steps), '--batch-size', '64',
         '--fit-head', '--output', str(human)], human / 'run.json'),
        ('object', native, 'scripts/run_object_pose.py', ['--episode', str(episode), '--max-frames', '0', '--recover-iou', '0.10', '--output', str(objects)], objects / 'run.json'),
        ('assemble', lightweight, 'scripts/assemble_episode.py', ['--episode', str(episode), '--human', str(human / 'human_params.npz'),
         '--object', str(objects)], ROOT / 'artifacts/episodes' / f'{sequence}_report.json'),
    ]
    started = time.monotonic()
    record = {'episode': episode, 'gpu': gpu, 'stages': [], 'complete': False}
    try:
        for stage, executable, script, arguments, marker in stages:
            if smoke and stage in ['depth', 'masks', 'landmarks', 'human', 'object']:
                if '--max-frames' in arguments:
                    arguments[arguments.index('--max-frames') + 1] = '16'
                else:
                    arguments += ['--max-frames', '16']
            if smoke and stage == 'assemble':
                break
            reused = False
            if marker.exists():
                previous = json.loads(marker.read_text())
                reused = stage in ['mesh', 'assemble'] or (previous.get('frames') == status['expected'][str(episode)] and not smoke)
            if reused:
                record['stages'].append({'stage': stage, 'reused': True})
                continue
            if stage == 'masks' and (inputs / f'{sequence}_masks_k0.h5').exists():
                arguments.append('--overwrite')
            command = [executable, '-u', script, *arguments]
            print(json.dumps({'episode': episode, 'gpu': gpu, 'stage': stage, 'status': 'running'}), flush=True)
            stage_started = time.monotonic()
            with (log_root / f'{stage}.log').open('w') as log:
                process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            record['stages'].append({'stage': stage, 'returncode': process.returncode, 'seconds': time.monotonic() - stage_started,
                                     'command': [executable, script, *arguments], 'log': str(log_root / f'{stage}.log')})
            if process.returncode:
                raise RuntimeError(f'{stage} failed with return code {process.returncode}')
        record['complete'] = not smoke
    except Exception as error:
        record['error'] = str(error)
    record['seconds'] = time.monotonic() - started
    (log_root / 'summary.json').write_text(json.dumps(record, indent=2))
    with lock:
        status['episodes'][str(episode)] = record
        (ROOT / 'artifacts/public_batch_status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps({'episode': episode, 'complete': record['complete'], 'error': record.get('error'), 'seconds': record['seconds']}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', nargs='+', type=int)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 7])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    dataset = Track1Dataset(ROOT / 'data/official/track_1')
    prepare_prompts(dataset)
    episodes = args.episodes if args.episodes is not None else list(range(1, 30))
    status = {'expected': {str(episode): entry['length'] for episode, entry in dataset.episodes.items()}, 'episodes': {}, 'started_at_unix': time.time()}
    lock = threading.Lock()
    groups = [episodes[index::len(args.gpus)] for index in range(len(args.gpus))]
    def worker(group, gpu):
        return [run_episode(episode, gpu, args.steps, args.smoke, status, lock) for episode in group]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, group, gpu) for group, gpu in zip(groups, args.gpus)]
        results = [record for future in concurrent.futures.as_completed(futures) for record in future.result()]
    status['seconds'] = time.time() - status['started_at_unix']
    (ROOT / 'artifacts/public_batch_status.json').write_text(json.dumps(status, indent=2))
    if any(record.get('error') for record in results):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
