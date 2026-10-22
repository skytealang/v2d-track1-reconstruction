import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, nargs='+', default=list(range(30)))
    parser.add_argument('--gpus', type=int, nargs='+', default=[2, 3, 4, 5, 7])
    args = parser.parse_args()
    assert len(set(args.episodes)) == len(args.episodes) and all(0 <= value < 30 for value in args.episodes)
    assert args.gpus and len(set(args.gpus)) == len(args.gpus)
    root = Path('artifacts/clean_v3')
    inputs = root / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    logs = Path('logs/clean_v3')
    logs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def worker(episodes, gpu):
        results = []
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', MAX_JOBS='2')
        for episode in episodes:
            sequence = f'episode_{episode:06d}'
            masks = Path(f'artifacts/seg_large/masks/{sequence}_masks_k0.h5')
            report = json.loads(masks.with_name(f'{sequence}_masks_report.json').read_text())
            assert report['complete_video'] and report['identity']['sam_config'].endswith('hiera_l.yaml')
            prompts = Path(f'artifacts/seg_large/prompts/{sequence}.json')
            assert report['identity']['prompts_sha256'] == hashlib.sha256(prompts.read_bytes()).hexdigest()
            annotations = json.loads(prompts.read_text())
            assert 'object_crop' not in annotations or report['identity'].get('object_crop') == annotations['object_crop']
            identity = hashlib.sha256(masks.read_bytes()).hexdigest()
            record_path = root / f'{sequence}_run.json'
            if record_path.exists():
                previous = json.loads(record_path.read_text())
                assert previous['masks_sha256'] == identity, 'Mask changed during reconstruction'
                if previous.get('complete'):
                    results.append(previous)
                    continue
            for source, destination in [(masks, inputs / masks.name),
                                        (Path(f'data/official/track_1/videos/chunk-000/observation.images.exo_camera/{sequence}.mp4'),
                                         inputs / f'{sequence}.0.color.mp4')]:
                if not destination.exists():
                    destination.symlink_to(source.resolve())
            human = root / 'human' / sequence
            objects = root / 'objects' / sequence
            landmarks = root / 'landmarks' / sequence
            native = '.envs/cari4d/bin/python'
            light = '.envs/v2d/bin/python'
            mesh = [native, 'scripts/reconstruct_hoop.py', '--inputs', str(inputs)] if episode == 0 else [
                native, 'scripts/reconstruct_visible_mesh.py', '--episode', str(episode), '--inputs', str(inputs)]
            stages = [
                ('mesh', mesh),
                ('landmarks', ['.envs/pose/bin/python', 'scripts/run_pose_landmarks.py', '--video', str(inputs / f'{sequence}.0.color.mp4'),
                               '--output', str(landmarks), '--masks', str(masks), '--sequence', sequence]),
                ('human', [native, 'scripts/fit_mhr_landmarks.py', '--episode', str(episode), '--max-frames', '0',
                           '--steps', '200', '--batch-size', '64', '--fit-head', '--landmarks-root', str(root / 'landmarks'), '--output', str(human)]),
                ('object', [native, 'scripts/run_object_pose.py', '--episode', str(episode), '--max-frames', '0',
                            '--recover-iou', '0.10', '--inputs', str(inputs), '--output', str(objects)]),
                ('assemble', [light, 'scripts/assemble_episode.py', '--episode', str(episode), '--human', str(human / 'human_params.npz'),
                              '--object', str(objects), '--inputs', str(inputs), '--output', str(root / 'episodes')])]
            record = {'episode': episode, 'gpu': gpu, 'masks_sha256': identity, 'complete': False, 'stages': []}
            for name, command in stages:
                if name == 'assemble':
                    tracking = json.loads((objects / 'run.json').read_text())
                    if sum(row['silhouette_iou'] > 0 for row in tracking['observations']) < 2:
                        fallback = [light, 'scripts/refine_object_pose_anchor.py', '--episode', str(episode),
                                    '--inputs', str(inputs), '--poses', str(objects / 'poses.npy'),
                                    '--output', str(objects / 'poses.npy'), '--tracking-report', str(objects / 'run.json')]
                        with (logs / f'{sequence}_anchor.log').open('w') as log:
                            subprocess.run(fallback, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                        record['translation_fallback_command'] = fallback
                print(json.dumps({'episode': episode, 'stage': name, 'gpu': gpu}), flush=True)
                stage_started = time.monotonic()
                with (logs / f'{sequence}_{name}.log').open('w') as log:
                    process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                record['stages'].append({'stage': name, 'command': command, 'seconds': time.monotonic() - stage_started,
                                         'returncode': process.returncode})
                record_path.write_text(json.dumps(record, indent=2))
                if process.returncode:
                    raise RuntimeError(f'{sequence} {name} failed; see {logs}')
            assert hashlib.sha256(masks.read_bytes()).hexdigest() == identity
            record['complete'] = True
            record_path.write_text(json.dumps(record, indent=2))
            results.append(record)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, args.episodes[index::len(args.gpus)], gpu) for index, gpu in enumerate(args.gpus)]
        results = [row for future in futures for row in future.result()]
    assert len(results) == len(args.episodes) and all(row['complete'] for row in results)
    (root / 'batch_report.json').write_text(json.dumps({'seconds': time.monotonic() - started, 'episodes': results}, indent=2))


if __name__ == '__main__':
    main()
