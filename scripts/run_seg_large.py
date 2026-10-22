import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def run_worker(gpu, episodes):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    results = []
    for episode in episodes:
        sequence = f'episode_{episode:06d}'
        started = time.monotonic()
        report = Path(f'artifacts/seg_large/masks/{sequence}_masks_report.json')
        prompt_path = Path(f'artifacts/seg_large/prompts/{sequence}.json')
        prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        cached = json.loads(report.read_text()) if report.exists() else {}
        reused = cached.get('complete_video') and cached.get('identity', {}).get('prompts_sha256') == prompt_hash
        with Path(f'logs/seg_large/{sequence}_resume.log').open('w') as log:
            if not reused:
                subprocess.run(['.envs/cari4d/bin/python', 'scripts/run_masks.py', '--prompts',
                                f'artifacts/seg_large/prompts/{sequence}.json', '--checkpoint',
                                'checkpoints/sam2-large/sam2.1_hiera_large.pt', '--sam-config',
                                'configs/sam2.1/sam2.1_hiera_l.yaml', '--bidirectional',
                                '--output', 'artifacts/seg_large/masks', '--overwrite'], env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            previous = json.loads(report.read_text())
            annotations = json.loads(prompt_path.read_text())
            if 'object_crop' in annotations and (not reused or previous['identity'].get('object_crop') != annotations['object_crop']):
                subprocess.run(['.envs/cari4d/bin/python', 'scripts/refine_mask_crop.py', '--episode', str(episode)],
                               env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                previous = json.loads(report.read_text())
                reused = False
            assert previous['complete_video'] and previous['identity']['sam_config'].endswith('hiera_l.yaml')
            video = Path(f'artifacts/seg_large/{sequence}_masks_h264.mp4')
            valid = False
            if reused and video.exists() and video.stat().st_mtime_ns >= report.stat().st_mtime_ns:
                probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                        '-show_entries', 'stream=codec_name,nb_frames', '-of', 'json', str(video)],
                                       capture_output=True, text=True)
                if probe.returncode == 0:
                    stream = json.loads(probe.stdout)['streams'][0]
                    valid = stream['codec_name'] == 'h264' and int(stream.get('nb_frames', -1)) == previous['frames']
            if not valid:
                subprocess.run(['.envs/v2d/bin/python', 'scripts/render_mask_video.py', '--episode', str(episode),
                                '--masks-dir', 'artifacts/seg_large/masks', '--output', str(video)],
                               env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        results.append({'episode': episode, 'gpu': gpu, 'wall_seconds': time.monotonic() - started})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, nargs='+', default=list(range(30)))
    args = parser.parse_args()
    assert args.episodes and len(set(args.episodes)) == len(args.episodes)
    assert all(0 <= episode < 30 for episode in args.episodes)
    prompts_root = Path('artifacts/seg_large/prompts')
    prompts_root.mkdir(parents=True, exist_ok=True)
    Path('logs/seg_large').mkdir(parents=True, exist_ok=True)
    base = json.loads(Path('configs/public_baseline_prompts.json').read_text())
    corrections = json.loads(Path('configs/seg_large_corrections.json').read_text())['corrections']
    for episode in args.episodes:
        if episode == 0:
            annotations = json.loads(Path('configs/episode_000000_prompts.json').read_text())
        else:
            source = base['episodes'][str(episode)]
            prompts = [{'frame_index': 0, 'object_id': 0, 'box': [value * 3 for value in source['human']]}]
            if 'object' in source:
                prompts.append({'frame_index': 0, 'object_id': 1, 'box': [value * 3 for value in source['object']]})
            else:
                prompts.append({'frame_index': 0, 'object_id': 1,
                                'points': [[value * 3 for value in point] for point in source['points']], 'labels': source['labels']})
            annotations = {'episode': episode, 'prompts': prompts, 'provenance': base['provenance']}
        if str(episode) in corrections:
            correction = corrections[str(episode)]
            if 'crop' in correction:
                annotations['object_crop'] = [value * 2 for value in correction['crop']]
            if 'first_points' in correction:
                annotations['prompts'][1] = {'frame_index': 0, 'object_id': 1,
                                            'points': [[value * 2 for value in point] for point in correction['first_points']],
                                            'labels': correction['first_labels']}
            for keyframe in correction.get('keyframes', [correction]):
                assert len(keyframe['points']) == len(keyframe['labels']) > 0
                assert set(keyframe['labels']) <= {0, 1}
                assert all(0 <= point[0] < 768 and 0 <= point[1] < 576 for point in keyframe['points'])
                annotations['prompts'].append({'frame_index': keyframe['frame'], 'object_id': 1,
                                              'points': [[value * 2 for value in point] for point in keyframe['points']],
                                              'labels': keyframe['labels']})
        annotations['revision'] = 'sam21_large_rgb_review_v1'
        destination = prompts_root / f'episode_{episode:06d}.json'
        if not destination.exists() or json.loads(destination.read_text()) != annotations:
            destination.write_text(json.dumps(annotations, indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_worker, gpu, args.episodes[slot::3]) for slot, gpu in enumerate([0, 1, 2])]
        results = [row for future in futures for row in future.result()]
    Path('artifacts/seg_large/batch_report.json').write_text(json.dumps(results, indent=2))
    subprocess.run(['.envs/v2d/bin/python', 'scripts/review_segmentation.py'], check=True)


if __name__ == '__main__':
    main()
