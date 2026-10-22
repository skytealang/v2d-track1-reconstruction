import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v2d_challenge.dataset import Track1Dataset


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=Path, default=Path('configs/episode_000000_prompts.json'))
    parser.add_argument('--checkpoint', default='checkpoints/sam2-small/sam2.1_hiera_small.pt')
    parser.add_argument('--sam-config', default='configs/sam2.1/sam2.1_hiera_s.yaml')
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--root', default='data/official/track_1')
    parser.add_argument('--output', type=Path, default=Path('artifacts/inputs'))
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    annotations = json.loads(args.prompts.read_text())
    episode = annotations['episode']
    dataset = Track1Dataset(args.root)
    sequence = f'episode_{episode:06d}'
    source = dataset.video_path(episode)
    expected = dataset.episodes[episode]['length']
    limit = min(expected, args.max_frames) if args.max_frames else expected
    if limit <= 0:
        raise ValueError('Frame limit must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    video_alias = args.output / f'{sequence}.0.color.mp4'
    if not video_alias.exists():
        video_alias.symlink_to(source.resolve())
    output = args.output / f'{sequence}_masks_k0.h5'
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    identity = {'video_sha256': file_hash(source), 'checkpoint_sha256': file_hash(args.checkpoint),
                'prompts_sha256': file_hash(args.prompts), 'sam_config': args.sam_config,
                'fill_hole_area': 0, 'bidirectional': args.bidirectional}
    torch.cuda.reset_peak_memory_stats()
    predictor = build_sam2_video_predictor_low_mem(
        config_file=identity['sam_config'], ckpt_path=args.checkpoint, device='cuda')
    predictor.fill_hole_area = 0
    temporary = output.with_suffix('.partial')
    areas = []
    seen = set()
    capture = cv2.VideoCapture(str(source))
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        state = predictor.init_state(str(source), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for prompt in annotations['prompts']:
            kwargs = {'inference_state': state, 'frame_idx': prompt['frame_index'], 'obj_id': prompt['object_id']}
            if 'box' in prompt:
                kwargs['box'] = np.asarray(prompt['box'], dtype=np.float32)
            else:
                kwargs['points'] = np.asarray(prompt['points'], dtype=np.float32)
                kwargs['labels'] = np.asarray(prompt['labels'], dtype=np.int32)
            predictor.add_new_points_or_box(**kwargs)
        with h5py.File(temporary, 'w') as stream:
            stream.attrs.update({'schema': 'v2d.cari4d.wild_masks.v1', 'sequence': sequence,
                                 'frame_count': limit, 'width': state['video_width'], 'height': state['video_height'],
                                 'complete_video': limit == expected, 'identity': json.dumps(identity)})
            group = stream.create_group(sequence)
            propagation = predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=limit - 1)
            prompt_frames = sorted({prompt['frame_index'] for prompt in annotations['prompts']})
            if args.bidirectional and prompt_frames[-1] > 0:
                reverse = predictor.propagate_in_video(state, start_frame_idx=limit - 1,
                                                       max_frame_num_to_track=limit - 1, reverse=True)
                propagation = itertools.chain(propagation, reverse)
            for frame_id, object_ids, logits in propagation:
                if frame_id >= limit:
                    break
                if frame_id in seen:
                    past = max((value for value in prompt_frames if value <= frame_id), default=-limit)
                    future = min((value for value in prompt_frames if value >= frame_id), default=2 * limit)
                    if frame_id - past <= future - frame_id:
                        continue
                masks = {int(object_id): (logits[index, 0] > 0).cpu().numpy() for index, object_id in enumerate(object_ids)}
                if set(masks) != {0, 1}:
                    raise ValueError(f'Frame {frame_id}: expected human/object IDs 0 and 1')
                for object_id, role in [(0, 'person_mask'), (1, 'obj_rend_mask')]:
                    key = f'{frame_id:06d}-k0.{role}.png'
                    if key in group:
                        del group[key]
                    group.create_dataset(key, data=masks[object_id].astype(np.uint8) * 255,
                                         compression='lzf', shuffle=True)
                observation = {'frame': frame_id, 'human_pixels': int(masks[0].sum()), 'object_pixels': int(masks[1].sum())}
                if frame_id in seen:
                    areas[frame_id] = observation
                else:
                    areas.append(observation)
                seen.add(frame_id)
                if frame_id in [0, 100, 300, 500, limit - 1]:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    success, bgr = capture.read()
                    if not success:
                        raise ValueError('Unable to decode preview frame')
                    overlay = bgr.copy()
                    overlay[masks[0]] = 0.55 * overlay[masks[0]] + 0.45 * np.array([0, 220, 0])
                    overlay[masks[1]] = 0.35 * overlay[masks[1]] + 0.65 * np.array([0, 0, 255])
                    cv2.imwrite(str(args.output / f'{sequence}_masks_{frame_id:06d}.jpg'), overlay)
                if len(seen) % 50 == 0 or len(seen) == 1:
                    print(json.dumps({'frames': len(seen), 'seconds': time.monotonic() - started, **areas[-1]}), flush=True)
    capture.release()
    if seen != set(range(limit)):
        raise ValueError('Predicted masks do not cover every requested video frame')
    temporary.replace(output)
    report = {'episode': episode, 'frames': len(seen), 'expected_frames': expected, 'complete_video': limit == expected,
              'seconds': time.monotonic() - started, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'identity': identity, 'areas': areas, 'source': 'Track 1 RGB prompts and public SAM2.1; predicted masks, not GT'}
    (args.output / f'{sequence}_masks_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'areas'}), flush=True)


if __name__ == '__main__':
    main()
