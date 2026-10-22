import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    sequence = f'episode_{args.episode:06d}'
    root = Path('artifacts/seg_large')
    prompts_path = root / 'prompts' / f'{sequence}.json'
    annotations = json.loads(prompts_path.read_text())
    left, top, right, bottom = annotations['object_crop']
    assert 0 <= left < right <= 1536 and 0 <= top < bottom <= 1152
    prompts = [prompt for prompt in annotations['prompts'] if prompt['object_id'] == 1]
    assert prompts and all('points' in prompt for prompt in prompts)
    for prompt in prompts:
        assert all(left <= point[0] < right and top <= point[1] < bottom for point in prompt['points'])
    mask_path = root / 'masks' / f'{sequence}_masks_k0.h5'
    report_path = root / 'masks' / f'{sequence}_masks_report.json'
    report = json.loads(report_path.read_text())
    assert report['identity']['prompts_sha256'] == hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    expected = report['expected_frames']
    source = Path('data/official/track_1/videos/chunk-000/observation.images.exo_camera') / f'{sequence}.mp4'
    torch.cuda.reset_peak_memory_stats()
    predictor = build_sam2_video_predictor_low_mem(
        config_file='configs/sam2.1/sam2.1_hiera_l.yaml',
        ckpt_path='checkpoints/sam2-large/sam2.1_hiera_large.pt', device='cuda')
    predictor.fill_hole_area = 0
    with tempfile.TemporaryDirectory(prefix=f'{sequence}_crop_', dir=root) as temporary:
        crop_video = Path(temporary) / 'crop.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-vf',
                        f'crop={right-left}:{bottom-top}:{left}:{top}', '-an', '-c:v', 'libx264',
                        '-threads', '2', '-crf', '0', '-pix_fmt', 'yuv420p', str(crop_video)], check=True)
        output = Path(temporary) / 'masks.h5'
        shutil.copyfile(mask_path, output)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), h5py.File(output, 'r+') as stream:
            state = predictor.init_state(str(crop_video), offload_video_to_cpu=True, offload_state_to_cpu=True)
            assert state['num_frames'] == expected
            for prompt in prompts:
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=prompt['frame_index'], obj_id=1,
                    points=np.asarray(prompt['points'], np.float32) - [left, top],
                    labels=np.asarray(prompt['labels'], np.int32))
            prompt_frames = sorted({prompt['frame_index'] for prompt in prompts})
            seen = set()
            for reverse in [False, True]:
                for frame, object_ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=expected - 1 if reverse else 0,
                        max_frame_num_to_track=expected - 1, reverse=reverse):
                    if reverse:
                        past = max((value for value in prompt_frames if value <= frame), default=-expected)
                        future = min((value for value in prompt_frames if value >= frame), default=2 * expected)
                        if frame - past <= future - frame:
                            continue
                    assert list(object_ids) == [1]
                    crop_mask = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
                    assert crop_mask.shape == (bottom - top, right - left)
                    mask = np.zeros((1152, 1536), np.uint8)
                    mask[top:bottom, left:right] = crop_mask
                    stream[f'{sequence}/{frame:06d}-k0.obj_rend_mask.png'][:] = mask
                    report['areas'][frame]['object_pixels'] = int((crop_mask > 0).sum())
                    seen.add(frame)
            assert seen == set(range(expected))
            report['identity']['object_crop'] = annotations['object_crop']
            stream.attrs['identity'] = json.dumps(report['identity'])
        os.replace(output, mask_path)
    report['crop_refinement_seconds'] = time.monotonic() - started
    report['crop_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['episode', 'crop_refinement_seconds', 'crop_peak_allocated_mib']}))


if __name__ == '__main__':
    main()
