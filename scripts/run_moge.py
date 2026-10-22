import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v2d_challenge.dataset import Track1Dataset


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--root', default='data/official/track_1')
    parser.add_argument('--checkpoint', default='checkpoints/moge2/model.pt')
    parser.add_argument('--output', type=Path, default=Path('artifacts/moge2'))
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--tokens', type=int, default=1200)
    args = parser.parse_args()
    if args.width < 32 or args.tokens < 100 or args.max_frames < 0:
        raise ValueError('Invalid inference size, token count or frame limit')
    started = time.monotonic()
    dataset = Track1Dataset(args.root)
    source = dataset.video_path(args.episode)
    destination = args.output / f'episode_{args.episode:06d}'
    destination.mkdir(parents=True, exist_ok=True)
    identity = {'source_sha256': sha256(source), 'checkpoint_sha256': sha256(args.checkpoint),
                'width': args.width, 'tokens': args.tokens, 'mixed_precision': True,
                'moge_revision': '925b8ed835a7a9cdb7578ba15c658a0afc969030',
                'utils3d_revision': '3fab839f0be9931dac7c8488eb0e1600c236e183', 'torch': torch.__version__}
    marker = destination / 'identity.json'
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError('Cached inputs or settings changed: choose a new output directory')
    marker.write_text(json.dumps(identity, indent=2))
    torch.cuda.reset_peak_memory_stats()
    model = MoGeModel.from_pretrained(args.checkpoint).to('cuda').eval()
    processed = 0
    reused = 0
    for frame_id, rgb in dataset.frames(args.episode):
        if args.max_frames and frame_id >= args.max_frames:
            break
        output_path = destination / f'{frame_id:06d}.npz'
        if output_path.exists():
            with np.load(output_path) as cache:
                if {'depth', 'mask', 'intrinsics_normalized', 'source_size'}.issubset(cache.files):
                    processed += 1
                    reused += 1
                    continue
            raise ValueError(f'Invalid cache: {output_path}')
        original_height, original_width = rgb.shape[:2]
        height = round(original_height * args.width / original_width)
        resized = cv2.resize(rgb, (args.width, height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255).to('cuda')
        try:
            with torch.inference_mode():
                predictions = model.infer(tensor, num_tokens=args.tokens, use_fp16=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError('OOM with one frame: rerun with --width 512 --tokens 800 and a new --output') from None
        arrays = {key: value.detach().cpu().numpy() for key, value in predictions.items() if key in ['depth', 'mask', 'intrinsics']}
        depth = arrays['depth']
        mask = arrays['mask'] & np.isfinite(depth) & (depth > 0)
        if not mask.any():
            raise ValueError(f'Frame {frame_id}: no valid predicted depth')
        temporary = output_path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, depth=depth.astype(np.float32), mask=mask,
                                intrinsics_normalized=arrays['intrinsics'], source_size=np.array([original_width, original_height]))
        temporary.replace(output_path)
        if frame_id == 0:
            low, high = np.percentile(depth[mask], [2, 98])
            scaled = np.zeros(depth.shape, dtype=np.uint8)
            scaled[mask] = np.clip((depth[mask] - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
            colored[~mask] = 0
            cv2.imwrite(str(destination / 'first_depth.jpg'), colored)
        processed += 1
        if processed % 25 == 0 or processed == 1:
            print(f'frames={processed} elapsed={time.monotonic() - started:.2f}s peak_allocated={torch.cuda.max_memory_allocated() / 2**20:.1f}MiB', flush=True)
        del predictions, tensor
    torch.cuda.synchronize()
    report = {'episode': args.episode, 'frames': processed, 'reused_frames': reused,
              'expected_frames': dataset.episodes[args.episode]['length'],
              'complete_video': processed == dataset.episodes[args.episode]['length'],
              'seconds': time.monotonic() - started, 'gpu': torch.cuda.get_device_name(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
              'identity': identity, 'scope': 'MoGe 2 depth initialization only; no MHR/object reconstruction or submission'}
    (destination / 'run.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
