import argparse
import json
import os
from pathlib import Path

from huggingface_hub import get_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--inputs', type=Path, default=Path('artifacts/inputs'))
    parser.add_argument('--proxy')
    args = parser.parse_args()
    if args.proxy:
        os.environ['HTTP_PROXY'] = args.proxy
        os.environ['HTTPS_PROXY'] = args.proxy
    token = get_token()
    if token:
        os.environ['HF_TOKEN'] = token
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    os.environ['MAX_JOBS'] = '4'
    sequence = f'episode_{args.episode:06d}'
    required = [args.inputs / f'{sequence}.0.color.mp4', args.inputs / f'{sequence}_masks_k0.h5',
                args.inputs / f'{sequence}_object.glb',
                Path('checkpoints/cari4d/cari4d/2026-08-25-09-35-57/step200000.pth'),
                Path('checkpoints/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({'ready': False, 'missing': missing}))
        raise SystemExit(2)
    from v2d.cari4d.lib.run_inference import run_inference
    frames = next(row['frames'] for row in json.loads(Path('artifacts/data_audit.json').read_text())['episodes']
                  if row['episode'] == args.episode)
    run_inference(video_path=str(required[0]), mask_h5_path=str(required[1]), object_mesh_path=str(required[2]),
                  weights_path='checkpoints/cari4d', output_dir='artifacts/cari4d', expected_frames=frames,
                  moge_batch_size=1, sam3d_chunk_size=4, sam3d_refit_batch_size=64,
                  depth_alignment_render_batch_size=1, coconet_render_batch_size=4,
                  postopt_batch_size=96, render_batch_size=1)


if __name__ == '__main__':
    main()
