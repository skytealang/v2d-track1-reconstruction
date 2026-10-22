#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
episode="${1:-000000}"
video="artifacts/inputs/episode_${episode}.0.color.mp4"
masks="artifacts/inputs/episode_${episode}_masks_k0.h5"
mesh="artifacts/inputs/episode_${episode}_object.glb"
for required in "$video" "$masks" "$mesh"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing predicted input: $required" >&2
        exit 2
    fi
done
if ! command -v docker >/dev/null || ! docker info >/dev/null 2>&1; then
    echo 'A working GPU-enabled Docker runtime is required by the official launcher.' >&2
    exit 2
fi
frames=$(.envs/v2d/bin/python -c 'import json,sys; rows=json.load(open("artifacts/data_audit.json"))["episodes"]; print(next(r["frames"] for r in rows if r["episode"] == int(sys.argv[1])))' "$episode")
.envs/v2d/bin/python -m v2d.cari4d.docker.run_inference \
    --video_path "$video" --mask_h5_path "$masks" --object_mesh_path "$mesh" \
    --weights_path checkpoints/cari4d --output_dir artifacts/cari4d \
    --expected_frames "$frames" --moge_batch_size 1 --sam3d_chunk_size 4 \
    --sam3d_refit_batch_size 64 --depth_alignment_render_batch_size 1 \
    --coconet_render_batch_size 4 --postopt_batch_size 96 --render_batch_size 1
