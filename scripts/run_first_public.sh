#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs
.envs/v2d/bin/python scripts/run_moge.py --episode 0 > logs/moge-first-public.log 2>&1
if [[ ! -f artifacts/inputs/episode_000000_masks_k0.h5 ]]; then
    .envs/cari4d/bin/python scripts/run_masks.py --output artifacts/inputs > logs/masks-first-public.log 2>&1
fi
.envs/cari4d/bin/python scripts/reconstruct_hoop.py > logs/mesh-first-public.log 2>&1
.envs/cari4d/bin/python scripts/run_object_pose.py --max-frames 0 --recover-iou 0.10 --output artifacts/object_pose_recovery > logs/object-first-public.log 2>&1
.envs/pose/bin/python scripts/run_pose_landmarks.py > logs/landmarks-first-public.log 2>&1
.envs/cari4d/bin/python scripts/fit_mhr_landmarks.py --max-frames 0 --output artifacts/mhr_landmarks_full > logs/human-first-public.log 2>&1
.envs/v2d/bin/python scripts/assemble_episode.py > logs/assemble-first-public.log 2>&1
cat logs/assemble-first-public.log
