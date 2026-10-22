#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs artifacts checkpoints data external
if [[ ! -x .envs/v2d/bin/python ]]; then
    conda create -p "$PWD/.envs/v2d" python=3.11 pip -y
fi
bash scripts/setup_sources.sh
.envs/v2d/bin/python -m pip install -r requirements-moge.txt
.envs/v2d/bin/python -m pip install --no-deps -e external/moge-pinned -e external/utils3d-pinned -e external/pipeline-pinned
.envs/v2d/bin/python -m pip install --no-deps -e external/video_to_data/reconstruction/modules/v2d_docker -e external/video_to_data/reconstruction/modules/v2d_cari4d/docker
.envs/v2d/bin/python -m pip check
.envs/v2d/bin/python -m pip freeze > artifacts/environment.lock.txt
