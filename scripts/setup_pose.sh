#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .envs/pose/bin/python ]]; then
    .envs/v2d/bin/python -m venv .envs/pose
fi
.envs/pose/bin/python -m pip install -r requirements-pose.lock.txt
.envs/pose/bin/python -m pip check
