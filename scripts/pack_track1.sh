#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${V2D_COMMIT_URL:?Set the real accessible GitHub /commit/<40-hex-sha> URL}"
mkdir -p artifacts/submissions
.envs/v2d/bin/python external/submission_kit/v2d_submission_kit/eval_reconstruction.py \
    --episodes artifacts/episodes \
    --sample external/submission_kit/v2d_submission_kit/data/track_1_sample_submission.parquet \
    --commit "$V2D_COMMIT_URL" --out artifacts/submissions/track1.parquet
