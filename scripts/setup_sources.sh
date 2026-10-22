#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external logs artifacts
if [[ ! -d external/video_to_data/.git ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/nvidia-isaac/video_to_data external/video_to_data
    GIT_LFS_SKIP_SMUDGE=1 git -C external/video_to_data checkout --detach 7eb9721088615fe9201634a6be5cf48b049d1090
fi
fetch_source() {
    local repository="$1" revision="$2" destination="$3"
    if [[ ! -d "$destination" ]]; then
        local archive
        archive=$(mktemp)
        curl -fL --retry 2 --connect-timeout 15 --max-time 180 "https://api.github.com/repos/$repository/tarball/$revision" -o "$archive"
        mkdir -p "$destination"
        tar -xzf "$archive" -C "$destination" --strip-components=1
        rm "$archive"
    fi
}
fetch_source microsoft/MoGe 925b8ed835a7a9cdb7578ba15c658a0afc969030 external/moge-pinned
fetch_source EasternJournalist/utils3d 3fab839f0be9931dac7c8488eb0e1600c236e183 external/utils3d-pinned
fetch_source EasternJournalist/pipeline 866f059d2a05cde05e4a52211ec5051fd5f276d6 external/pipeline-pinned
if [[ ! -f external/submission_kit/v2d_submission_kit/KIT_MANIFEST.json ]]; then
    git -C external/video_to_data lfs pull --include='docs/v2d_challenge/assets/v2d_submission_kit.zip,docs/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf'
    unzip -q external/video_to_data/docs/v2d_challenge/assets/v2d_submission_kit.zip -d external/submission_kit
fi
