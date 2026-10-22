#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-4}"
export FORCE_CUDA=1
export PYOPENGL_PLATFORM=egl
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/setup_sources.sh
if [[ ! -x .envs/cari4d/bin/python ]]; then
    conda create -p "$PWD/.envs/cari4d" python=3.11 pip -y
fi
runtime="$PWD/.envs/cari4d"
conda install -p "$runtime" -c conda-forge eigen=3.4 libboost-devel=1.86 pybind11 -y
export CPATH="$runtime/include/eigen3${CPATH:+:$CPATH}"
"$runtime/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"$runtime/bin/python" -m pip install -r requirements-native.lock.txt
"$runtime/bin/python" -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
fetch_source() {
    local repository="$1" revision="$2" destination="$3"
    if [[ ! -d "$destination" ]]; then
        local archive
        archive=$(mktemp)
        curl -fL --retry 2 "https://api.github.com/repos/$repository/tarball/$revision" -o "$archive"
        mkdir -p "$destination"
        tar -xzf "$archive" -C "$destination" --strip-components=1
        rm "$archive"
    fi
}
fetch_source facebookresearch/pytorch3d v0.7.9 external/pytorch3d
fetch_source NVlabs/nvdiffrast 253ac4fcea7de5f396371124af597e6cc957bfae external/nvdiffrast
fetch_source facebookresearch/sam2 2b90b9f5ceec907a1c18123530e92e794ad901a4 external/sam2
"$runtime/bin/python" -m pip install --no-build-isolation --no-deps ./external/pytorch3d ./external/nvdiffrast
"$runtime/bin/python" -m pip install --no-build-isolation --no-deps -e external/sam2 -e external/moge-pinned -e external/utils3d-pinned -e external/pipeline-pinned
modules="$PWD/external/video_to_data/reconstruction/modules"
for package in v2d_common v2d_mv v2d_mesh/lib v2d_foundation_pose/lib v2d_moge/lib v2d_sam3d_body/lib v2d_sam2/lib v2d_cari4d/lib; do
    "$runtime/bin/python" -m pip install --no-deps -e "$modules/$package"
done
foundation="$modules/v2d_foundation_pose/lib/FoundationPose"
cmake -S "$foundation/mycpp" -B "$foundation/mycpp/build" -DCMAKE_PREFIX_PATH="$runtime" -DPYTHON_EXECUTABLE="$runtime/bin/python"
cmake --build "$foundation/mycpp/build" -j "$MAX_JOBS"
"$runtime/bin/python" -m pip install --no-build-isolation --no-deps -e "$foundation/bundlesdf/mycuda"
if git -C external/video_to_data apply --check "$PWD/patches/cari4d-native-paths.patch"; then
    git -C external/video_to_data apply "$PWD/patches/cari4d-native-paths.patch"
else
    git -C external/video_to_data apply --reverse --check "$PWD/patches/cari4d-native-paths.patch"
fi
"$runtime/bin/python" scripts/fix_decord_metadata.py
"$runtime/bin/python" -m pip check
