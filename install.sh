#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One-shot installer for StreamingASR on the NeMo 26.02 container
# (PyTorch 2.10.0a0 / nv25.11 base / CUDA 13.0). Installs, in order:
#   1. k2          — pruned RNN-T loss
#   2. torchaudio  — built FROM SOURCE against the container's custom torch
#   3. torchcodec  — built FROM SOURCE (the torchaudio.load() backend, ta>=2.9)
#   4. bitsandbytes— the 8-bit AdamW optimizer (configs use optim.name: adamw_8bit)
#
# torchaudio / torchcodec must be built from source because NGC ships a custom
# torch build; PyPI wheels link against stock PyTorch's C++ ABI and will not load.
#
# Usage:
#   bash install.sh                      # all components (default)
#   bash install.sh k2 bitsandbytes      # only the listed components
#
# Env overrides:
#   K2_COMMIT, TORCHAUDIO_BRANCH, TORCHCODEC_TAG, BNB_VERSION,
#   USE_CUDA, USE_FFMPEG, BUILD_SOX (torchaudio), ENABLE_CUDA (torchcodec)

set -euo pipefail

# --- Component selection -----------------------------------------------------
COMPONENTS=("$@")
if [ ${#COMPONENTS[@]} -eq 0 ]; then
  COMPONENTS=(k2 torchaudio torchcodec bitsandbytes)
fi
want() { for c in "${COMPONENTS[@]}"; do [ "$c" = "$1" ] && return 0; done; return 1; }

# --- Configurable knobs (env overrides) --------------------------------------
K2_REPO="${K2_REPO:-https://github.com/k2-fsa/k2}"
K2_COMMIT="${K2_COMMIT:-e625cb971dbe945c6a0a67426bb2c1db0b8320d1}"  # fix for PyTorch 2.7.0+

TORCHAUDIO_BRANCH="${TORCHAUDIO_BRANCH:-release/2.9}"  # newest branch that builds vs torch 2.10/nv25.11
USE_CUDA="${USE_CUDA:-1}"        # build torchaudio CUDA ops (rnnt_loss, forced_align, CUCTC)
USE_FFMPEG="${USE_FFMPEG:-1}"
BUILD_SOX="${BUILD_SOX:-1}"
TORCHAUDIO_BUILD_DIR="${TORCHAUDIO_BUILD_DIR:-torchaudio_src}"

TORCHCODEC_TAG="${TORCHCODEC_TAG:-v0.10.0}"  # 0.10 matches torch 2.10
ENABLE_CUDA="${ENABLE_CUDA:-0}"  # audio decode is CPU/FFmpeg; CUDA (NVDEC) is video-only
TORCHCODEC_BUILD_DIR="${TORCHCODEC_BUILD_DIR:-torchcodec_src}"

BNB_VERSION="${BNB_VERSION:-0.49.2}"  # ships CUDA 13 support; verified vs this container

DEPENDENCIES_INSTALL_CMD="apt update && apt install -y ffmpeg sox libavdevice-dev"

# Clean up source-build dirs on exit (success OR failure), so a failed build
# never blocks a later run with a git-clone collision.
cleanup() { rm -rf "${TORCHAUDIO_BUILD_DIR}" "${TORCHCODEC_BUILD_DIR}"; }
trap cleanup EXIT

TORCH_FULL_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "================================================================"
echo "StreamingASR installer"
echo "  Torch:      ${TORCH_FULL_VERSION}"
echo "  Components: ${COMPONENTS[*]}"
echo "================================================================"

# --- System deps for torchaudio / torchcodec ---------------------------------
if want torchaudio || want torchcodec; then
  echo; echo "### system audio backends (ffmpeg / sox / libavdevice) ###"
  for lib in libavdevice sox; do
    if ! grep -q "${lib}" <<< "$(ldconfig -p)"; then
      echo "ERROR: ${lib} not found. Install dependencies first: '${DEPENDENCIES_INSTALL_CMD}'"
      exit 1
    fi
  done
  if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found. Install dependencies first: '${DEPENDENCIES_INSTALL_CMD}'"
    exit 1
  fi
fi

# --- 1) k2 -------------------------------------------------------------------
if want k2; then
  echo; echo "### [1/4] installing k2 ###"
  pip install wheel setuptools cmake
  K2_MAKE_ARGS="-j" pip install -v --no-build-isolation "git+${K2_REPO}@${K2_COMMIT}#egg=k2" \
    || { echo "k2 could not be installed!"; exit 1; }
  python3 -m k2.version > /dev/null \
    || { echo "k2 installed with errors! Please check installation manually."; exit 1; }
  echo "k2 installed successfully!"
fi

# --- 2) torchaudio (from source) ---------------------------------------------
if want torchaudio; then
  echo; echo "### [2/4] building torchaudio (${TORCHAUDIO_BRANCH}) from source ###"
  TA_VER="${TORCHAUDIO_BRANCH#release/}"
  BUILD_VERSION="${BUILD_VERSION:-${TA_VER}.0}"
  echo "  build version: ${BUILD_VERSION} | USE_CUDA=${USE_CUDA} USE_FFMPEG=${USE_FFMPEG} BUILD_SOX=${BUILD_SOX}"

  rm -rf "${TORCHAUDIO_BUILD_DIR}"
  git clone --depth 1 --branch "${TORCHAUDIO_BRANCH}" https://github.com/pytorch/audio.git "${TORCHAUDIO_BUILD_DIR}"
  ( cd "${TORCHAUDIO_BUILD_DIR}"
    git submodule update --init --recursive
    # PYTORCH_VERSION must match the installed torch for CUDA support (some NGC
    # images set it but wrong).
    USE_CUDA="${USE_CUDA}" \
    USE_FFMPEG="${USE_FFMPEG}" \
    BUILD_SOX="${BUILD_SOX}" \
    PYTORCH_VERSION="${TORCH_FULL_VERSION}" \
    BUILD_VERSION="${BUILD_VERSION}" \
      python setup.py install )
  rm -rf "${TORCHAUDIO_BUILD_DIR}"

  python3 - <<'PY'
import torchaudio
import torchaudio.transforms as T
T.MFCC()
print("torchaudio", torchaudio.__version__, "imported OK")
PY
  if [[ "${USE_CUDA}" == "1" ]]; then
    python3 -c "from torchaudio.functional import rnnt_loss; print('torchaudio CUDA ops present:', rnnt_loss is not None)" \
      || echo "WARNING: torchaudio built but CUDA op import failed."
  fi
  echo "torchaudio installed successfully!"
fi

# --- 3) torchcodec (from source) ---------------------------------------------
if want torchcodec; then
  echo; echo "### [3/4] building torchcodec (${TORCHCODEC_TAG}) from source ###"
  if ! pkg-config --exists libavutil libavcodec libavformat; then
    echo "ERROR: FFmpeg dev headers not found. Install dependencies first: '${DEPENDENCIES_INSTALL_CMD}'"
    exit 1
  fi
  pip install -q wheel setuptools cmake ninja pybind11
  echo "  ENABLE_CUDA=${ENABLE_CUDA}"

  rm -rf "${TORCHCODEC_BUILD_DIR}"
  git clone --depth 1 --branch "${TORCHCODEC_TAG}" https://github.com/pytorch/torchcodec.git "${TORCHCODEC_BUILD_DIR}"
  # I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 builds against the *system* FFmpeg
  # (a license acknowledgement for a local, non-distributed build -- not an error).
  # --no-build-isolation / --no-deps: build against the installed (NGC) torch and
  # never let pip swap it out.
  ( cd "${TORCHCODEC_BUILD_DIR}"
    ENABLE_CUDA="${ENABLE_CUDA}" \
    I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
      pip install . --no-build-isolation --no-deps )
  rm -rf "${TORCHCODEC_BUILD_DIR}"

  python3 - <<'PY'
import torch, torchaudio, torchcodec
from torchcodec.decoders import AudioDecoder  # noqa: F401
print("torchcodec", torchcodec.__version__, "loaded OK")
PY
  echo "torchcodec installed successfully!"
fi

# --- 4) bitsandbytes (8-bit AdamW optimizer) ---------------------------------
if want bitsandbytes; then
  echo; echo "### [4/4] installing bitsandbytes (${BNB_VERSION}) ###"
  # --no-deps so pip never swaps out the container's custom torch for a stock wheel.
  pip install --no-deps "bitsandbytes==${BNB_VERSION}"
  python3 - <<'PY'
import bitsandbytes as bnb
from bitsandbytes.optim import AdamW8bit  # the optimizer the training scripts register
print("bitsandbytes", bnb.__version__, "AdamW8bit OK")
PY
  echo "bitsandbytes installed successfully!"
fi

echo; echo "All requested components installed successfully!"
