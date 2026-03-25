#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

SRC_MODEL_DIR=${1:-/home/i_liuxinyu/models/MiniCPM-o-4_5-awq-w4a16}
OUT_DIR=${2:-/home/i_liuxinyu/models/minicpm-o-4_5-awq-w4a16-gguf}
REFERENCE_DIR=${3:-/home/i_liuxinyu/models/MiniCPM-o-4_5-gguf}
DOCKER_IMAGE=${DOCKER_IMAGE:-ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin}

LLM_OUT_NAME="MiniCPM-o-4_5-AWQ-W4A16.gguf"
LLM_OUT_PATH="${OUT_DIR}/${LLM_OUT_NAME}"

echo "============================================"
echo "MiniCPM-o 4.5 AWQ GGUF Layout Export"
echo "============================================"
echo "Repo:       ${REPO_DIR}"
echo "Source:     ${SRC_MODEL_DIR}"
echo "Reference:  ${REFERENCE_DIR}"
echo "Output:     ${OUT_DIR}"
echo "Docker:     ${DOCKER_IMAGE}"
echo "LLM output: ${LLM_OUT_PATH}"
echo

if [[ ! -d "${SRC_MODEL_DIR}" ]]; then
    echo "error: source model dir not found: ${SRC_MODEL_DIR}" >&2
    exit 1
fi

if [[ ! -d "${REFERENCE_DIR}" ]]; then
    echo "error: reference gguf dir not found: ${REFERENCE_DIR}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

for subdir in audio tts token2wav-gguf vision; do
    if [[ -d "${REFERENCE_DIR}/${subdir}" ]]; then
        mkdir -p "${OUT_DIR}/${subdir}"
        cp -a "${REFERENCE_DIR}/${subdir}/." "${OUT_DIR}/${subdir}/"
        echo "copied ${subdir}/ from reference"
    fi
done

echo
echo "Converting LLM to GGUF..."
docker run --rm \
    -v "${REPO_DIR}:/work/llama.cpp-omni" \
    -v "/home/i_liuxinyu/models:/work/models" \
    "${DOCKER_IMAGE}" \
    bash -lc "
        set -euo pipefail
        cd /work/llama.cpp-omni
        python3 -m pip install --no-input --editable gguf-py --disable-pip-version-check >/tmp/gguf-install.log 2>&1
        python3 convert_hf_to_gguf.py \
            /work/models/$(basename "${SRC_MODEL_DIR}") \
            --outfile /work/models/$(basename "${OUT_DIR}")/${LLM_OUT_NAME} \
            --outtype f16
    "

echo
echo "Done."
echo "Result directory:"
find "${OUT_DIR}" -maxdepth 3 \( -type d -o -type f \) | sort
