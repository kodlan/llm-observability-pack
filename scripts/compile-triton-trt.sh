#!/bin/bash
# Compile Qwen model to TensorRT-LLM format
# This script runs the compilation inside the TensorRT-LLM container
# Usage: ./compile-triton-trt.sh [MODEL_NAME] [MAX_BATCH_SIZE] [MAX_INPUT_LEN] [MAX_OUTPUT_LEN]

set -e

MODEL_NAME=${1:-Qwen/Qwen2.5-1.5B-Instruct}
MAX_BATCH_SIZE=${2:-8}
MAX_INPUT_LEN=${3:-1024}
MAX_OUTPUT_LEN=${4:-1024}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PROJECT_ROOT/serving/triton-trt/model_repository/qwen/1"

# Triton TRT-LLM container (has tensorrt_llm package installed)
TRTLLM_IMAGE="nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3"

# TensorRT-LLM version matching the container
TRTLLM_VERSION="v0.16.0"

echo "=== TensorRT-LLM Model Compilation ==="
echo "Model: $MODEL_NAME"
echo "Max batch size: $MAX_BATCH_SIZE"
echo "Max input length: $MAX_INPUT_LEN"
echo "Max output length: $MAX_OUTPUT_LEN"
echo "Engine output: $ENGINE_DIR"
echo ""

# Check if engine already exists
if [ -f "$ENGINE_DIR/rank0.engine" ]; then
    echo "Engine already exists at $ENGINE_DIR"
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf "$ENGINE_DIR"/*
fi

# Create directories
mkdir -p "$ENGINE_DIR"
CACHE_DIR="$PROJECT_ROOT/serving/triton-trt/.cache"
CHECKPOINT_DIR="$CACHE_DIR/checkpoint"
HF_CACHE_DIR="$CACHE_DIR/huggingface"
MODEL_DIR="$CACHE_DIR/models/$(echo $MODEL_NAME | tr '/' '_')"
EXAMPLES_DIR="$CACHE_DIR/tensorrt_llm_examples"
mkdir -p "$CHECKPOINT_DIR" "$HF_CACHE_DIR" "$MODEL_DIR"

# Clone TensorRT-LLM examples if not present
if [ ! -d "$EXAMPLES_DIR/examples/qwen" ]; then
    echo "Cloning TensorRT-LLM examples..."
    rm -rf "$EXAMPLES_DIR"
    git clone --depth 1 --branch $TRTLLM_VERSION \
        https://github.com/NVIDIA/TensorRT-LLM.git "$EXAMPLES_DIR"
fi

echo ""
echo "Step 1/3: Downloading model from HuggingFace..."
echo "Model will be saved to: $MODEL_DIR"
echo ""

# Download the model first using the container
docker run --rm \
    -v "$MODEL_DIR:/model" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    $TRTLLM_IMAGE \
    python3 -c "
from huggingface_hub import snapshot_download
import os

model_name = '$MODEL_NAME'
local_dir = '/model'

print(f'Downloading {model_name}...')
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir
)
print(f'Model downloaded to {local_dir}')
print('Contents:')
for f in os.listdir(local_dir):
    print(f'  {f}')
"

echo ""
echo "Step 2/3: Converting model to TensorRT-LLM checkpoint..."
echo ""

# Run conversion using the downloaded model directory
docker run --rm \
    --gpus all \
    -v "$CHECKPOINT_DIR:/checkpoint" \
    -v "$MODEL_DIR:/model:ro" \
    -v "$EXAMPLES_DIR/examples:/examples:ro" \
    $TRTLLM_IMAGE \
    python3 /examples/qwen/convert_checkpoint.py \
        --model_dir /model \
        --output_dir /checkpoint \
        --dtype float16 \
        --tp_size 1

echo ""
echo "Step 3/3: Building TensorRT-LLM engine..."
docker run --rm \
    --gpus all \
    -v "$CHECKPOINT_DIR:/checkpoint" \
    -v "$ENGINE_DIR:/engine" \
    $TRTLLM_IMAGE \
    trtllm-build \
        --checkpoint_dir /checkpoint \
        --output_dir /engine \
        --gemm_plugin float16 \
        --gpt_attention_plugin float16 \
        --context_fmha disable \
        --max_batch_size $MAX_BATCH_SIZE \
        --max_input_len $MAX_INPUT_LEN \
        --max_seq_len $((MAX_INPUT_LEN + MAX_OUTPUT_LEN)) \
        --paged_kv_cache enable \
        --remove_input_padding enable

echo ""
echo "=== Compilation complete ==="
echo "Engine files written to: $ENGINE_DIR"
ls -la "$ENGINE_DIR"

echo ""
echo "You can now start the service with: make up-triton-trt"