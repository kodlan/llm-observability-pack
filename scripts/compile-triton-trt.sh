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

# TensorRT-LLM image
TRTLLM_IMAGE="nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3"

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

# Create temp directory for checkpoints
CHECKPOINT_DIR=$(mktemp -d)
trap "rm -rf $CHECKPOINT_DIR" EXIT

echo "Pulling TensorRT-LLM image (if needed)..."
docker pull $TRTLLM_IMAGE

echo ""
echo "Step 1/2: Converting HuggingFace model to TensorRT-LLM checkpoint..."
docker run --rm \
    --gpus all \
    -v "$SCRIPT_DIR:/scripts:ro" \
    -v "$CHECKPOINT_DIR:/checkpoints" \
    -v "$ENGINE_DIR:/engine" \
    -e HF_HOME=/checkpoints/hf_cache \
    $TRTLLM_IMAGE \
    python3 /scripts/convert_checkpoint.py \
        --model-name "$MODEL_NAME" \
        --output-dir /checkpoints/trtllm_ckpt \
        --dtype float16

echo ""
echo "Step 2/2: Building TensorRT-LLM engine..."
docker run --rm \
    --gpus all \
    -v "$CHECKPOINT_DIR:/checkpoints" \
    -v "$ENGINE_DIR:/engine" \
    $TRTLLM_IMAGE \
    trtllm-build \
        --checkpoint_dir /checkpoints/trtllm_ckpt \
        --output_dir /engine \
        --gemm_plugin float16 \
        --gpt_attention_plugin float16 \
        --max_batch_size $MAX_BATCH_SIZE \
        --max_input_len $MAX_INPUT_LEN \
        --max_seq_len $((MAX_INPUT_LEN + MAX_OUTPUT_LEN)) \
        --paged_kv_cache enable \
        --remove_input_padding enable \
        --use_fused_mlp enable

echo ""
echo "=== Compilation complete ==="
echo "Engine files written to: $ENGINE_DIR"
ls -la "$ENGINE_DIR"