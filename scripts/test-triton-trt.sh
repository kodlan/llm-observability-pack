#!/bin/bash
# Test script for Triton with TensorRT-LLM backend
# Requires: pip install transformers requests numpy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_URL="${TRITON_URL:-http://localhost:8000}"

python3 "$SCRIPT_DIR/test-triton-trt.py" --url "$TRITON_URL"