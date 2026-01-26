#!/bin/bash
# Load test for Triton with TensorRT-LLM backend
# Usage: ./load-test-triton-trt.sh [CONCURRENCY] [TRITON_URL]
# Requires: pip install transformers requests numpy
# Stop with Ctrl+C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONCURRENCY=${1:-3}
TRITON_URL=${2:-http://localhost:8000}

python3 "$SCRIPT_DIR/load-test-triton-trt.py" --url "$TRITON_URL" --concurrency "$CONCURRENCY"