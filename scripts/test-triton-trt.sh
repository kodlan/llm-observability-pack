#!/bin/bash
# Test script for Triton with TensorRT-LLM backend

TRITON_URL="${TRITON_URL:-http://localhost:8000}"

echo "=== Testing Triton TensorRT-LLM API ==="
echo ""

# Health check
echo "1. Health check:"
curl -s "${TRITON_URL}/v2/health/ready" && echo " OK" || echo " FAILED"
echo ""

# List models
echo "2. List models:"
curl -s "${TRITON_URL}/v2/models" | python3 -m json.tool 2>/dev/null || curl -s "${TRITON_URL}/v2/models"
echo ""

# Model info
echo "3. Model info (qwen):"
curl -s "${TRITON_URL}/v2/models/qwen" | python3 -m json.tool 2>/dev/null || curl -s "${TRITON_URL}/v2/models/qwen"
echo ""

# Generate request using Triton inference API
echo "4. Generate request:"
curl -s "${TRITON_URL}/v2/models/qwen/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "text_input",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["Say hello in 5 words or less"]
      },
      {
        "name": "max_tokens",
        "shape": [1, 1],
        "datatype": "INT32",
        "data": [20]
      }
    ],
    "outputs": [
      {"name": "text_output"}
    ]
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"
echo ""

# Longer generate request
echo "5. Generate request (longer):"
curl -s "${TRITON_URL}/v2/models/qwen/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "text_input",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["What is C++? Explain in 10 sentences."]
      },
      {
        "name": "max_tokens",
        "shape": [1, 1],
        "datatype": "INT32",
        "data": [300]
      }
    ],
    "outputs": [
      {"name": "text_output"}
    ]
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"
echo ""

echo "=== Done ==="