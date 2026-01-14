#!/bin/bash
# Simple test script for Triton with vLLM backend

TRITON_URL="${TRITON_URL:-http://localhost:8000}"

echo "=== Testing Triton API ==="
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

# Generate request
echo "4. Generate request:"
curl -s "${TRITON_URL}/v2/models/qwen/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "Say hello in 5 words or less",
    "parameters": {
      "max_tokens": 20
    }
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"
echo ""

# Longer generate request
echo "5. Generate request (longer):"
curl -s "${TRITON_URL}/v2/models/qwen/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "What is C++? Explain in 10 sentences.",
    "parameters": {
      "max_tokens": 300
    }
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"
echo ""

echo "=== Done ==="
