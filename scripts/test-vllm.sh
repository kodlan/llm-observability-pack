#!/bin/bash
# Simple test script for vLLM API

VLLM_URL="${VLLM_URL:-http://localhost:8000}"

echo "=== Testing vLLM API ==="
echo ""

# Health check
echo "1. Health check:"
curl -s "${VLLM_URL}/health" && echo " OK" || echo " FAILED"
echo ""

# List models
echo "2. List models:"
curl -s "${VLLM_URL}/v1/models" | python3 -m json.tool 2>/dev/null || curl -s "${VLLM_URL}/v1/models"
echo ""

# Chat completion
echo "3. Chat completion (short):"
curl -s "${VLLM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in 5 words or less"}],
    "max_tokens": 20
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"
echo ""

# Longer chat completion
echo "4. Chat completion (longer):"
curl -s "${VLLM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "What is C++? Explain in 10 sentences."}],
    "max_tokens": 300
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"
echo ""

# Streaming chat completion
echo "5. Streaming chat completion:"
curl -s "${VLLM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "max_tokens": 30,
    "stream": true
  }'
echo ""
echo ""

echo "=== Done ==="
