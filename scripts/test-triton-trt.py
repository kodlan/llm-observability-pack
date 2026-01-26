#!/usr/bin/env python3
"""Test script for Triton with TensorRT-LLM backend.

Uses client-side tokenization since TRT-LLM backend expects pre-tokenized input.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests

# Add project root to path for importing
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOKENIZER_DIR = PROJECT_ROOT / "serving/triton-trt/.cache/models/Qwen_Qwen2.5-1.5B-Instruct"


def load_tokenizer():
    """Load the tokenizer from the downloaded model."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers library required. Install with: pip install transformers")
        sys.exit(1)

    if not TOKENIZER_DIR.exists():
        print(f"ERROR: Tokenizer not found at {TOKENIZER_DIR}")
        print("Run 'make compile-triton-trt' first to download the model.")
        sys.exit(1)

    print(f"Loading tokenizer from {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
    return tokenizer


def create_inference_request(input_ids: list, max_tokens: int = 50):
    """Create a Triton inference request with tokenized input."""
    input_ids_array = np.array([input_ids], dtype=np.int32)
    input_lengths = np.array([[len(input_ids)]], dtype=np.int32)
    request_output_len = np.array([[max_tokens]], dtype=np.int32)

    return {
        "inputs": [
            {
                "name": "input_ids",
                "shape": list(input_ids_array.shape),
                "datatype": "INT32",
                "data": input_ids_array.flatten().tolist()
            },
            {
                "name": "input_lengths",
                "shape": [1, 1],
                "datatype": "INT32",
                "data": input_lengths.flatten().tolist()
            },
            {
                "name": "request_output_len",
                "shape": [1, 1],
                "datatype": "INT32",
                "data": request_output_len.flatten().tolist()
            }
        ],
        "outputs": [
            {"name": "output_ids"},
            {"name": "sequence_length"}
        ]
    }


def send_request(triton_url: str, request_data: dict):
    """Send inference request to Triton."""
    url = f"{triton_url}/v2/models/qwen/infer"
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=request_data)
    return response


def decode_response(response_json: dict, tokenizer):
    """Decode the output tokens from Triton response."""
    outputs = {out["name"]: out for out in response_json.get("outputs", [])}

    if "output_ids" not in outputs:
        return None, "No output_ids in response"

    output_data = outputs["output_ids"]["data"]
    output_shape = outputs["output_ids"]["shape"]

    # Reshape and decode
    output_ids = np.array(output_data, dtype=np.int32).reshape(output_shape)

    # Get the generated tokens (first sequence)
    generated_ids = output_ids[0].tolist()

    # Decode to text
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, None


def main():
    parser = argparse.ArgumentParser(description="Test Triton TensorRT-LLM backend")
    parser.add_argument("--url", default="http://localhost:8000", help="Triton server URL")
    args = parser.parse_args()

    triton_url = args.url

    print("=== Testing Triton TensorRT-LLM API ===")
    print()

    # 1. Health check
    print("1. Health check:")
    try:
        resp = requests.get(f"{triton_url}/v2/health/ready")
        print(f"   Status: {'OK' if resp.status_code == 200 else 'FAILED'}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return
    print()

    # 2. Model info
    print("2. Model info (qwen):")
    try:
        resp = requests.get(f"{triton_url}/v2/models/qwen")
        print(f"   {json.dumps(resp.json(), indent=2)[:500]}...")
    except Exception as e:
        print(f"   FAILED: {e}")
    print()

    # 3. Load tokenizer
    print("3. Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"   Tokenizer loaded: {tokenizer.__class__.__name__}")
    print()

    # 4. Generate request (short)
    print("4. Generate request (short):")
    prompt = "Say hello in 5 words or less"
    print(f"   Prompt: {prompt}")

    input_ids = tokenizer.encode(prompt)
    print(f"   Tokenized: {len(input_ids)} tokens")

    request_data = create_inference_request(input_ids, max_tokens=20)
    response = send_request(triton_url, request_data)

    if response.status_code == 200:
        text, error = decode_response(response.json(), tokenizer)
        if error:
            print(f"   Decode error: {error}")
        else:
            print(f"   Response: {text}")
    else:
        print(f"   ERROR {response.status_code}: {response.text[:200]}")
    print()

    # 5. Generate request (longer)
    print("5. Generate request (longer):")
    prompt = "What is Python? Explain in 3 sentences."
    print(f"   Prompt: {prompt}")

    input_ids = tokenizer.encode(prompt)
    print(f"   Tokenized: {len(input_ids)} tokens")

    request_data = create_inference_request(input_ids, max_tokens=100)
    response = send_request(triton_url, request_data)

    if response.status_code == 200:
        text, error = decode_response(response.json(), tokenizer)
        if error:
            print(f"   Decode error: {error}")
        else:
            print(f"   Response: {text}")
    else:
        print(f"   ERROR {response.status_code}: {response.text[:200]}")
    print()

    print("=== Done ===")


if __name__ == "__main__":
    main()
