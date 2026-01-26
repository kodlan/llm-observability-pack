#!/usr/bin/env python3
"""Load test for Triton with TensorRT-LLM backend.

Uses client-side tokenization since TRT-LLM backend expects pre-tokenized input.
"""

import argparse
import random
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOKENIZER_DIR = PROJECT_ROOT / "serving/triton-trt/.cache/models/Qwen_Qwen2.5-1.5B-Instruct"

PROMPTS = [
    "Explain what Python is in 2 sentences.",
    "What is machine learning? Brief answer.",
    "Count from 1 to 10.",
    "Say hello in 3 different languages.",
    "What is 2 + 2? Just the number.",
]

stop_event = threading.Event()


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


def worker(worker_id: int, triton_url: str, tokenizer, prompts_tokenized: list):
    """Worker thread that sends continuous requests."""
    url = f"{triton_url}/v2/models/qwen/infer"
    headers = {"Content-Type": "application/json"}

    while not stop_event.is_set():
        # Pick a random prompt
        input_ids = random.choice(prompts_tokenized)

        start_time = time.time()

        try:
            request_data = create_inference_request(input_ids, max_tokens=50)
            response = requests.post(url, headers=headers, json=request_data, timeout=30)
            duration = time.time() - start_time

            if response.status_code == 200:
                print(f"[Worker {worker_id}] OK - {duration:.2f}s")
            else:
                print(f"[Worker {worker_id}] ERROR {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"[Worker {worker_id}] TIMEOUT")
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR: {e}")

        time.sleep(0.1)


def signal_handler(signum, frame):
    """Handle Ctrl+C."""
    print("\nStopping...")
    stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="Load test Triton TensorRT-LLM backend")
    parser.add_argument("--url", default="http://localhost:8000", help="Triton server URL")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of concurrent workers")
    args = parser.parse_args()

    print(f"Starting Triton TensorRT-LLM load test with {args.concurrency} concurrent requests")
    print(f"Target: {args.url}")
    print("Press Ctrl+C to stop")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    # Pre-tokenize all prompts
    print("Tokenizing prompts...")
    prompts_tokenized = [tokenizer.encode(p) for p in PROMPTS]
    print(f"Ready with {len(prompts_tokenized)} prompts")
    print()

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Start worker threads
    threads = []
    for i in range(args.concurrency):
        t = threading.Thread(target=worker, args=(i + 1, args.url, tokenizer, prompts_tokenized))
        t.daemon = True
        t.start()
        threads.append(t)

    # Wait for stop signal
    while not stop_event.is_set():
        time.sleep(0.1)

    # Wait for threads to finish
    for t in threads:
        t.join(timeout=1)


if __name__ == "__main__":
    main()
