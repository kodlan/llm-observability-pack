#!/usr/bin/env python3
"""Convert HuggingFace Qwen model to TensorRT-LLM checkpoint format."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert HF model to TRT-LLM checkpoint")
    parser.add_argument("--model-name", required=True, help="HuggingFace model name")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoint")
    parser.add_argument("--dtype", default="float16", help="Data type (float16, bfloat16)")
    args = parser.parse_args()

    try:
        from tensorrt_llm.models import QWenForCausalLM
    except ImportError:
        print("ERROR: tensorrt_llm not installed. Run this inside the TRT-LLM container.")
        sys.exit(1)

    print(f"Converting model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data type: {args.dtype}")
    print()

    print("Downloading and converting checkpoint...")
    QWenForCausalLM.convert_hf_checkpoint(
        model_dir=args.model_name,
        output_dir=args.output_dir,
        dtype=args.dtype
    )
    print("Checkpoint conversion complete.")


if __name__ == "__main__":
    main()
