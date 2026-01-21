#!/usr/bin/env python3
"""Convert HuggingFace Qwen model to TensorRT-LLM checkpoint format.

Compatible with TensorRT-LLM 0.16.0+
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert HF model to TRT-LLM checkpoint")
    parser.add_argument("--model-name", required=True, help="HuggingFace model name")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoint")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="Data type")
    args = parser.parse_args()

    print(f"Converting model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data type: {args.dtype}")
    print()

    try:
        from tensorrt_llm.models import LLaMAForCausalLM
        from transformers import AutoConfig
    except ImportError as e:
        print(f"ERROR: Required module not found: {e}")
        print("Run this inside the TRT-LLM container.")
        sys.exit(1)

    # Detect model architecture
    print("Detecting model architecture...")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    arch = config.architectures[0] if config.architectures else "unknown"
    print(f"Architecture: {arch}")

    # TensorRT-LLM 0.16+ uses a unified conversion approach
    # Qwen2 models use the LLaMA-style converter
    try:
        from tensorrt_llm.models.convert_utils import convert_hf_llama
        from tensorrt_llm.models.modeling_utils import PretrainedConfig

        print("Downloading and converting checkpoint...")

        # For Qwen2, we use the convert_and_save_hf method
        from tensorrt_llm.models.llama.convert import convert_hf_config, convert_hf_weights
        from tensorrt_llm import Mapping
        import torch

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert using the high-level API
        dtype = args.dtype

        # Use AutoModelForCausalLM to get the model
        from tensorrt_llm.models import LLaMAForCausalLM

        LLaMAForCausalLM.convert_hf_checkpoint(
            args.model_name,
            dtype=dtype,
            output_dir=args.output_dir,
            load_model_on_cpu=True
        )

        print("Checkpoint conversion complete.")

    except Exception as e:
        print(f"High-level conversion failed: {e}")
        print("Trying alternative conversion method...")

        # Alternative: Use the examples converter script approach
        try:
            import subprocess
            import os

            # Find the TRT-LLM examples directory
            trtllm_examples = "/opt/tritonserver/backends/tensorrtllm/examples"
            if not os.path.exists(trtllm_examples):
                trtllm_examples = "/app/tensorrt_llm/examples"

            qwen_convert = f"{trtllm_examples}/qwen/convert_checkpoint.py"

            if os.path.exists(qwen_convert):
                print(f"Using Qwen converter: {qwen_convert}")
                subprocess.run([
                    sys.executable, qwen_convert,
                    "--model_dir", args.model_name,
                    "--output_dir", args.output_dir,
                    "--dtype", args.dtype,
                    "--tp_size", "1"
                ], check=True)
                print("Checkpoint conversion complete.")
            else:
                raise FileNotFoundError(f"Converter not found at {qwen_convert}")

        except Exception as e2:
            print(f"ERROR: All conversion methods failed.")
            print(f"Error 1: {e}")
            print(f"Error 2: {e2}")
            sys.exit(1)


if __name__ == "__main__":
    main()