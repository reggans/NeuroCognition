#!/usr/bin/env python3
"""
Test script for verifiers environments: RAPM, SWM, WCST.
Uses Qwen/Qwen3-0.6B model on local GPU.

Usage:
    python test_verifiers_env.py --task rapm --num_examples 2
    python test_verifiers_env.py --task swm --num_examples 2
    python test_verifiers_env.py --task wcst --num_examples 2
    python test_verifiers_env.py --task all --num_examples 2
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import task modules
try:
    from RAPM import rapm_verifiers_env
    from SWM import swm_verifiers_env
    from WCST import wcst_verifiers_env
except ImportError as e:
    print(f"Warning: Could not import task modules: {e}")
    print("Continuing with available modules...")

try:
    import verifiers as vf
except ImportError:
    print("Error: verifiers library not installed. Install with:")
    print("pip install git+https://github.com/PrimeIntellect-ai/verifiers.git")
    sys.exit(1)


class QwenInferenceModel:
    """Wrapper for Qwen model inference on GPU."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"Loading model: {model_name}")
        print(f"Device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Load model with manual device placement
        print("Loading model to CPU first...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        if device == "cuda":
            print("Moving model to GPU...")
            self.model = self.model.to(device).half()  # Use half precision on GPU

        self.model.eval()
        print(f"Model loaded successfully")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response for given prompt."""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return response.strip()


async def test_rapm_env(model: QwenInferenceModel, num_examples: int = 2):
    """Test RAPM verifiers environment."""
    print("\n" + "=" * 60)
    print("Testing RAPM Verifiers Environment")
    print("=" * 60)

    # Use sample data if available
    rapm_data_dir = project_root / "RAPM"
    text_dataset_path = rapm_data_dir / "text_rapm.jsonl"

    if not text_dataset_path.exists():
        print(f"RAPM dataset not found at {text_dataset_path}")
        print("Creating minimal test dataset...")
        # Create minimal test dataset
        os.makedirs(rapm_data_dir, exist_ok=True)
        test_data = [
            {
                "id": "test_0",
                "question_grid": [
                    ["123", "234", "345"],
                    ["456", "567", "678"],
                    ["789", "890", "?"],
                ],
                "options": ["901", "012", "111", "222"],
                "correct_index": 0,
                "cell_constraints": {
                    "2,2": {
                        "fixed_length": 3,
                        "target_counts": {"9": 1, "0": 1, "1": 1},
                    }
                },
            }
        ]
        with open(text_dataset_path, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

    try:
        env = rapm_verifiers_env.load_environment(
            mode="text",
            answer_mode="mc",
            text_dataset_path=str(text_dataset_path),
            max_examples=num_examples,
        )
        print(f"✓ RAPM environment loaded successfully")
        print(f"  Mode: text-mc")
        print(f"  Dataset examples: {num_examples}")

        # Get first example from dataset
        if hasattr(env, "dataset") and len(env.dataset) > 0:
            example = env.dataset[0]
            print(f"\n  Sample prompt:\n  {example['prompt'][:100]}...")

            # Generate response
            print(f"\n  Generating response...")
            response = model.generate(
                prompt=example["prompt"],
                system_prompt=(
                    env.system_prompt if hasattr(env, "system_prompt") else None
                ),
            )
            print(f"  Model response: {response[:100]}...")

    except Exception as e:
        print(f"✗ Error loading RAPM environment: {e}")
        import traceback

        traceback.print_exc()


async def test_swm_env(model: QwenInferenceModel, num_examples: int = 2):
    """Test SWM verifiers environment."""
    print("\n" + "=" * 60)
    print("Testing SWM Verifiers Environment")
    print("=" * 60)

    try:
        env = swm_verifiers_env.load_environment(
            n_boxes=4,
            n_tokens=1,
            mode="text",
            num_episodes=num_examples,
            seed=42,
        )
        print(f"✓ SWM environment loaded successfully")
        print(f"  Mode: text")
        print(f"  Boxes: 4, Tokens: 1")
        print(f"  Episodes: {num_examples}")

        # Get first example from dataset
        if hasattr(env, "dataset") and len(env.dataset) > 0:
            example = env.dataset[0]
            print(f"\n  Sample prompt: {example['prompt']}")

            # Generate response
            print(f"\n  Generating response...")
            response = model.generate(
                prompt=example["prompt"],
                system_prompt=(
                    env.system_prompt if hasattr(env, "system_prompt") else None
                ),
            )
            print(f"  Model response: {response[:100]}...")

    except Exception as e:
        print(f"✗ Error loading SWM environment: {e}")
        import traceback

        traceback.print_exc()


async def test_wcst_env(model: QwenInferenceModel, num_examples: int = 2):
    """Test WCST verifiers environment."""
    print("\n" + "=" * 60)
    print("Testing WCST Verifiers Environment")
    print("=" * 60)

    try:
        env = wcst_verifiers_env.load_environment(
            variant="string",
            max_trials=32,
            num_episodes=num_examples,
            seed=42,
        )
        print(f"✓ WCST environment loaded successfully")
        print(f"  Variant: string")
        print(f"  Max trials: 32")
        print(f"  Episodes: {num_examples}")

        # Get first example from dataset
        if hasattr(env, "dataset") and len(env.dataset) > 0:
            example = env.dataset[0]
            print(f"\n  Sample prompt: {example['prompt']}")

            # Generate response
            print(f"\n  Generating response...")
            response = model.generate(
                prompt=example["prompt"],
                system_prompt=(
                    env.system_prompt if hasattr(env, "system_prompt") else None
                ),
            )
            print(f"  Model response: {response[:100]}...")

    except Exception as e:
        print(f"✗ Error loading WCST environment: {e}")
        import traceback

        traceback.print_exc()


async def main():
    parser = argparse.ArgumentParser(description="Test verifiers environments")
    parser.add_argument(
        "--task",
        type=str,
        choices=["rapm", "swm", "wcst", "all"],
        default="all",
        help="Which task to test",
    )
    parser.add_argument(
        "--num-examples", type=int, default=2, help="Number of examples to test"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model name to use for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Verifiers Environments Test Suite")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Examples per task: {args.num_examples}")

    # Check GPU availability
    if args.device == "cuda":
        if torch.cuda.is_available():
            print(f"\nGPU Information:")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Available: True")
            print(
                f"  Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            print("\nWarning: CUDA requested but not available. Falling back to CPU.")
            args.device = "cpu"

    # Load model
    try:
        model = QwenInferenceModel(
            model_name=args.model,
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: Model download may take time on first run.")
        return

    # Run tests
    tasks = []
    if args.task in ["rapm", "all"]:
        tasks.append(test_rapm_env(model, args.num_examples))
    if args.task in ["swm", "all"]:
        tasks.append(test_swm_env(model, args.num_examples))
    if args.task in ["wcst", "all"]:
        tasks.append(test_wcst_env(model, args.num_examples))

    await asyncio.gather(*tasks)

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Verify all environments loaded successfully")
    print("2. Check model responses in the output above")
    print("3. Run actual RL training: python multi_turn_qa_experiment.py")


if __name__ == "__main__":
    asyncio.run(main())
