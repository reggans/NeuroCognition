#!/usr/bin/env python3
"""
SWM Multi-Turn GRPO Training Script
Train a language model on Spatial Working Memory using MT-GRPO.

Usage:
    python swm_mt_grpo_train.py --gpu_id 1
    CUDA_VISIBLE_DEVICES=1 python swm_mt_grpo_train.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import verifiers as vf
from shared.trainers import MTGRPOEnvTrainer
from shared.utils import get_default_grpo_config
from SWM.swm_verifiers_env import load_environment
from SWM.swm_rubric import SWMRubric


def main():
    parser = argparse.ArgumentParser(description="Train LLM on SWM with MT-GRPO")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU device ID (default: None = use CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "hard"],
        help="SWM difficulty: easy (8 boxes, 1 token, 64 trials) or hard (12 boxes, 2 tokens, 144 trials) (default: easy)",
    )
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default="text",
        choices=["text", "image-only", "image+text"],
        help="Feedback modality: text, image-only, or image+text (default: text)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help="Number of episodes in dataset (default: 200)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of training steps (default: 500)",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of rollouts per prompt (default: 4)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device (default: 1)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--vllm_server_url",
        type=str,
        default=None,
        help="Optional external vLLM server URL. If not provided, TRL will automatically spawn a vLLM server.",
    )

    args = parser.parse_args()

    # Set GPU device if specified
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")
    else:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        print(f"Using GPU from CUDA_VISIBLE_DEVICES: {gpu_id}")

    # Configure SWM based on difficulty
    if args.difficulty == "easy":
        n_boxes = 8
        n_tokens = 1
        max_trials = 64
    else:  # hard
        n_boxes = 12
        n_tokens = 2
        max_trials = 144

    # Configure feedback based on feedback_mode
    if args.feedback_mode == "text":
        mode = "text"
        image_only = False
    elif args.feedback_mode == "image-only":
        mode = "image"
        image_only = True
    else:  # image+text
        mode = "image"
        image_only = False

    print("=" * 70)
    print("SWM Multi-Turn GRPO Training")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(
        f"Difficulty: {args.difficulty} (boxes={n_boxes}, tokens={n_tokens}, max_trials={max_trials})"
    )
    print(f"Feedback mode: {args.feedback_mode} (mode={mode}, image_only={image_only})")
    print(f"Episodes: {args.num_episodes}")
    print(f"Seed: {args.seed}")

    # Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    model, tokenizer = vf.get_model_and_tokenizer(args.model_name)
    print(f"✓ Model loaded: {args.model_name}")

    # Create environment
    print("\n[2/5] Creating SWM environment...")
    env = load_environment(
        n_boxes=n_boxes,
        n_tokens=n_tokens,
        mode=mode,
        image_only=image_only,
        max_trials=max_trials,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )
    print(f"✓ Environment created with {args.num_episodes} episodes")

    # Get dataset
    print("\n[3/5] Loading dataset...")
    train_dataset = env.get_dataset()
    print(f"✓ Dataset loaded: {len(train_dataset)} examples")

    # Instantiate rubric and get reward functions
    print("\n[4/5] Setting up reward functions...")
    rubric = SWMRubric()

    turn_reward_funcs = rubric.turn_reward_funcs
    outcome_reward_funcs = rubric.outcome_reward_funcs

    print(f"✓ Turn-level reward functions: {len(turn_reward_funcs)}")
    for func in turn_reward_funcs:
        print(f"    - {func.__name__}")

    print(f"✓ Outcome-level reward functions: {len(outcome_reward_funcs)}")
    for func in outcome_reward_funcs:
        print(f"    - {func.__name__}")

    # Setup training configuration
    print("\n[5/5] Configuring trainer...")
    run_name = (
        f"swm-{args.feedback_mode}-{args.difficulty}"
        f"-{args.model_name.split('/')[-1].lower()}"
    )

    # Use default GRPO config for single GPU
    training_args = get_default_grpo_config(
        run_name=run_name,
        num_gpus=1,
        reward_weights=None,  # Equal weighting for all rewards
        vllm_server_url=args.vllm_server_url,
    )

    # Override with user-specified parameters
    training_args.max_steps = args.max_steps
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    training_args.num_generations = args.num_generations
    training_args.use_vllm = True  # Enable vLLM for faster generation
    training_args.gradient_checkpointing = True  # Save memory

    print(f"✓ Training configuration:")
    print(f"    Run name: {run_name}")
    print(f"    Learning rate: {training_args.learning_rate}")
    print(f"    Num generations: {training_args.num_generations}")
    print(f"    Batch size: {training_args.per_device_train_batch_size}")
    print(f"    Grad accumulation: {training_args.gradient_accumulation_steps}")
    print(f"    Max steps: {training_args.max_steps}")
    print(f"    Beta (KL penalty): {training_args.beta}")
    print(f"    W&B enabled: {training_args.report_to}")

    # Create trainer
    print("\n[Training] Creating MT-GRPO trainer...")
    trainer = MTGRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        turn_reward_funcs=turn_reward_funcs,
        outcome_reward_funcs=outcome_reward_funcs,
        turn_advantage_coef=1.0,  # Default coefficient
        args=training_args,
        train_dataset=train_dataset,
    )

    print("✓ Trainer created successfully")
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    # Train!
    trainer.train()

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
