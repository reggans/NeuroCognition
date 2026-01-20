#!/usr/bin/env python3
"""
Multi-Task Multi-Setup MT-GRPO Training Script

Train a single VLM on all cognitive tasks with all setup variants:
- RAPM (3 setups): text-mc, text-gen, image-mc
- WCST (4 setups): text-nobg, text-bg, image-nobg, image-bg (variant=card-random)
- SWM (6 setups): easy/hard × text/image+text/image-only

Total: 13 setups across 3 tasks

Usage:
    # Full training on GPUs 0-3 with vLLM server on 4-5
    CUDA_VISIBLE_DEVICES=0,1,2,3 python multi_task_mt_grpo_train.py --vllm_server_url http://localhost:8000

    # Quick test
    python multi_task_mt_grpo_train.py --quick_test
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import torch

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datasets import Dataset, concatenate_datasets
from transformers import TrainerCallback

# Import shared modules
from shared import MTGRPOEnvTrainer, get_model_and_tokenizer
from shared import get_default_grpo_config

# Import task-specific modules
from WCST.wcst_mt_env import load_environment as load_wcst_env, WCSTMultiTurnEnv
from WCST.wcst_rubric import WCSTRubric
from SWM.swm_mt_env import load_environment as load_swm_env, SWMMultiTurnEnv
from SWM.swm_rubric import SWMRubric
from RAPM.rapm_mt_env import load_environment as load_rapm_env, RAPMMultiTurnEnv
from RAPM.rapm_rubric import RAPMRubric


# =============================================================================
# Setup Configurations
# =============================================================================


@dataclass
class SetupConfig:
    """Configuration for a single setup."""

    task: str  # "wcst", "swm", "rapm"
    setup_name: str  # Unique identifier like "wcst_text_nobg"
    params: Dict[str, Any]  # Parameters for load_environment


# Define all 13 setups
RAPM_SETUPS = [
    SetupConfig("rapm", "rapm_text_mc", {"mode": "text", "answer_mode": "mc"}),
    SetupConfig("rapm", "rapm_text_gen", {"mode": "text", "answer_mode": "gen"}),
    SetupConfig("rapm", "rapm_image_mc", {"mode": "image", "answer_mode": "mc"}),
]

WCST_SETUPS = [
    # variant=card-random, modality × bg_color
    SetupConfig(
        "wcst",
        "wcst_text_nobg",
        {"variant": "card-random", "bg_color": False, "image_mode": False},
    ),
    SetupConfig(
        "wcst",
        "wcst_text_bg",
        {"variant": "card-random", "bg_color": True, "image_mode": False},
    ),
    SetupConfig(
        "wcst",
        "wcst_image_nobg",
        {"variant": "card-random", "bg_color": False, "image_mode": True},
    ),
    SetupConfig(
        "wcst",
        "wcst_image_bg",
        {"variant": "card-random", "bg_color": True, "image_mode": True},
    ),
]

SWM_SETUPS = [
    # easy (8 boxes, 1 token) × feedback_mode
    SetupConfig(
        "swm", "swm_easy_text", {"n_boxes": 8, "n_tokens": 1, "feedback_mode": "text"}
    ),
    SetupConfig(
        "swm",
        "swm_easy_imgtext",
        {"n_boxes": 8, "n_tokens": 1, "feedback_mode": "image+text"},
    ),
    SetupConfig(
        "swm",
        "swm_easy_imgonly",
        {"n_boxes": 8, "n_tokens": 1, "feedback_mode": "image-only"},
    ),
    # hard (12 boxes, 2 tokens) × feedback_mode
    SetupConfig(
        "swm", "swm_hard_text", {"n_boxes": 12, "n_tokens": 2, "feedback_mode": "text"}
    ),
    SetupConfig(
        "swm",
        "swm_hard_imgtext",
        {"n_boxes": 12, "n_tokens": 2, "feedback_mode": "image+text"},
    ),
    SetupConfig(
        "swm",
        "swm_hard_imgonly",
        {"n_boxes": 12, "n_tokens": 2, "feedback_mode": "image-only"},
    ),
]

ALL_SETUPS = RAPM_SETUPS + WCST_SETUPS + SWM_SETUPS


# =============================================================================
# Multi-Task Multi-Setup Environment
# =============================================================================


class MultiTaskMultiSetupEnvironment:
    """
    Unified environment that combines all tasks and setups.
    Routes interactions to appropriate task-specific environment based on setup info.
    """

    def __init__(
        self,
        envs: Dict[str, Any],  # setup_name -> environment instance
        rubrics: Dict[str, Any],  # task/setup -> rubric instance
        enable_thinking: bool = False,
        direct_answer: bool = True,
        system_prompt: Optional[str] = None,
    ):
        self.envs = envs  # Map setup_name -> env
        self.rubrics = rubrics  # Map task/setup -> rubric
        self.enable_thinking = enable_thinking
        self.direct_answer = direct_answer
        self._custom_system_prompt = system_prompt
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create unified system prompt for all tasks."""
        if self._custom_system_prompt:
            return self._custom_system_prompt

        base = (
            "You are an AI assistant that excels at cognitive reasoning tasks. "
            "You will be presented with various cognitive tests including:\n"
            "- Wisconsin Card Sorting Test (WCST): Learn to sort cards by changing rules\n"
            "- Spatial Working Memory (SWM): Remember and search for tokens in boxes\n"
            "- Raven's Progressive Matrices (RAPM): Identify patterns in sequences\n\n"
        )

        if self.direct_answer or not self.enable_thinking:
            return (
                base
                + "Provide a direct answer only. Do NOT include reasoning.\n"
                + "Respond in this format:\n"
                + "<answer>Your final answer</answer>"
            )

        return (
            base
            + "For each task, carefully analyze the problem, reason through your approach, "
            + "and provide your answer in the requested format:\n"
            + "<reasoning>Your step-by-step thinking</reasoning>\n"
            + "<answer>Your final answer</answer>"
        )

    def get_env_for_setup(self, setup_name: str):
        """Get environment for a specific setup."""
        return self.envs.get(setup_name)

    def get_rubric_for_task(self, task: str):
        """Get rubric for a specific task."""
        return self.rubrics.get(task)

    def is_completed(
        self, messages: List[Dict[str, str]], setup_name: str = None, **kwargs
    ) -> bool:
        """Check if episode is completed for given setup."""
        if setup_name:
            env = self.envs.get(setup_name)
            if env:
                return env.is_completed(messages, **kwargs)
        # Fallback: check all envs
        for env in self.envs.values():
            if not env.is_completed(messages, **kwargs):
                return False
        return True

    def env_response(
        self, messages: List[Dict[str, str]], setup_name: str = None, **kwargs
    ) -> Dict[str, str]:
        """Get environment response for given setup."""
        if setup_name:
            env = self.envs.get(setup_name)
            if env:
                return env.env_response(messages, **kwargs)
        return {"role": "user", "content": "Episode completed."}

    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm,
        sampling_params,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate completions for prompts, routing each to appropriate environment.

        Since all prompts may be from different setups, we group them by setup
        and delegate to each setup's environment generate method.
        """
        # Get setup info from kwargs - this should be passed from the trainer
        setups_for_prompts = kwargs.get("setups", [None] * len(prompts))

        # Group prompts by setup
        from collections import defaultdict

        setup_groups = defaultdict(list)  # setup -> [(original_idx, prompt)]
        for idx, (prompt, setup_name) in enumerate(zip(prompts, setups_for_prompts)):
            if setup_name and setup_name in self.envs:
                setup_groups[setup_name].append((idx, prompt))
            else:
                # Fallback to first available env
                first_setup = next(iter(self.envs.keys()))
                setup_groups[first_setup].append((idx, prompt))

        # Process each group through its environment
        results = {
            "ids": [None] * len(prompts),
            "messages": [None] * len(prompts),
            "mask": [None] * len(prompts),
        }

        for setup_name, indexed_prompts in setup_groups.items():
            env = self.envs[setup_name]
            indices = [ip[0] for ip in indexed_prompts]
            group_prompts = [ip[1] for ip in indexed_prompts]

            # Generate using setup's env
            group_result = env.generate(
                prompts=group_prompts,
                llm=llm,
                sampling_params=sampling_params,
            )

            # Store results at correct indices
            for i, orig_idx in enumerate(indices):
                results["ids"][orig_idx] = group_result["ids"][i]
                results["messages"][orig_idx] = group_result["messages"][i]
                results["mask"][orig_idx] = group_result["mask"][i]

        return results


def create_setup_environments(
    setups: List[SetupConfig],
    episodes_per_setup: int,
    seed: int,
    rapm_image_data: Optional[str] = None,
    rapm_image_base_path: Optional[str] = None,
    max_episode_tokens: int = 32768,
    max_trials: Optional[int] = None,
    wcst_max_trials: Optional[int] = None,
    swm_max_trials: Optional[int] = None,
    enable_thinking: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create environments and rubrics for all setups.

    Args:
        setups: List of setup configurations
        episodes_per_setup: Number of episodes per setup
        seed: Random seed
        rapm_image_data: Path to RAPM image data
        rapm_image_base_path: Base path for RAPM images
        max_episode_tokens: Maximum tokens per episode
        max_trials: Global max_trials override (optional)
        wcst_max_trials: WCST-specific max_trials override
        swm_max_trials: SWM-specific max_trials override
        enable_thinking: Whether to include thinking instructions in prompts

    Returns:
        envs: Dict mapping setup_name -> environment instance
        rubrics: Dict mapping task/setup -> rubric instance
    """
    envs = {}
    rubrics = {}

    # Create task rubrics (one per task type or mode)
    rubrics["wcst"] = WCSTRubric()
    rubrics["swm"] = SWMRubric()
    # RAPM rubrics stored separately for each mode
    rubrics["rapm_text_mc"] = RAPMRubric(mode="text", answer_mode="mc")
    rubrics["rapm_text_gen"] = RAPMRubric(mode="text", answer_mode="gen")
    rubrics["rapm_image_mc"] = RAPMRubric(mode="image", answer_mode="mc")

    # Default paths
    rapm_text_path = str(project_root / "rapm_data" / "text_rapm_train.jsonl")
    rapm_image_path = rapm_image_data or str(
        project_root / "rapm_data" / "raven_train_500.json"
    )
    # Images are in HuggingFace cache by default
    image_base_path = rapm_image_base_path or os.path.expanduser(
        "~/.cache/huggingface/RAPM"
    )

    for i, setup in enumerate(setups):
        setup_seed = seed + i * 1000

        try:
            if setup.task == "wcst":
                wcst_trials = (
                    wcst_max_trials if wcst_max_trials is not None else max_trials
                )
                if wcst_trials is None:
                    wcst_trials = 96 if setup.params.get("bg_color", False) else 64
                wcst_kwargs = {
                    "num_episodes": episodes_per_setup,
                    "seed": setup_seed,
                    "max_episode_tokens": max_episode_tokens,
                    "enable_thinking": enable_thinking,
                    **setup.params,
                }
                wcst_kwargs["max_trials"] = wcst_trials
                # max_steps controls max assistant turns - set to max_trials (easy=64, hard=96)
                wcst_kwargs["max_steps"] = wcst_trials
                env = load_wcst_env(**wcst_kwargs)
            elif setup.task == "swm":
                swm_trials = (
                    swm_max_trials if swm_max_trials is not None else max_trials
                )
                if swm_trials is None:
                    swm_trials = int(setup.params.get("n_boxes", 8)) ** 2
                swm_kwargs = {
                    "num_episodes": episodes_per_setup,
                    "seed": setup_seed,
                    "max_episode_tokens": max_episode_tokens,
                    "enable_thinking": enable_thinking,
                    **setup.params,
                }
                swm_kwargs["max_trials"] = swm_trials
                # max_steps controls max assistant turns - set to max_trials since each trial is one turn
                swm_kwargs["max_steps"] = swm_trials
                env = load_swm_env(**swm_kwargs)
            elif setup.task == "rapm":
                mode = setup.params.get("mode", "text")
                answer_mode = setup.params.get("answer_mode", "mc")
                # RAPM max_steps = 8 (number of choices/attempts)
                rapm_max_steps = 8

                if mode == "image":
                    # Check if image data exists
                    if not os.path.exists(rapm_image_path):
                        print(
                            f"  ⚠ Skipping {setup.setup_name}: image data not found at {rapm_image_path}"
                        )
                        continue
                    env = load_rapm_env(
                        mode="image",
                        answer_mode="mc",
                        eval_data=rapm_image_path,
                        image_base_path=image_base_path,
                        num_episodes=episodes_per_setup,
                        max_episode_tokens=max_episode_tokens,
                        enable_thinking=enable_thinking,
                        max_steps=rapm_max_steps,
                    )
                else:
                    env = load_rapm_env(
                        mode="text",
                        answer_mode=answer_mode,
                        eval_data=rapm_text_path,
                        num_episodes=episodes_per_setup,
                        max_episode_tokens=max_episode_tokens,
                        enable_thinking=enable_thinking,
                        max_steps=rapm_max_steps,
                    )
            else:
                raise ValueError(f"Unknown task: {setup.task}")

            envs[setup.setup_name] = env
            print(f"  ✓ {setup.setup_name}: environment created")

        except Exception as e:
            print(f"  ⚠ Failed to create {setup.setup_name}: {e}")

    return envs, rubrics


def create_combined_dataset(
    envs: Dict[str, Any],
    setups: List[SetupConfig],
) -> Dataset:
    """
    Combine datasets from all setups with task and setup labels.

    Each example will have:
    - prompt: The conversation messages
    - info: Task-specific info dict
    - task: Task name (wcst, swm, rapm)
    - setup: Setup name (e.g., wcst_text_nobg)
    """
    all_datasets = []

    for setup in setups:
        env = envs.get(setup.setup_name)
        if not env:
            continue

        ds = env.get_dataset()

        # Add task and setup columns
        ds = ds.map(
            lambda x, s=setup: {
                **x,
                "task": s.task,
                "setup": s.setup_name,
            }
        )

        all_datasets.append(ds)
        print(f"  ✓ {setup.setup_name}: {len(ds)} examples")

    if not all_datasets:
        raise ValueError("No datasets created! Check setup configurations.")

    # Concatenate and shuffle
    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)

    return combined


# =============================================================================
# Multi-Task Reward Functions
# =============================================================================


def create_multi_task_reward_functions(
    rubrics: Dict[str, Any],
) -> Tuple[List, List]:
    """
    Create reward functions that dispatch to task/setup-specific rubrics.

    The reward functions use the 'task' and 'setup' columns from the dataset
    to route to the appropriate rubric.
    """

    def get_rubric_for_sample(task: str, setup: str):
        """Get the appropriate rubric based on task and setup."""
        if task == "wcst":
            return rubrics.get("wcst")
        elif task == "swm":
            return rubrics.get("swm")
        elif task == "rapm":
            # RAPM has mode-specific rubrics
            if "text_mc" in setup:
                return rubrics.get("rapm_text_mc")
            elif "text_gen" in setup:
                return rubrics.get("rapm_text_gen")
            else:
                return rubrics.get("rapm_image_mc")
        return None

    def _extract_correct_answers_from_info(info_list, completions):
        """Extract correct answers from info dict based on trial progress.

        For WCST: info contains 'trials' list with 'correct' field.
        We determine current trial by counting assistant messages in completions.
        """
        if not info_list:
            return None

        answers = []
        for idx, completion in enumerate(completions):
            info = info_list[idx] if idx < len(info_list) else {}
            if not info:
                answers.append(None)
                continue

            trials = info.get("trials", [])
            if not trials:
                answers.append(None)
                continue

            # Count assistant messages to determine trial index
            # In multi-turn, each assistant response corresponds to a trial
            assistant_count = sum(
                1 for msg in completion if msg.get("role") == "assistant"
            )
            trial_idx = max(0, assistant_count - 1)  # Current trial being answered

            if trial_idx < len(trials):
                answers.append(trials[trial_idx].get("correct"))
            else:
                # Past all trials, use last one
                answers.append(trials[-1].get("correct") if trials else None)

        return answers

    def _flatten_info_to_kwargs(info_list, kwargs):
        """Flatten info dict fields into kwargs for reward functions.

        Rubric functions expect fields like 'token_box', 'legal_boxes', 'n_boxes',
        'trials' etc. as direct kwargs, but they're nested inside the 'info' dict.

        This extracts common fields from info and adds them to kwargs.
        For per-example fields, we take the first example's value (assuming batch
        is homogeneous within a task).
        """
        if not info_list:
            return kwargs

        # Get first info dict as reference
        first_info = info_list[0] if info_list else {}

        # Fields to extract (SWM needs these)
        swm_fields = [
            "token_box",
            "legal_boxes",
            "n_boxes",
            "n_tokens",
            "tokens",
            "max_trials",
        ]
        for field in swm_fields:
            if field not in kwargs and field in first_info:
                kwargs[field] = first_info[field]

        # WCST fields
        wcst_fields = ["trials", "rules", "variant", "max_trials"]
        for field in wcst_fields:
            if field not in kwargs and field in first_info:
                kwargs[field] = first_info[field]

        # RAPM fields
        rapm_fields = [
            "correct_answer",
            "choices",
            "question",
            "options",
            "mode",
            "answer_mode",
            "constraints",
            "cell_constraint",
        ]
        for field in rapm_fields:
            if field not in kwargs and field in first_info:
                kwargs[field] = first_info[field]

        # RAPM uses 'correct_answer' but rubric expects 'answer' - map it
        if "answer" not in kwargs and "correct_answer" in first_info:
            kwargs["answer"] = first_info["correct_answer"]

        return kwargs

    def multi_task_turn_reward(completions, **kwargs):
        """Dispatch turn-level rewards based on task and setup."""
        # Extract task/setup from kwargs
        task_raw = kwargs.get("task", "wcst")
        setup_raw = kwargs.get("setup", "")
        info_list = kwargs.get("info", [])

        task = task_raw[0] if isinstance(task_raw, list) else task_raw
        setup = setup_raw[0] if isinstance(setup_raw, list) else setup_raw

        # Flatten info fields into kwargs FIRST (RAPM's correct_answer maps to answer)
        kwargs = _flatten_info_to_kwargs(info_list, kwargs)

        # Get answer AFTER flattening
        answer = kwargs.get("answer", None)

        # If no answer provided, try to extract from info (for WCST with trials)
        if answer is None and info_list:
            answer = _extract_correct_answers_from_info(info_list, completions)

        # Ensure answer is a list matching completions length
        if answer is None:
            answer = [None] * len(completions)
        elif not isinstance(answer, list):
            answer = [answer] * len(completions)

        rubric = get_rubric_for_sample(task, setup)
        if not rubric:
            return [0.0] * len(completions)

        # Remove 'answer' from kwargs since we pass it positionally
        call_kwargs = {k: v for k, v in kwargs.items() if k != "answer"}

        # Aggregate rewards from all turn functions
        all_rewards = []
        turn_funcs = [f for f in (rubric.turn_reward_funcs or []) if f is not None]
        for func in turn_funcs:
            try:
                func_rewards = func(completions, answer, **call_kwargs)
                if func_rewards is None:
                    raise ValueError("Reward function returned None")
                all_rewards.append(func_rewards)
            except Exception as e:
                print(f"Warning: Turn reward function failed: {e}")
                all_rewards.append([0.0] * len(completions))

        # Sum across all turn functions
        num_examples = len(completions)
        rewards = []
        for i in range(num_examples):
            total = sum(r[i] for r in all_rewards if i < len(r))
            rewards.append(total)

        return rewards

    def multi_task_outcome_reward(completions, **kwargs):
        """Dispatch outcome-level rewards based on task and setup."""
        # Extract task/setup from kwargs
        task_raw = kwargs.get("task", "wcst")
        setup_raw = kwargs.get("setup", "")
        info_list = kwargs.get("info", [])

        task = task_raw[0] if isinstance(task_raw, list) else task_raw
        setup = setup_raw[0] if isinstance(setup_raw, list) else setup_raw

        # Flatten info fields into kwargs FIRST (RAPM's correct_answer maps to answer)
        kwargs = _flatten_info_to_kwargs(info_list, kwargs)

        # Get answer AFTER flattening
        answer = kwargs.get("answer", None)

        # If no answer provided, try to extract from info (for WCST with trials)
        if answer is None and info_list:
            answer = _extract_correct_answers_from_info(info_list, completions)

        # Ensure answer is a list matching completions length
        if answer is None:
            answer = [None] * len(completions)
        elif not isinstance(answer, list):
            answer = [answer] * len(completions)

        rubric = get_rubric_for_sample(task, setup)
        if not rubric:
            return [0.0] * len(completions)

        # Remove 'answer' from kwargs since we pass it positionally
        call_kwargs = {k: v for k, v in kwargs.items() if k != "answer"}

        # Aggregate rewards from all outcome functions
        all_rewards = []
        outcome_funcs = [
            f for f in (rubric.outcome_reward_funcs or []) if f is not None
        ]
        for func in outcome_funcs:
            try:
                func_rewards = func(completions, answer, **call_kwargs)
                if func_rewards is None:
                    raise ValueError("Reward function returned None")
                all_rewards.append(func_rewards)
            except Exception as e:
                print(f"Warning: Outcome reward function failed: {e}")
                all_rewards.append([0.0] * len(completions))

        # Sum across all outcome functions
        num_examples = len(completions)
        rewards = []
        for i in range(num_examples):
            total = sum(r[i] for r in all_rewards if i < len(r))
            rewards.append(total)

        return rewards

    return [multi_task_turn_reward], [multi_task_outcome_reward]


# =============================================================================
# Main Training Script
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train VLM on all cognitive tasks with all setup variants"
    )

    # Model configuration
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

    # Training configuration
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of training steps (default: 1000)",
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

    # Data configuration
    parser.add_argument(
        "--episodes_per_setup",
        type=int,
        default=50,
        help="Number of episodes per setup (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--rapm_image_data",
        type=str,
        default=None,
        help="Path to RAPM image data JSON (default: rapm_data/raven_train_500.json)",
    )
    parser.add_argument(
        "--rapm_image_base_path",
        type=str,
        default=None,
        help="Base path for RAPM images (default: ~/.cache/huggingface/RAPM)",
    )

    # vLLM configuration
    parser.add_argument(
        "--vllm_server_url",
        type=str,
        default=None,
        help="vLLM server URL for server mode (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=str(project_root / "shared" / "deepspeed_config.json"),
        help="Path to DeepSpeed config JSON (default: shared/deepspeed_config.json)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for training (default: 1). Used for gradient accumulation calculation.",
    )

    # Setup selection
    parser.add_argument(
        "--setups",
        type=str,
        nargs="+",
        default=None,
        help="Specific setups to include (default: all). Options: "
        + ", ".join(s.setup_name for s in ALL_SETUPS),
    )
    parser.add_argument(
        "--exclude_image",
        action="store_true",
        help="Exclude image-based setups (for text-only training)",
    )
    parser.add_argument(
        "--text_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use text-only setups across all tasks (default: true)",
    )

    # Testing
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick test mode: minimal data, no actual training",
    )

    # Episode configuration
    parser.add_argument(
        "--max_episode_tokens",
        type=int,
        default=4096,
        help="Maximum tokens per episode (default: 4096). Reduce for faster testing.",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=None,
        help="Maximum trials per WCST/SWM episode (default: None = task defaults).",
    )
    parser.add_argument(
        "--wcst_max_trials",
        type=int,
        default=None,
        help="Override max trials for WCST only (default: None = use --max_trials)",
    )
    parser.add_argument(
        "--swm_max_trials",
        type=int,
        default=None,
        help="Override max trials for SWM only (default: None = use --max_trials)",
    )

    # Generation configuration
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=128,
        help="Maximum tokens per turn (default: 128). Controls per-turn generation limit.",
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable reasoning mode in prompts (default: false)",
    )
    parser.add_argument(
        "--direct_answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require direct answer only (default: true)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Override the default system prompt",
    )

    # LoRA / PEFT configuration
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA (Low-Rank Adaptation) for memory-efficient training",
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantized LoRA) for maximum memory efficiency. Implies --use_lora.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16). Higher = more capacity but more memory.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32). Typical: 2x rank.",
    )

    args = parser.parse_args()

    # Set GPU device if specified
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")
    else:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        print(f"Using GPU from CUDA_VISIBLE_DEVICES: {gpu_id}")

    # Select setups
    if args.setups and args.setups != ["all"]:
        setups = [s for s in ALL_SETUPS if s.setup_name in args.setups]
    else:
        setups = ALL_SETUPS.copy()

    if args.text_only:
        # Enforce text-only setups across all tasks
        setups = [
            s
            for s in setups
            if not any(kw in s.setup_name.lower() for kw in ["image", "img"])
        ]

    if args.exclude_image:
        # Exclude any setup with image-related keywords
        setups = [
            s
            for s in setups
            if not any(kw in s.setup_name.lower() for kw in ["image", "img"])
        ]

    # Quick test mode
    if args.quick_test:
        args.episodes_per_setup = 2
        args.max_steps = 2
        setups = setups[:3]  # Only first 3 setups

    print("=" * 70)
    print("Multi-Task Multi-Setup MT-GRPO Training")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Total setups: {len(setups)}")
    print(f"Episodes per setup: {args.episodes_per_setup}")
    print(f"Seed: {args.seed}")
    print()
    print("Setups:")
    for setup in setups:
        print(f"  • {setup.setup_name} ({setup.task}): {setup.params}")

    # Create environments and rubrics
    print("\n[1/5] Creating environments for all setups...")
    envs, rubrics = create_setup_environments(
        setups=setups,
        episodes_per_setup=args.episodes_per_setup,
        seed=args.seed,
        rapm_image_data=args.rapm_image_data,
        rapm_image_base_path=args.rapm_image_base_path,
        max_episode_tokens=args.max_episode_tokens,
        max_trials=args.max_trials,
        wcst_max_trials=args.wcst_max_trials,
        swm_max_trials=args.swm_max_trials,
        enable_thinking=args.enable_thinking,
    )
    print(f"✓ Created {len(envs)} environments and {len(rubrics)} rubrics")

    # Create combined dataset
    print("\n[2/5] Creating combined dataset...")
    train_dataset = create_combined_dataset(envs, setups)
    total_examples = len(train_dataset)
    print(f"✓ Combined dataset: {total_examples} examples")

    # Show distribution
    task_counts = {}
    for ex in train_dataset:
        task = ex.get("task", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    print("Distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} examples ({100*count/total_examples:.1f}%)")

    # Create multi-task reward functions
    print("\n[3/5] Setting up multi-task reward functions...")
    turn_reward_funcs, outcome_reward_funcs = create_multi_task_reward_functions(
        rubrics
    )
    print(f"✓ Turn reward functions: {len(turn_reward_funcs)}")
    print(f"✓ Outcome reward functions: {len(outcome_reward_funcs)}")

    # Create multi-task environment wrapper
    print("\n[4/5] Creating multi-task environment wrapper...")
    multi_env = MultiTaskMultiSetupEnvironment(
        envs=envs,
        rubrics=rubrics,
        enable_thinking=args.enable_thinking,
        direct_answer=args.direct_answer,
        system_prompt=args.system_prompt,
    )
    print("✓ Multi-task multi-setup environment ready")

    if args.quick_test:
        print("\n" + "=" * 70)
        print("QUICK TEST MODE - Skipping actual training")
        print("=" * 70)
        print("\n✓ All components created successfully!")
        print(f"  Environments: {len(envs)}")
        print(f"  Rubrics: {len(rubrics)}")
        print(f"  Dataset size: {total_examples}")
        print("\nRun without --quick_test for actual training.")
        return

    # Load model and tokenizer
    print("\n[5/5] Loading model and tokenizer...")

    # QLoRA: load model in 4-bit quantization
    model_kwargs = None
    if args.use_qlora:
        args.use_lora = True  # QLoRA implies LoRA
        try:
            from transformers import BitsAndBytesConfig

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "quantization_config": quantization_config,
                "attn_implementation": "sdpa",  # flash_attn has issues with quantization
                "device_map": {"": local_rank},
            }
            print("✓ QLoRA mode: Loading model with 4-bit quantization")
        except ImportError:
            print(
                "⚠ bitsandbytes not installed. Install with: pip install bitsandbytes"
            )
            print("  Falling back to standard LoRA.")
            args.use_qlora = False

    model, tokenizer = get_model_and_tokenizer(args.model_name, model_kwargs)

    # QLoRA prep: ensure inputs require grads + cast layer norms
    if args.use_qlora:
        try:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
            print("✓ QLoRA prep: model set for k-bit training")
        except ImportError:
            print("⚠ PEFT not installed. Install with: pip install peft")

    print(f"✓ Model loaded: {args.model_name}")

    # Setup training configuration
    print("\n[Training Setup] Configuring trainer...")
    run_name = (
        f"multi-task-{len(setups)}setups-{args.model_name.split('/')[-1].lower()}"
    )

    training_args = get_default_grpo_config(
        run_name=run_name,
        num_gpus=args.num_gpus,
        reward_weights=None,
        vllm_server_url=args.vllm_server_url,
    )

    # Override with user-specified parameters
    training_args.max_steps = args.max_steps
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    training_args.num_generations = args.num_generations
    training_args.max_completion_length = args.max_completion_length
    training_args.deepspeed = args.deepspeed_config

    # Gradient checkpointing config
    # With LoRA/QLoRA, use non-reentrant checkpointing to avoid "no requires_grad" warning
    if args.use_lora:
        training_args.gradient_checkpointing = True
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    else:
        training_args.gradient_checkpointing = True

    print(f"✓ Training configuration:")
    print(f"    Run name: {run_name}")
    print(f"    Learning rate: {training_args.learning_rate}")
    print(f"    Num generations: {training_args.num_generations}")
    print(f"    Batch size: {training_args.per_device_train_batch_size}")
    print(f"    Grad accumulation: {training_args.gradient_accumulation_steps}")
    print(f"    Max steps: {training_args.max_steps}")
    print(
        f"    Max completion length (per-turn): {training_args.max_completion_length}"
    )
    print(f"    vLLM mode: {training_args.vllm_mode}")

    # vLLM server preflight: close any stale weight-update communicator (rank 0 only)
    if getattr(training_args, "vllm_mode", None) == "server" and args.vllm_server_url:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            try:
                import requests

                close_url = args.vllm_server_url.rstrip("/") + "/close_communicator/"
                resp = requests.post(close_url, timeout=5)
                if resp.status_code != 200:
                    print(
                        f"⚠ vLLM close_communicator returned {resp.status_code}: {resp.text}"
                    )
                else:
                    print("✓ vLLM communicator reset")
            except Exception as e:
                print(f"⚠ vLLM communicator reset skipped: {e}")

    # Setup LoRA if requested
    peft_config = None
    if args.use_lora:
        try:
            from peft import LoraConfig

            peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",  # Attention
                    "gate_proj",
                    "up_proj",
                    "down_proj",  # MLP
                ],
                task_type="CAUSAL_LM",
                bias="none",
            )
            mode_str = "QLoRA (4-bit)" if args.use_qlora else "LoRA"
            print(
                f"✓ {mode_str} enabled (rank={args.lora_rank}, alpha={args.lora_alpha})"
            )
        except ImportError:
            print("⚠ PEFT not installed. Install with: pip install peft")
            print("  Falling back to full fine-tuning.")
    else:
        print("  LoRA: disabled (use --use_lora to enable)")

    # Create trainer
    print("\n[Training] Creating MT-GRPO trainer...")
    trainer = MTGRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        env=multi_env,
        turn_reward_funcs=turn_reward_funcs,
        outcome_reward_funcs=outcome_reward_funcs,
        turn_advantage_coef=1.0,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        enable_thinking=args.enable_thinking,
    )

    print("✓ Multi-task trainer created successfully")
    print("\n" + "=" * 70)
    print("Starting multi-task multi-setup training...")
    print("=" * 70)
    print("\nThe model will learn from all cognitive tasks and setups:")
    print("  🧩 WCST - 4 setups (text/image × bg/no-bg)")
    print("  🎯 SWM - 6 setups (easy/hard × 3 feedback modes)")
    print("  🔮 RAPM - 3 setups (text-mc/text-gen/image-mc)")
    print("=" * 70)

    # Train!
    trainer.train()

    print("\n" + "=" * 70)
    print("Multi-task multi-setup training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
