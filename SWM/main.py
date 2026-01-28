from huggingface_hub import login
import transformers
import torch
import numpy as np
from tqdm.auto import tqdm

import random, json, re, argparse, os
import string
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..shared.model_wrapper import ModelWrapper
except ImportError:
    from shared.model_wrapper import ModelWrapper
from .swm import image_swm, text_swm, score


def swm_main(
    model=None,
    model_source="hf",
    n_boxes=6,
    n_tokens=1,
    image_only=False,
    cot=False,
    runs=1,
    max_tokens=512,
    think_budget=64,
    notes=False,
    image=False,
    api_key=None,
):
    """
    Run the Spatial Working Memory (SWM) test.

    Args:
        model: The model to use
        model_source: The source of the model ("hf", "google", "litellm", "vllm")
        n_boxes: The number of boxes in the test
        n_tokens: The number of different tokens present at the same time
        cot: Whether to use chain-of-thought reasoning
        runs: The number of runs to perform
        max_tokens: Maximum number of tokens to generate
        think_budget: Budget tokens for reasoning
        notes: Whether to use note-taking assistance
        image: Whether to use image mode
        api_key: API key to use
    """
    # Input validation
    if model_source not in ["openai", "openrouter", "vllm", "google"]:
        raise ValueError(
            "Model source must be either 'openai', 'openrouter', 'vllm', or 'google'."
        )
    if not image and image_only:
        raise ValueError("Image-only mode requires image mode to be enabled.")

    if model is None:
        if model_source == "vllm":
            model = "Qwen/Qwen3-32B"
        elif model_source == "openai":
            model = "o4-mini-2025-04-16"
        elif model_source == "openrouter":
            model = "qwen/qwen3-235b-a22b-07-25"
        elif model_source == "google":
            # Default to Gemini 3 Pro for Google model_source; users can override via --model
            model = "gemini-3-pro-preview"

    if image:
        if image_only:
            if notes:
                os.makedirs("SWM/data/image/baselines", exist_ok=True)
            else:
                os.makedirs("SWM/data/image", exist_ok=True)
        else:
            if notes:
                os.makedirs("SWM/data/image-text/baselines", exist_ok=True)
            else:
                os.makedirs("SWM/data/image-text", exist_ok=True)

        img_path = "SWM/images"
        os.makedirs(img_path, exist_ok=True)
    else:
        if notes:
            os.makedirs("SWM/data/text/baselines", exist_ok=True)
        else:
            os.makedirs("SWM/data/text", exist_ok=True)
        img_path = None

    if image:
        if image_only:
            data_subdir = "image"
        else:
            data_subdir = "image-text"
    else:
        data_subdir = "text"

    file_header = f"SWM/data/{data_subdir}/{'baselines/' if notes else ''}{model_source}_{model.replace('/', '-')}{'_cot' if cot else ''}_{n_boxes}_{n_tokens}_"
    print(f"Saving to: {file_header}")

    # Check if results already exist
    stats_file = file_header + "run_stats.json"
    history_file = file_header + "run_history.json"
    structured_history_file = file_header + "run_structured_history.json"
    reasoning_file = file_header + "run_reasoning.json"

    run_stats = {}
    run_history = {}
    structured_history = {}
    run_reasoning = {}

    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r") as f:
                run_stats = json.load(f)
        except Exception as e:
            print(f"Error loading stats file: {e}")

    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                run_history = json.load(f)
        except Exception:
            pass

    if os.path.exists(structured_history_file):
        try:
            with open(structured_history_file, "r") as f:
                structured_history = json.load(f)
        except Exception:
            pass

    if os.path.exists(reasoning_file):
        try:
            with open(reasoning_file, "r") as f:
                run_reasoning = json.load(f)
        except Exception:
            pass

    existing_runs = len(run_stats)

    if existing_runs >= runs:
        print(
            f"Results already exist at {stats_file} ({existing_runs} runs completed, {runs} requested)"
        )

        # Calculate and display statistics
        avg_stats = {}
        # Use the first available run key instead of hardcoded "run_1"
        first_key = next(iter(run_stats)) if run_stats else None

        if first_key:
            for key in run_stats[first_key].keys():
                if type(run_stats[first_key][key]) in [int, float]:
                    avg_stats[key] = np.mean(
                        [stats[key] for stats in run_stats.values()]
                    )

            for key, value in avg_stats.items():
                print(f"{key}: {value}")

            tot_score = 0
            for stats in run_stats.values():
                tot_score += score(stats)
            print(f"Score: {tot_score / len(run_stats.keys())}")
        return

    print(
        f"Found {existing_runs} runs. Continuing from run {existing_runs + 1} to {runs}."
    )

    for i in range(existing_runs, runs):
        model_instance = None
        torch.cuda.empty_cache()

        model_instance = ModelWrapper(
            model,
            model_source,
            api_key=api_key,
            max_new_tokens=max_tokens,
            image_input=image,
            image_path=img_path,
        )

        print(f"Run {i+1}/{runs}")
        try:
            if image:
                run_stats[f"run_{i+1}"], structured_history[f"run_{i+1}"] = image_swm(
                    model_instance,
                    n_boxes,
                    n_tokens=n_tokens,
                    cot=cot,
                    think_budget=think_budget,
                    note_assist=notes,
                    image_only=image_only,
                )
            else:
                run_stats[f"run_{i+1}"], structured_history[f"run_{i+1}"] = text_swm(
                    model_instance,
                    n_boxes,
                    n_tokens=n_tokens,
                    cot=cot,
                    think_budget=think_budget,
                    note_assist=notes,
                )
        except Exception as e:
            print(f"Run {i+1} failed with error: {e}")
            continue
        run_history[f"run_{i+1}"] = model_instance.history
        run_reasoning[f"run_{i+1}"] = (
            model_instance.reasoning_trace
        )  # Save reasoning trace

        with open(file_header + "run_stats.json", "w") as f:
            json.dump(run_stats, f, indent=4)

        with open(file_header + "run_history.json", "w") as f:
            json.dump(run_history, f, indent=4)

        with open(
            file_header + "run_structured_history.json", "w"
        ) as f:  # Save structured history
            json.dump(structured_history, f, indent=4)

        with open(
            file_header + "run_reasoning.json", "w"
        ) as f:  # Save reasoning traces
            json.dump(run_reasoning, f, indent=4)

    avg_stats = {}
    for key in run_stats["run_1"].keys():
        if type(run_stats["run_1"][key]) == int:
            avg_stats[key] = np.mean(
                [
                    stats[key]
                    for stats in run_stats.values()
                    if stats.get("finished_run", False)
                ]
            )

    for key, value in avg_stats.items():
        print(f"{key}: {value}")

    tot_score = 0
    for stats in run_stats.values():
        tot_score += score(stats)
    print(f"Score: {tot_score / len(run_stats.keys())}")
    print(
        f"Finished runs: {sum([1 for stats in run_stats.values() if not stats.get('finished_run', False)])} out of {runs}"
    )


# ============================================================================
# Environment-based evaluation (using SWMEnv)
# ============================================================================


def run_swm_with_env(
    model=None,
    model_source: str = "vllm",
    n_boxes: int = 6,
    n_tokens: int = 1,
    mode: str = "text",  # "text" or "image"
    cot: bool = False,
    think_budget: int = 64,
    note_assist: bool = False,
    runs: int = 1,
    max_tokens: int = 512,
    api_key=None,
    save_dir=None,
    verbose: bool = True,
):
    """
    Run SWM evaluation using the SWMEnv environment class.

    This provides a cleaner interface for RL training integration while
    maintaining compatibility with existing analysis code.

    Args:
        model: Model name/path
        model_source: Model source ("vllm", "openai", "openrouter", etc.)
        n_boxes: Number of boxes in the task
        n_tokens: Number of token types
        mode: "text" or "image"
        cot: Enable chain-of-thought reasoning
        think_budget: Token budget for CoT
        note_assist: Enable note-taking assistance
        runs: Number of runs
        max_tokens: Max tokens to generate
        api_key: API key for model
        save_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dict with run statistics and histories
    """
    from .swm_env import SWMEnv

    # Default model
    if model is None:
        if model_source == "vllm":
            model = "Qwen/Qwen3-32B"
        elif model_source == "openai":
            model = "o4-mini-2025-04-16"
        elif model_source == "openrouter":
            model = "qwen/qwen3-235b-a22b-07-25"
        elif model_source == "google":
            model = "gemini-3-pro"

    # Set up save paths
    if save_dir is None:
        save_dir = os.path.join("SWM", "data", "env")
    os.makedirs(save_dir, exist_ok=True)

    image_mode = mode == "image"
    image_path = os.path.join("SWM", "images") if image_mode else None
    if image_path:
        os.makedirs(image_path, exist_ok=True)

    cot_suffix = "_cot" if cot else ""
    notes_suffix = "_notes" if note_assist else ""

    save_prefix = os.path.join(
        save_dir,
        f"{model_source}_{model.replace('/', '-')}_{mode}{cot_suffix}{notes_suffix}_{n_boxes}_{n_tokens}",
    )

    results = {
        "runs": {},
        "config": {
            "model": model,
            "model_source": model_source,
            "n_boxes": n_boxes,
            "n_tokens": n_tokens,
            "mode": mode,
            "cot": cot,
            "note_assist": note_assist,
        },
    }

    for run_idx in range(runs):
        if verbose:
            print(f"\n=== Run {run_idx + 1}/{runs} ===")

        torch.cuda.empty_cache()

        # Create environment
        env = SWMEnv(
            n_boxes=n_boxes,
            n_tokens=n_tokens,
            mode=mode,
            cot=cot,
            think_budget=think_budget,
            note_assist=note_assist,
            image_path=image_path,
            seed=run_idx,
        )

        # Create model wrapper
        model_instance = ModelWrapper(
            model,
            model_source,
            api_key=api_key,
            max_new_tokens=max_tokens,
            think_budget=think_budget,
            image_input=image_mode,
            image_path=image_path,
        )

        # Initialize with system prompt
        system_prompt = env.get_system_prompt()
        model_instance.init_chat(system_prompt)

        # Run episode
        observation = env.reset()
        episode_history = []

        while not env._done:
            # Get model response
            response = model_instance.send_message(
                observation,
                truncate_history=True,
                cot=cot,
            )

            if response is None:
                if verbose:
                    print("Model returned None, ending episode early")
                break

            # Step environment
            result = env.step(response)

            # Record step
            episode_history.append(
                {
                    "observation": observation,
                    "response": response,
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                }
            )

            observation = result.observation

            if result.done:
                break

        # Get metrics
        metrics = env.get_metrics()

        # Store results
        run_key = f"run_{run_idx + 1}"
        results["runs"][run_key] = {
            "history": env.history,
            "episode_history": episode_history,
            "metrics": metrics,
            "model_history": model_instance.history,
            "reasoning_trace": model_instance.reasoning_trace,
        }

        if verbose:
            print(f"Tokens found: {metrics.get('tokens_found', 0)}")
            print(f"Valid guesses: {metrics.get('valid', 0)}")
            print(f"Total guesses: {metrics.get('total_guesses', 0)}")

        # Compute score
        env_score = score(metrics)
        if verbose:
            print(f"Score: {env_score:.3f}")

        # Save intermediate results
        with open(f"{save_prefix}_results.json", "w") as f:
            json_results = {"config": results["config"], "runs": {}}
            for rk, rv in results["runs"].items():
                json_results["runs"][rk] = {
                    "history": rv["history"],
                    "metrics": rv["metrics"],
                }
            json.dump(json_results, f, indent=2)

    return results
