#!/usr/bin/env python3
"""
Main script to run cognitive evaluation tests.
Supports:
- Wisconsin Card Sorting Test (WCST)
- Spatial Working Memory (SWM)
- Raven's Progressive Matrices (RAPM) image & text variants

This file merged legacy SWM runner and new orchestrator. The legacy SWM run/score
functions moved into run_swm / score for backward compatibility.
"""

import argparse
import sys
import os
import random, json, re, string
import numpy as np
import torch
from tqdm.auto import tqdm

# Add the current directory to Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model_wrapper import ModelWrapper
except ImportError:
    from .model_wrapper import ModelWrapper  # type: ignore

try:
    from swm import image_swm  # legacy image mode
except Exception:
    image_swm = None  # type: ignore

# RAPM imports
try:
    from RAPM.rapm_evaluation import (
        run_rapm_evaluation,
        run_text_rapm_evaluation,
    )
except ImportError:
    run_rapm_evaluation = run_text_rapm_evaluation = None  # type: ignore

# ---------------- Legacy SWM implementation (from HEAD) ---------------- #


def run_swm(model, n_boxes, n_tokens=1, cot=None, think_budget=64, note_assist=False):
    """Run the (text) Spatial Working Memory (SWM) test with the given model."""
    task_prompt = f"""You will be performing a text version of the Spatial Working Memory (SWM) test.
There are {n_tokens} types of tokens, hidden in any one of {n_boxes} boxes.
Your goal is to find the {n_tokens} types of tokens {n_boxes} times each, by repeatedly selecting a box to open.
If the box contains a token, you will be informed which token type it is.
If the box does not contain a token, you will be informed that it is empty.
Once the token is found, another token of the same type will be regenerated in another box.
The token will be generated in a box that has never contained a token of that type before in the trial.
The token may be generated in a box that has been opened and found empty before, as long as it never contained the token of that type previously.
Your final answer should be a number from 1-{n_boxes}, the index of the box you selected.
"""
    model.init_chat(task_prompt)

    if cot is not None:
        cot_prompt = (
            f"Think step-by-step, utilizing information from previous feedbacks, and state your reasoning in maximum {think_budget} tokens, "
            "wrapped with <think> and </think>. After </think>, give a short summary and final answer.\n"
        )
        question = (
            f"Answer concisely. {cot_prompt}Which of the {n_boxes} boxes would you like to open?\n"
            f"Your final answer should be a box number, wrapped with <answer> and </answer>"
        )
    else:
        question = (
            f"Answer only with your final answer. Which of the {n_boxes} boxes would you like to open?\n"
            f"Your final answer should be a box number, wrapped with <answer> and </answer>"
        )

    tokens = [string.ascii_uppercase[x] for x in range(n_tokens)]
    legal_boxes = {t: [x for x in range(1, n_boxes + 1)] for t in tokens}

    worst_case_n = n_boxes**2
    total_guess = illegal_guess = invalid_guess = repeated_guess = 0

    response = model.send_message(question, cot=cot, truncate_history=True)
    with tqdm(total=worst_case_n, desc="Total guesses") as guess_bar, tqdm(
        total=n_boxes * n_tokens, desc="Tokens"
    ) as token_bar:
        token_box = {t: random.choice(legal_boxes[t]) for t in tokens}
        found_tokens = []
        while True:
            for token in found_tokens:
                if len(legal_boxes[token]) == 0:
                    continue
                token_box[token] = random.choice(legal_boxes[token])

            with open("data/temp_history.json", "w") as f:
                json.dump(model.history, f, indent=4)

            if (
                all(len(legal) == 0 for legal in legal_boxes.values())
                or total_guess >= worst_case_n
            ):
                break

            opened_boxes = set()
            found_tokens = []
            found = False
            while not found:
                total_guess += 1
                guess_bar.update(1)
                if total_guess >= worst_case_n:
                    break

                notes = ""
                if note_assist:
                    for token, legal in legal_boxes.items():
                        notes += f"Boxes that has contained token {token}: "
                        for box in range(1, n_boxes + 1):
                            if box not in legal:
                                notes += f"{box}, "
                        notes += "\n"
                    notes += (
                        "Opened boxes: "
                        + ", ".join(str(b) for b in opened_boxes)
                        + "\n"
                    )

                msg = "".join(
                    f"{token} tokens found: {n_boxes - len(legal_boxes[token])}\n"
                    for token in tokens
                )

                match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", response)
                if match is not None:
                    chosen_box = match.group(1).strip()
                    try:
                        chosen_box = int(chosen_box)
                    except ValueError:
                        response = model.send_message(
                            f"Please answer with a box number (1-{n_boxes}).\n"
                            + msg
                            + notes
                            + question,
                            truncate_history=True,
                            cot=cot,
                        )
                        invalid_guess += 1
                        continue
                else:
                    response = model.send_message(
                        f"Please answer with the specified format\n"
                        + msg
                        + notes
                        + question,
                        truncate_history=True,
                        cot=cot,
                    )
                    invalid_guess += 1
                    continue

                legal = any(chosen_box in legal for legal in legal_boxes.values())
                if not legal:
                    illegal_guess += 1
                elif chosen_box in opened_boxes:
                    repeated_guess += 1

                opened_boxes.add(chosen_box)
                found = False
                found_tokens = []
                for token in tokens:
                    if chosen_box == token_box[token]:
                        found = True
                        token_box[token] = -1
                        token_bar.update(1)
                        legal_boxes[token].remove(chosen_box)
                        found_tokens.append(token)

                if found:
                    msg = "".join(
                        f"Token {token} found in box {chosen_box}.\n"
                        for token in found_tokens
                    )
                else:
                    msg = f"No tokens found in box {chosen_box}.\n"

                response = model.send_message(
                    msg + notes + question, truncate_history=True, cot=cot
                )
                model.history[-2]["content"] = msg

    return {
        "worst_case_guesses": worst_case_n,
        "illegal": illegal_guess,
        "guesses": total_guess,
        "invalid": invalid_guess,
        "repeated": repeated_guess,
    }


def score(run_stats):
    return (
        1
        - (run_stats["illegal"] + run_stats["repeated"])
        / (run_stats["guesses"] - run_stats["invalid"])
        if (run_stats["guesses"] - run_stats["invalid"])
        else 0.0
    )


# ---------------- New orchestrator entrypoint (from image branch) ---------------- #


def orchestrate():  # renamed from main() to avoid confusion
    parser = argparse.ArgumentParser(
        description="Run cognitive evaluation tests (WCST or SWM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py wcst
  python main.py swm --cot
  python main.py wcst --model gpt-4 --model_source openai --few_shot
  python main.py swm --image --runs 5 --n_boxes 8
        """,
    )
    subparsers = parser.add_subparsers(dest="test", required=True)

    wcst_parser = subparsers.add_parser("wcst", help="Run Wisconsin Card Sorting Test")
    wcst_parser.add_argument("--model", type=str, default="llama")
    wcst_parser.add_argument(
        "--variant",
        type=str,
        default="card",
        choices=["card", "card-random", "string", "empty"],
    )
    wcst_parser.add_argument("--max_trials", type=int, default=64)
    wcst_parser.add_argument("--num_correct", type=int, default=5)
    wcst_parser.add_argument("--repeats", type=int, default=1)
    wcst_parser.add_argument("--few_shot", action="store_true")
    wcst_parser.add_argument("--cot", action="store_true")
    wcst_parser.add_argument("--hint", action="store_true")
    wcst_parser.add_argument("--image", action="store_true")
    wcst_parser.add_argument(
        "--model_source",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "openrouter"],
    )
    wcst_parser.add_argument("--max_tokens", type=int, default=512)
    wcst_parser.add_argument("--think_budget", type=int, default=64)
    wcst_parser.add_argument("--api_key", type=str, default=None)
    wcst_parser.add_argument("--verbose", type=int, default=15)

    swm_parser = subparsers.add_parser("swm", help="Run Spatial Working Memory test")
    swm_parser.add_argument("--model", type=str, default=None)
    swm_parser.add_argument(
        "--model_source",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "openrouter"],
    )
    swm_parser.add_argument("--n_boxes", type=int, default=6)
    swm_parser.add_argument("--n_tokens", type=int, default=1)
    swm_parser.add_argument("--cot", action="store_true")
    swm_parser.add_argument("--runs", type=int, default=1)
    swm_parser.add_argument("--max_tokens", type=int, default=512)
    swm_parser.add_argument("--think_budget", type=int, default=64)
    swm_parser.add_argument("--notes", action="store_true")
    swm_parser.add_argument("--image", action="store_true")
    swm_parser.add_argument("--api_key", type=str, default=None)

    # RAPM parser
    rapm_parser = subparsers.add_parser("rapm", help="Run Raven's Progressive Matrices (image or text)")
    rapm_parser.add_argument("--model", type=str, default=None, help="Model name")
    rapm_parser.add_argument(
        "--model_source",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "openrouter"],
    )
    rapm_parser.add_argument(
        "--mode", type=str, default="image", choices=["image", "text"], help="RAPM mode: image JSON or text JSONL"
    )
    rapm_parser.add_argument(
        "--eval_data", type=str, required=True, help="Path to RAPM data file (JSON for image, JSONL for text)"
    )
    rapm_parser.add_argument("--cot", action="store_true", help="Enable chain-of-thought reasoning")
    rapm_parser.add_argument(
        "--patterns", action="store_true", help="Include pattern archetype descriptions in system prompt"
    )
    rapm_parser.add_argument("--max_tokens", type=int, default=512)
    rapm_parser.add_argument("--think_budget", type=int, default=256)
    rapm_parser.add_argument("--api_key", type=str, default=None)
    rapm_parser.add_argument(
        "--limit_per_type", type=int, default=100, help="Limit per dataset type (image mode only; 0 = no limit)"
    )
    rapm_parser.add_argument(
        "--output_dir", type=str, default="rapm_data", help="Directory to write RAPM results"
    )
    rapm_parser.add_argument("--verbose", type=int, default=15)
    rapm_parser.add_argument(
        "--answer_mode",
        type=str,
        default="mc",
        choices=["mc", "gen"],
        help="Text RAPM only: 'mc' for multiple-choice, 'gen' to generate the missing cell directly.",
    )
    rapm_parser.add_argument(
        "--batch_mode",
        type=str,
        default="off",
        choices=["off", "submit", "collect"],
        help="OpenAI Batch API mode for RAPM (off/submit/collect).",
    )
    rapm_parser.add_argument(
        "--batch_requests_path",
        type=str,
        default="rapm_batch_requests.jsonl",
        help="Path to write RAPM batch requests JSONL (submit).",
    )
    rapm_parser.add_argument(
        "--batch_id",
        type=str,
        default=None,
        help="Batch id for collection (if omitted, read from --batch_id_path).",
    )
    rapm_parser.add_argument(
        "--batch_id_path",
        type=str,
        default="rapm_batch_id.txt",
        help="File to store/read batch id when using submit/collect.",
    )
    rapm_parser.add_argument(
        "--batch_output_jsonl",
        type=str,
        default="rapm_batch_output.jsonl",
        help="Where to save raw batch output JSONL during collection.",
    )
    rapm_parser.add_argument(
        "--batch_completion_window",
        type=str,
        default="24h",
        help="Batch completion window request (e.g. 24h).",
    )

    args = parser.parse_args()

    if args.test == "wcst":
        if args.image:
            from WCST.wcst import run_wcst_image  # type: ignore

            run_wcst_image(
                model=args.model,
                max_trials=args.max_trials,
                num_correct=args.num_correct,
                repeats=args.repeats,
                few_shot=args.few_shot,
                cot=args.cot,
                hint=args.hint,
                model_source=args.model_source,
                max_tokens=args.max_tokens,
                think_budget=args.think_budget,
                api_key=args.api_key,
                verbose=args.verbose,
            )
        else:
            from WCST.wcst import run_wcst  # type: ignore

            run_wcst(
                model=args.model,
                variant=args.variant,
                max_trials=args.max_trials,
                num_correct=args.num_correct,
                repeats=args.repeats,
                few_shot=args.few_shot,
                cot=args.cot,
                hint=args.hint,
                model_source=args.model_source,
                max_tokens=args.max_tokens,
                think_budget=args.think_budget,
                api_key=args.api_key,
                verbose=args.verbose,
            )
    elif args.test == "swm":
        # run multiple runs using legacy functions
        if args.model is None:
            if args.model_source == "vllm":
                args.model = "Qwen/Qwen3-32B"
            elif args.model_source == "openai":
                args.model = "o4-mini-2025-04-16"
            else:
                args.model = "qwen/qwen3-235b-a22b-07-25"
        img_path = "./images" if args.image else None
        if args.image and image_swm is None:
            raise RuntimeError("image_swm not available")
        run_stats = {}
        for i in range(args.runs):
            torch.cuda.empty_cache()
            model = ModelWrapper(
                args.model,
                args.model_source,
                api_key=args.api_key,
                max_new_tokens=args.max_tokens,
                image_input=args.image,
                image_path=img_path,
            )
            if args.image:
                if image_swm is None:
                    raise RuntimeError("image_swm function unavailable for image mode")
                run_stats[f"run_{i+1}"] = image_swm(
                    model,
                    args.n_boxes,
                    n_tokens=args.n_tokens,
                    cot=args.cot,
                    think_budget=args.think_budget,
                    note_assist=args.notes,
                )
            else:
                run_stats[f"run_{i+1}"] = run_swm(
                    model,
                    args.n_boxes,
                    n_tokens=args.n_tokens,
                    cot=args.cot,
                    think_budget=args.think_budget,
                    note_assist=args.notes,
                )
        # simple aggregate output
        avg = np.mean([score(s) for s in run_stats.values()])
        print(f"Average SWM score over {args.runs} runs: {avg:.4f}")
    elif args.test == "rapm":
        # Validate RAPM availability
        if args.mode == "image" and (run_rapm_evaluation is None):
            raise SystemExit("RAPM image evaluation function not available")
        if args.mode == "text" and (run_text_rapm_evaluation is None):
            raise SystemExit("RAPM text evaluation function not available")
        # Default model if not provided
        if args.model is None:
            if args.model_source == "vllm":
                args.model = "Qwen/Qwen3-32B"
            elif args.model_source == "openai":
                args.model = "o4-mini-2025-04-16"
            else:
                args.model = "qwen/qwen3-235b-a22b-07-25"
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        base_name = f"{args.model_source}_{args.model.replace('/', '-')}_rapm_{args.mode}"
        if args.mode == "text" and getattr(args, "answer_mode", "mc") == "gen":
            base_name += "_gen"
        if args.patterns:
            base_name += "_pat"
        if args.cot:
            base_name += "_cot"
        results_path = os.path.join(args.output_dir, f"{base_name}_results.json")
        summary_path = os.path.join(args.output_dir, f"{base_name}_summary.json")
        history_path = os.path.join(args.output_dir, f"{base_name}_history.json")
        reasoning_path = os.path.join(args.output_dir, f"{base_name}_reasoning.json")
        if os.path.exists(results_path) and args.batch_mode == "off":
            print(f"Results already exist at {results_path}")
            return
        # Defer to module's batch logic by reusing its CLI style flow
        from RAPM.rapm_evaluation import batch_submit_rapm, batch_collect_rapm  # type: ignore
        if args.batch_mode == "submit":
            batch_id = batch_submit_rapm(args)
            print(f"Submitted RAPM batch {batch_id}")
            return
        if args.batch_mode == "collect":
            if not args.batch_id and os.path.exists(args.batch_id_path):
                with open(args.batch_id_path, "r") as f:
                    args.batch_id = f.read().strip()
            if not args.batch_id:
                raise SystemExit("No batch id provided or stored.")
            results, summary, history, reasoning_traces = batch_collect_rapm(args)
            base_name += "_batch"
            results_path = os.path.join(args.output_dir, f"{base_name}_results.json")
            summary_path = os.path.join(args.output_dir, f"{base_name}_summary.json")
            history_path = os.path.join(args.output_dir, f"{base_name}_history.json")
            reasoning_path = os.path.join(args.output_dir, f"{base_name}_reasoning.json")
        else:
            if args.mode == "image":
                print("Starting RAPM image evaluation...")
                results, summary, history, reasoning_traces = run_rapm_evaluation(args)  # type: ignore
            else:
                print("Starting RAPM text evaluation...")
                results, summary, history, reasoning_traces = run_text_rapm_evaluation(args)  # type: ignore
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        if reasoning_traces:
            with open(reasoning_path, "w") as f:
                json.dump(reasoning_traces, f, indent=2)
        print("RAPM evaluation complete!")


if __name__ == "__main__":
    orchestrate()
