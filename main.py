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
import json

# Add the current directory to Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from RAPM.rapm_evaluation import (
        run_rapm_evaluation,
        run_text_rapm_evaluation,
    )
except ImportError:
    run_rapm_evaluation = run_text_rapm_evaluation = None  # type: ignore


def orchestrate():
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
    wcst_parser.add_argument(
        "--ambiguous", type=str, default="off", choices=["off", "first", "rest"]
    )
    wcst_parser.add_argument("--few_shot", action="store_true")
    wcst_parser.add_argument("--cot", action="store_true")
    wcst_parser.add_argument("--hint", action="store_true")
    wcst_parser.add_argument("--notes", action="store_true")
    wcst_parser.add_argument("--notes-window", type=int, default=6)
    wcst_parser.add_argument("--image", action="store_true")
    wcst_parser.add_argument("--bg-color", action="store_true")
    wcst_parser.add_argument(
        "--model_source",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "openrouter", "google"],
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
        choices=["vllm", "openai", "openrouter", "google"],
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
    swm_parser.add_argument("--image-only", action="store_true")

    # RAPM parser
    rapm_parser = subparsers.add_parser(
        "rapm", help="Run Raven's Progressive Matrices (image or text)"
    )
    rapm_parser.add_argument("--model", type=str, default=None, help="Model name")
    rapm_parser.add_argument(
        "--model_source",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "openrouter", "google"],
    )
    rapm_parser.add_argument(
        "--mode",
        type=str,
        default="image",
        choices=["image", "text"],
        help="RAPM mode: image JSON or text JSONL",
    )
    rapm_parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="Path to RAPM data file (JSON for image, JSONL for text)",
    )
    rapm_parser.add_argument(
        "--cot", action="store_true", help="Enable chain-of-thought reasoning"
    )
    rapm_parser.add_argument(
        "--patterns",
        action="store_true",
        help="Include pattern archetype descriptions in system prompt",
    )
    rapm_parser.add_argument("--max_tokens", type=int, default=512)
    rapm_parser.add_argument("--think_budget", type=int, default=256)
    rapm_parser.add_argument("--api_key", type=str, default=None)
    rapm_parser.add_argument(
        "--limit_per_type",
        type=int,
        default=100,
        help="Limit per dataset type (image mode only; 0 = no limit)",
    )
    rapm_parser.add_argument(
        "--output_dir",
        type=str,
        default="rapm_data",
        help="Directory to write RAPM results",
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

    human_parser = subparsers.add_parser(
        "human", help="Launch browser-based human benchmark runner"
    )
    human_parser.add_argument("--host", type=str, default="127.0.0.1")
    human_parser.add_argument("--port", type=int, default=7860)
    human_parser.add_argument("--share", action="store_true")
    human_parser.add_argument("--inbrowser", action="store_true")
    human_parser.add_argument("--output_dir", type=str, default="human_data")

    args = parser.parse_args()

    if args.test == "wcst":
        if args.image:
            from WCST.wcst import run_wcst_image  # type: ignore

            run_wcst_image(
                model=args.model,
                max_trials=args.max_trials,
                num_correct=args.num_correct,
                repeats=args.repeats,
                bg_color=args.bg_color,
                ambiguous_mode=args.ambiguous,
                few_shot=args.few_shot,
                cot=args.cot,
                hint=args.hint,
                notes=args.notes,
                notes_window=args.notes_window,
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
                bg_color=args.bg_color,
                ambiguous_mode=args.ambiguous,
                few_shot=args.few_shot,
                cot=args.cot,
                hint=args.hint,
                notes=args.notes,
                notes_window=args.notes_window,
                model_source=args.model_source,
                max_tokens=args.max_tokens,
                think_budget=args.think_budget,
                api_key=args.api_key,
                verbose=args.verbose,
            )
    elif args.test == "swm":
        # Delegate to SWM package runner which handles runs and saving results
        from SWM.main import swm_main  # type: ignore

        swm_main(
            model=args.model,
            model_source=args.model_source,
            n_boxes=args.n_boxes,
            n_tokens=args.n_tokens,
            cot=args.cot,
            runs=args.runs,
            max_tokens=args.max_tokens,
            think_budget=args.think_budget,
            notes=args.notes,
            image=args.image,
            api_key=args.api_key,
            image_only=args.image_only,
        )
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
        base_name = (
            f"{args.model_source}_{args.model.replace('/', '-')}_rapm_{args.mode}"
        )
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
        existing_results = []
        existing_reasoning = []
        existing_summary = None
        if args.batch_mode == "off":
            if os.path.exists(results_path):
                try:
                    with open(results_path, "r") as f:
                        existing_results = json.load(f)
                    print(
                        f"Loaded {len(existing_results)} existing RAPM results; will resume run."
                    )
                except Exception as exc:
                    print(
                        f"Warning: couldn't load existing results ({exc}); starting fresh."
                    )
                    existing_results = []
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r") as f:
                        existing_summary = json.load(f)
                except Exception as exc:
                    print(f"Warning: couldn't load existing summary ({exc}); ignoring.")
                    existing_summary = None
            if os.path.exists(reasoning_path):
                try:
                    with open(reasoning_path, "r") as f:
                        existing_reasoning = json.load(f)
                except Exception as exc:
                    print(
                        f"Warning: couldn't load existing reasoning traces ({exc}); ignoring."
                    )
                    existing_reasoning = []
            if args.mode == "text" and isinstance(existing_summary, dict):
                failed_from_summary = existing_summary.get("failed_items") or []
                if failed_from_summary:
                    print(
                        f"Summary lists {len(failed_from_summary)} failed text items; they will be retried."
                    )
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
            reasoning_path = os.path.join(
                args.output_dir, f"{base_name}_reasoning.json"
            )
        else:
            if args.mode == "image":
                print("Starting RAPM image evaluation...")
                results, summary, history, reasoning_traces = run_rapm_evaluation(  # type: ignore
                    args,
                    existing_results=existing_results,
                    existing_reasoning=existing_reasoning,
                    existing_summary=existing_summary,
                    results_path=results_path,
                    summary_path=summary_path,
                    reasoning_path=reasoning_path,
                )
            else:
                print("Starting RAPM text evaluation...")
                results, summary, history, reasoning_traces = run_text_rapm_evaluation(  # type: ignore
                    args,
                    existing_results=existing_results,
                    existing_reasoning=existing_reasoning,
                    existing_summary=existing_summary,
                    results_path=results_path,
                    summary_path=summary_path,
                    reasoning_path=reasoning_path,
                )
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
    elif args.test == "human":
        from human_benchmark import launch_human_benchmark  # type: ignore

        launch_human_benchmark(args)


if __name__ == "__main__":
    orchestrate()
