#!/usr/bin/env python3
"""
RAPM (Raven's Progressive Matrices) Evaluation Script

Supports two modes:
- image: classic visual RPM items (JSON with questions list referencing image files)
- text: text-based 3x3 string pattern matrices (JSONL, one item per line)

This script evaluates language models on the Raven's Progressive Matrices test,
a non-verbal intelligence test that measures abstract reasoning ability.

Usage examples:
  # Basic evaluation
  python rapm_evaluation.py --model "gpt-4-vision-preview" --model_source openai --eval_data /path/to/raven_evaluation_data.json

  # With chain-of-thought reasoning
  python rapm_evaluation.py --model "gpt-4-vision-preview" --model_source openai --eval_data /path/to/raven_evaluation_data.json --cot

  # Using OpenRouter
  python rapm_evaluation.py --model "anthropic/claude-3-opus" --model_source openrouter --eval_data /path/to/raven_evaluation_data.json --cot

The script will:
1. Load the evaluation data from the JSON file
2. For each question, send the full image (matrix + answer choices) to the model
3. Extract the model's answer and check if it's correct
4. Calculate accuracy overall and by dataset type
5. Save results, history, and reasoning traces (if CoT is enabled)
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None):
        print(f"{desc}..." if desc else "Processing...")
        return iterable


from model_wrapper import ModelWrapper

# Base image prompt (without pattern catalogue)
RAPM_BASE_PROMPT = (
    "You are taking the Raven's Progressive Matrices (RPM) test, a non-verbal intelligence test that measures abstract reasoning ability.\n\n"
    "You will see a 3x3 matrix of images with the bottom-right image missing (shown as a question mark), followed by 8 answer choices numbered 1-8.\n\n"
    "Your task is to: \n"
    "1. Analyze rows and columns\n2. Infer the governing logical rule(s)\n3. Select the answer choice (1-8) that correctly completes the matrix.\n\n"
)
RAPM_PATTERN_INFO = (
    "The patterns can involve: \n"
    "- Shape transformations (rotation, reflection, scaling)\n"
    "- Position changes (movement, arrangement)\n"
    "- Attribute changes (color, size, number of elements)\n"
    "- Logical operations (addition, subtraction, intersection)\n"
    "- Sequence progressions (systematic changes across rows/columns)\n\n"
)
# New: concise additional rule archetypes (image mode)
RAPM_ADDITIONAL_RULES = (
    "Additional common rule types:\n"
    "- Constant-in-row: Same value across a row; varies down columns.\n"
    "- Quantitative step: Fixed +/− increment between adjacent cells (size / count / position offset).\n"
    "- Figure add/subtract: Combine (overlay or juxtapose) or remove elements from two cells to form the third.\n"
    "- Distribution-of-three: Three distinct categorical values appear once each per row (order may permute).\n"
    "- Distribution-of-two: Two values each appear once; third slot is empty / null.\n\n"
    "Look horizontally and vertically; the missing piece must satisfy ALL relevant row and column rules.\n\n"
)
RAPM_ANSWER_SUFFIX = "Your final answer should be a number between 1-8 corresponding to the correct choice.\n"

# Base text prompt (without pattern catalogue)
TEXT_RAPM_BASE_PROMPT = (
    "You are solving a TEXT-BASED 3x3 pattern matrix (Raven-style). Each cell contains a string; the bottom-right cell is missing ('?').\n\n"
    "Goal: Infer the rule(s) acting across rows and columns and pick which option (1-8) correctly fills the missing cell.\n\n"
)
TEXT_RAPM_PATTERN_INFO = (
    "Possible dimensions (one or more):\n"
    "- Character set restriction (digits / letters / symbols)\n"
    "- Quantitative constant (exact length / count / unique)\n"
    "- Quantitative progression (arithmetic step across row/column)\n"
    "- Parity / multiple rules (all even / all odd / multiples of N)\n"
    "- Positional constraints (first/last/even/odd positions restricted)\n"
    "- Ordering (ascending / descending / mixed)\n"
    "- Layered combinations (e.g. constant + parity, progression + positional)\n\n"
    "Select the option satisfying ALL inferred row and column constraints.\n\n"
)


def load_evaluation_data(eval_data_path, limit_per_type=None):
    """Load the evaluation data from JSON file.

    Args:
        eval_data_path: Path to the JSON file containing questions
        limit_per_type: If provided, limit to this many questions per dataset type
    """
    with open(eval_data_path, "r") as f:
        data = json.load(f)

    questions = data["questions"]

    if limit_per_type is not None:
        # Group questions by dataset type
        questions_by_type = defaultdict(list)
        for question in questions:
            questions_by_type[question["dataset_type"]].append(question)

        # Take only the first N questions from each type
        limited_questions = []
        for dataset_type, type_questions in questions_by_type.items():
            limited_questions.extend(type_questions[:limit_per_type])
            print(
                f"Selected {min(len(type_questions), limit_per_type)} questions from {dataset_type}"
            )

        questions = limited_questions
        print(f"Total questions after limiting: {len(questions)}")

    return questions


# --- New: load text RAPM JSONL ---
def load_text_rapm_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Normalize schema
            qgrid = obj.get("question_grid") or obj.get("full_grid")
            options = obj.get("options", [])
            correct_index = obj.get("correct_index")
            items.append(
                {
                    "id": obj.get("id") or f"text_rapm_{len(items)}",
                    "question_grid": qgrid,
                    "options": options,
                    "correct_index": correct_index,
                    "raw": obj,
                }
            )
    return items


def run_rapm_evaluation(args):
    """Run the RAPM evaluation."""

    # Image-based mode
    limit = args.limit_per_type if args.limit_per_type > 0 else None
    questions = load_evaluation_data(args.eval_data, limit_per_type=limit)
    print(f"Loaded {len(questions)} questions")

    # Get unique dataset types for statistics
    dataset_types = set(q["dataset_type"] for q in questions)
    print(f"Dataset types: {dataset_types}")

    # Initialize model
    model = ModelWrapper(
        args.model,
        args.model_source,
        api_key=args.api_key,
        max_new_tokens=args.max_tokens,
        think_budget=args.think_budget,
        image_input=True,
        image_path=os.path.dirname(
            args.eval_data
        ),  # Set to directory containing the JSON file
    )

    # Build system prompt conditionally
    system_prompt = RAPM_BASE_PROMPT
    if args.patterns:
        system_prompt += RAPM_PATTERN_INFO + RAPM_ADDITIONAL_RULES
    system_prompt += RAPM_ANSWER_SUFFIX
    if args.cot:
        system_prompt += f"\nExplain your thought process (max {args.think_budget} tokens) inside <think>...</think> then give final answer.\n"
    else:
        system_prompt += "\nAnswer only with your final answer.\n"
    system_prompt += "State your final answer as: <answer>number</answer>\n"

    # Results storage
    results = []
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    reasoning_traces = []

    # Process each question
    for i, question in enumerate(tqdm(questions, desc="Processing questions")):
        model.init_chat(system_prompt)
        # Get image path
        full_image_path = os.path.join(
            os.path.dirname(args.eval_data), question["full_image"]
        )

        # Copy image to current.png in the image_path directory (required by model wrapper)
        current_image_path = os.path.join(model.image_path, "current.png")  # type: ignore

        # Use Python's shutil for more reliable copying
        shutil.copy2(full_image_path, current_image_path)

        # Send image to model (empty text message since everything is in the image)
        response = model.send_message(
            "",  # Empty message since question and choices are in the image
            cot=args.cot,
            truncate_history=True,  # Truncate history to avoid exceeding token limits
        )

        # Extract answer
        answer_match = re.search(r"<answer>(.*?)</answer>", response)
        predicted_answer = None
        if answer_match:
            answer_text = answer_match.group(1).strip()
            numbers = re.findall(r"\d+", answer_text)
            if numbers:
                try:
                    predicted_answer = int(numbers[0])
                    # Validate answer is in valid range (1-8 for choices, but stored as 0-7)
                    if predicted_answer < 1 or predicted_answer > 8:
                        predicted_answer = None
                except ValueError:
                    predicted_answer = None

        # Check if correct
        # The correct_answer in the data is 0-indexed (0-7), predicted_answer is 1-indexed (1-8)
        correct_answer = question["correct_answer"]
        is_correct = (
            predicted_answer is not None and (predicted_answer - 1) == correct_answer
        )

        # Update statistics
        dataset_type = question["dataset_type"]
        total_by_type[dataset_type] += 1
        if is_correct:
            correct_by_type[dataset_type] += 1

        # Store result
        result = {
            "id": question["id"],
            "dataset_type": dataset_type,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "response": response,
            "image_path": question["full_image"],
        }
        results.append(result)

        # Store reasoning if CoT was used
        if args.cot and model.reasoning_trace:
            reasoning_traces.append(
                {
                    "question_id": question["id"],
                    "reasoning": (
                        model.reasoning_trace[-1] if model.reasoning_trace else None
                    ),
                }
            )

        # Print progress every 100 questions
        if (i + 1) % 100 == 0:
            current_accuracy = sum(r["is_correct"] for r in results) / len(results)
            print(
                f"Progress: {i+1}/{len(questions)}, Current accuracy: {current_accuracy:.3f}"
            )

    # Clean up temporary image
    if os.path.exists(current_image_path):
        os.remove(current_image_path)

    # Calculate final statistics
    total_correct = sum(r["is_correct"] for r in results)
    total_questions = len(results)
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    # Print final results
    print("\n" + "=" * 50)
    print("RAPM EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Total questions: {total_questions}")
    print(
        f"Overall accuracy: {overall_accuracy:.3f} ({total_correct}/{total_questions})"
    )
    print("\nAccuracy by dataset type:")
    for dataset_type in sorted(dataset_types):
        type_accuracy = (
            correct_by_type[dataset_type] / total_by_type[dataset_type]
            if total_by_type[dataset_type] > 0
            else 0
        )
        print(
            f"  {dataset_type}: {type_accuracy:.3f} ({correct_by_type[dataset_type]}/{total_by_type[dataset_type]})"
        )

    # Prepare summary data
    summary = {
        "model": args.model,
        "model_source": args.model_source,
        "total_questions": total_questions,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "accuracy_by_type": {
            dataset_type: {
                "correct": correct_by_type[dataset_type],
                "total": total_by_type[dataset_type],
                "accuracy": (
                    correct_by_type[dataset_type] / total_by_type[dataset_type]
                    if total_by_type[dataset_type] > 0
                    else 0
                ),
            }
            for dataset_type in dataset_types
        },
        "args": vars(args),
    }

    return results, summary, model.history, reasoning_traces


# --- New: text RAPM evaluation ---
def run_text_rapm_evaluation(args):
    items = load_text_rapm_jsonl(args.eval_data)
    print(f"Loaded {len(items)} text RAPM items")
    model = ModelWrapper(
        args.model,
        args.model_source,
        api_key=args.api_key,
        max_new_tokens=args.max_tokens,
        think_budget=args.think_budget,
        image_input=False,
    )
    system_prompt = TEXT_RAPM_BASE_PROMPT
    if args.patterns:
        system_prompt += TEXT_RAPM_PATTERN_INFO
    system_prompt += RAPM_ANSWER_SUFFIX
    if args.cot:
        system_prompt += f"Explain your thought process (max {args.think_budget} tokens) inside <think>...</think> then final answer.\n"
    else:
        system_prompt += "Answer only with your final answer.\n"
    system_prompt += "State your final answer as: <answer>number</answer>\n"
    results = []
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    reasoning_traces = []

    for i, item in enumerate(tqdm(items, desc="Processing text RAPM")):
        model.init_chat(system_prompt)
        grid = item["question_grid"]
        # Format grid (None or missing bottom-right as '?')
        rows_fmt = []
        for r in range(3):
            row_cells = []
            for c in range(3):
                v = grid[r][c]
                if v is None:
                    row_cells.append("?")
                else:
                    row_cells.append(v)
            rows_fmt.append(" | ".join(row_cells))
        grid_text = "\n".join(rows_fmt)
        options = item["options"]
        opt_lines = [f"{i+1}. {o}" for i, o in enumerate(options)]
        prompt = (
            f"Matrix:\n{grid_text}\n\nOptions:\n"
            + "\n".join(opt_lines)
            + "\n\nAnswer with <answer>N</answer>."
        )
        response = model.send_message(prompt, cot=args.cot, truncate_history=True)
        answer_match = re.search(r"<answer>(.*?)</answer>", response)
        predicted = None
        if answer_match:
            ans_txt = answer_match.group(1).strip()
            nums = re.findall(r"\d+", ans_txt)
            if nums:
                try:
                    n = int(nums[0])
                    if 1 <= n <= 8:
                        predicted = n
                except ValueError:
                    pass
        correct_index = item["correct_index"]
        is_correct = predicted is not None and (predicted - 1) == correct_index
        # Category extraction (if available)
        raw = item["raw"]
        cats = (
            raw.get("credited_categories")
            or raw.get("assigned_categories")
            or [raw.get("primary_category")]
            if raw.get("primary_category")
            else []
        )
        for c in cats:
            if c:
                cat_total[c] += 1
                if is_correct:
                    cat_correct[c] += 1
        results.append(
            {
                "id": item["id"],
                "predicted_answer": predicted,
                "correct_index": correct_index,
                "is_correct": is_correct,
                "response": response,
                "categories": cats,
            }
        )
        if args.cot and model.reasoning_trace:
            reasoning_traces.append(
                {"id": item["id"], "reasoning": model.reasoning_trace[-1]}
            )

    total_correct = sum(r["is_correct"] for r in results)
    total = len(results)
    overall_accuracy = total_correct / total if total else 0
    print("\n" + "=" * 50)
    print("TEXT RAPM EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Total items: {total}")
    print(f"Overall accuracy: {overall_accuracy:.3f} ({total_correct}/{total})")
    if cat_total:
        print("\nAccuracy by category:")
        for c in sorted(cat_total.keys()):
            acc = cat_correct[c] / cat_total[c] if cat_total[c] else 0
            print(f"  {c}: {acc:.3f} ({cat_correct[c]}/{cat_total[c]})")
    summary = {
        "model": args.model,
        "mode": "text",
        "total": total,
        "correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "accuracy_by_category": {
            c: {
                "correct": cat_correct[c],
                "total": cat_total[c],
                "accuracy": cat_correct[c] / cat_total[c] if cat_total[c] else 0,
            }
            for c in cat_total
        },
        "args": vars(args),
    }
    return results, summary, model.history, reasoning_traces


def main():
    parser = argparse.ArgumentParser(
        description="Run RAPM evaluation on language models (image or text)"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--model_source",
        type=str,
        default="openrouter",
        choices=["openai", "openrouter", "vllm"],
        help="Model source",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        choices=["image", "text"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="Path to evaluation data (JSON for image, JSONL for text)",
    )
    parser.add_argument(
        "--cot", action="store_true", help="Enable chain-of-thought reasoning"
    )
    parser.add_argument(
        "--patterns",
        action="store_true",
        help="Include pattern category explanations in the system prompt",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--think_budget",
        type=int,
        default=256,
        help="Budget tokens for reasoning (when --cot)",
    )
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument(
        "--output_dir", type=str, default="rapm_data", help="Output directory"
    )
    parser.add_argument(
        "--limit_per_type",
        type=int,
        default=100,
        help="Limit per dataset type (image mode only; 0 = no limit)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    base_name = f"{args.model_source}_{args.model.replace('/', '-')}_rapm_{args.mode}"
    if args.patterns:
        base_name += "_pat"
    if args.cot:
        base_name += "_cot"
    results_path = os.path.join(args.output_dir, f"{base_name}_results.json")
    summary_path = os.path.join(args.output_dir, f"{base_name}_summary.json")
    history_path = os.path.join(args.output_dir, f"{base_name}_history.json")
    reasoning_path = os.path.join(args.output_dir, f"{base_name}_reasoning.json")

    if os.path.exists(results_path):
        print(f"Results already exist at {results_path}")
        with open(summary_path, "r") as f:
            summary = json.load(f)
        print("\nExisting Results Summary:")
        if args.mode == "image":
            print(
                f"Overall accuracy: {summary['overall_accuracy']:.3f} ({summary['total_correct']}/{summary['total_questions']})"
            )
        else:
            print(
                f"Overall accuracy: {summary['overall_accuracy']:.3f} ({summary['correct']}/{summary['total']})"
            )
        return

    if args.mode == "image":
        print("Starting image RAPM evaluation...")
        results, summary, history, reasoning_traces = run_rapm_evaluation(args)
    else:
        print("Starting text RAPM evaluation...")
        results, summary, history, reasoning_traces = run_text_rapm_evaluation(args)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    if reasoning_traces:
        with open(reasoning_path, "w") as f:
            json.dump(reasoning_traces, f, indent=2)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
