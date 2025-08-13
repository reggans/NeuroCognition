#!/usr/bin/env python3
"""
RAPM (Raven's Progressive Matrices) Evaluation Script

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

RAPM_SYSTEM_PROMPT = """You are taking the Raven's Progressive Matrices (RPM) test, a non-verbal intelligence test that measures abstract reasoning ability.

You will see a 3x3 matrix of images with the bottom-right image missing (shown as a question mark), followed by 8 answer choices numbered 1-8.

Your task is to:
1. Analyze the patterns in the rows and columns of the 3x3 matrix
2. Determine the logical rule or pattern that governs the progression
3. Select which of the 8 answer choices correctly completes the matrix

The patterns can involve:
- Shape transformations (rotation, reflection, scaling)
- Position changes (movement, arrangement)
- Attribute changes (color, size, number of elements)
- Logical operations (addition, subtraction, intersection)
- Sequence progressions (systematic changes across rows/columns)

Look for patterns both horizontally (across rows) and vertically (down columns). The missing piece should follow the same logical rule that applies to the rest of the matrix.

Your final answer should be a number between 1-8 corresponding to the answer choice you think is correct.
"""


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


def run_rapm_evaluation(args):
    """Run the RAPM evaluation."""

    # Load evaluation data with limit per type
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

    # Initialize system prompt
    system_prompt = RAPM_SYSTEM_PROMPT
    if args.cot:
        system_prompt += f"\nExplain your thought process and reasoning in maximum {args.think_budget} tokens wrapped with <think> and </think>. After the closing </think> tag, provide a brief summary of your reasoning and your final answer.\n"
    else:
        system_prompt += "\nAnswer only with your final answer.\n"
    system_prompt += (
        'State your final answer using the template: "<answer>your answer</answer>"\n'
    )

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
            # Try to extract number
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


def main():
    parser = argparse.ArgumentParser(
        description="Run RAPM evaluation on language models"
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
        "--eval_data", type=str, required=True, help="Path to evaluation data JSON file"
    )
    parser.add_argument(
        "--cot", action="store_true", help="Enable chain-of-thought reasoning"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--think_budget", type=int, default=256, help="Budget tokens for reasoning"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (if not set, uses environment variable)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rapm_data",
        help="Output directory for results",
    )
    parser.add_argument(
        "--limit_per_type",
        type=int,
        default=100,
        help="Limit number of questions per dataset type (default: 100, set to 0 for no limit)",
    )

    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate output filename
    model_name_safe = args.model.replace("/", "-")
    base_filename = f"{args.model_source}_{model_name_safe}_rapm"
    if args.limit_per_type > 0:
        base_filename += f"_limit{args.limit_per_type}"
    if args.cot:
        base_filename += "_cot"

    results_path = os.path.join(args.output_dir, f"{base_filename}_results.json")
    summary_path = os.path.join(args.output_dir, f"{base_filename}_summary.json")
    history_path = os.path.join(args.output_dir, f"{base_filename}_history.json")
    reasoning_path = os.path.join(args.output_dir, f"{base_filename}_reasoning.json")

    # Check if results already exist
    if os.path.exists(results_path):
        print(f"Results already exist at {results_path}")
        with open(summary_path, "r") as f:
            summary = json.load(f)

        print("\nExisting Results Summary:")
        print(f"Model: {summary['model']}")
        print(
            f"Overall accuracy: {summary['overall_accuracy']:.3f} ({summary['total_correct']}/{summary['total_questions']})"
        )
        print("\nAccuracy by dataset type:")
        for dataset_type, stats in summary["accuracy_by_type"].items():
            print(
                f"  {dataset_type}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})"
            )
        return

    print(f"Starting RAPM evaluation...")
    print(f"Model: {args.model}")
    print(f"Model source: {args.model_source}")
    print(f"CoT enabled: {args.cot}")
    print(f"Output directory: {args.output_dir}")

    # Run evaluation
    results, summary, history, reasoning_traces = run_rapm_evaluation(args)

    # Save results
    print(f"\nSaving results to {args.output_dir}")

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
