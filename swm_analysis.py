import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_run_stats(stats_file):
    """Load run statistics from a JSON file"""
    with open(stats_file, "r") as f:
        return json.load(f)


def calculate_score(stats):
    """Calculate score for a single run's statistics"""
    # Treat 'nobox' the same as 'illegal' and 'repeated' (penalized moves)
    illegal_moves = stats.get("illegal", 0) + stats.get("repeated", 0) + stats.get("nobox", 0)
    denom = stats["guesses"] - stats["invalid"]
    if denom <= 0:
        # Avoid division by zero; define worst-case score
        return 0.0
    return 1 - illegal_moves / denom


def parse_setup(filename):
    """Parse setup information from filename"""
    parts = filename.split("_")
    for i, part in enumerate(parts):
        if part.isdigit():
            n_boxes = int(part)
            n_tokens = int(parts[i + 1])
            is_cot = "_cot_" in filename
            return f"{n_boxes}_{n_tokens}", is_cot
    return None, None


def analyze_results():
    data_dir = Path("./SWM/data/image")
    if not data_dir.exists():
        raise FileNotFoundError("Data directory not found")

    # Dictionary to store results by setup and cot/non-cot
    results = {"cot": {}, "non_cot": {}}

    # Process all run_stats.json files
    for stats_file in data_dir.rglob("*run_stats.json"):
        if "old" in str(stats_file):
            continue

        model_name = stats_file.parent.name
        setup, is_cot = parse_setup(stats_file.stem)
        if not setup:
            continue

        category = "cot" if is_cot else "non_cot"
        if setup not in results[category]:
            results[category][setup] = {}

        stats = load_run_stats(stats_file)
        # Gather all stats for aggregation
        scores = []
        guesses = []
        illegal = []
        invalid = []
        repeated = []
        nobox = []
        for run in stats.values():  # each run is a dictionary
            try:
                score = calculate_score(run)
                scores.append(score)
                guesses.append(run.get("guesses", 0))
                illegal.append(run.get("illegal", 0))
                invalid.append(run.get("invalid", 0))
                repeated.append(run.get("repeated", 0))
                nobox.append(run.get("nobox", 0))
            except (TypeError, KeyError) as e:
                print(f"Error processing {stats_file}: {e}")
                continue

        if scores:  # Only add results if we have valid scores
            results[category][setup][model_name] = {
                "avg_score": np.mean(scores),
                "std_score": np.std(scores),
                "max_score": np.max(scores),
                "min_score": np.min(scores),
                "n_runs": len(scores),
                # Add aggregated stats for guesses, illegal, invalid, repeated
                "guesses": {
                    "avg": float(np.mean(guesses)),
                    "std": float(np.std(guesses)),
                    "min": int(np.min(guesses)),
                    "max": int(np.max(guesses)),
                },
                "illegal": {
                    "avg": float(np.mean(illegal)),
                    "std": float(np.std(illegal)),
                    "min": int(np.min(illegal)),
                    "max": int(np.max(illegal)),
                },
                "invalid": {
                    "avg": float(np.mean(invalid)),
                    "std": float(np.std(invalid)),
                    "min": int(np.min(invalid)),
                    "max": int(np.max(invalid)),
                },
                "repeated": {
                    "avg": float(np.mean(repeated)),
                    "std": float(np.std(repeated)),
                    "min": int(np.min(repeated)),
                    "max": int(np.max(repeated)),
                },
                "nobox": {
                    "avg": float(np.mean(nobox)) if nobox else 0.0,
                    "std": float(np.std(nobox)) if nobox else 0.0,
                    "min": int(np.min(nobox)) if nobox else 0,
                    "max": int(np.max(nobox)) if nobox else 0,
                },
            }

    # Generate plots for each setup
    setups = set()
    for category in results.values():
        setups.update(category.keys())

    for setup in sorted(setups):
        plt.figure(figsize=(12, 6))

        all_models = set()
        for category in ["non_cot", "cot"]:
            if setup in results[category]:
                all_models.update(results[category][setup].keys())
        all_models = sorted(list(all_models))

        x = np.arange(len(all_models))
        width = 0.35

        for i, category in enumerate(["non_cot", "cot"]):
            scores = []
            min_scores = []
            max_scores = []
            for model in all_models:
                if setup in results[category] and model in results[category][setup]:
                    # Use the results we already have instead of searching for files again
                    model_data = results[category][setup][model]
                    scores.append(model_data["avg_score"])
                    min_scores.append(model_data["min_score"])
                    max_scores.append(model_data["max_score"])
                else:
                    scores.append(0)
                    min_scores.append(0)
                    max_scores.append(0)

            # Calculate asymmetric error bars
            yerr = np.array(
                [
                    [s - min_s for s, min_s in zip(scores, min_scores)],  # lower errors
                    [max_s - s for s, max_s in zip(scores, max_scores)],  # upper errors
                ]
            )

            bars = plt.bar(
                x + (i - 0.5) * width,
                scores,
                width,
                yerr=yerr,
                label=category.replace("_", " ").title(),
            )

            # Add score labels on top of bars
            for idx, rect in enumerate(bars):
                height = rect.get_height()
                if height > 0:  # Only add label if bar exists
                    plt.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        height,
                        f"{scores[idx]:.2f}",
                        ha="center",
                        va="bottom",
                    )

        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.title(f"Setup {setup} (Boxes_Tokens) - Average Scores by Model")
        plt.xticks(x, all_models, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        plots_dir = Path("./data/plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(f"data/plots/analysis_setup_{setup}.png")
        plt.close()

    # Save summary statistics
    with open("data/swm_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    results = analyze_results()
    print("\nAnalysis Results:")
    for category in ["non_cot", "cot"]:
        print(f"\n{category.upper()}:")
        for setup, models in results[category].items():
            print(f"\nSetup {setup}:")
            for model, stats in models.items():
                print(f"  {model}:")
                for metric, value in stats.items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value}")
