import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_run_stats(stats_file):
    """Load WCST statistics from a JSON file"""
    with open(stats_file, "r") as f:
        return json.load(f)


def complete_score(data):
    scores = {
        "accuracy": [],
        "perserverative_error": [],
        "cat_complete": [],
        "first_cat_trials": [],
        "failure_set": [],
    }

    for trial in data:
        correct_rule = ""
        perserverated_response = -1
        first_complete = False
        correct_run = 0
        conceptual_response = False

        total_complete = 0
        total_perseverated = 0
        total_fms = 0
        total_correct = 0
        first_cat_trial_idx = None

        for i, query in enumerate(trial):
            if correct_rule != query["rule"] and correct_rule != "":
                perserverated_response = -1
                correct_run = 0
                conceptual_response = False

                if not first_complete:
                    first_complete = True
                    first_cat_trial_idx = i + 1

            # Always update correct_rule to current query's rule
            correct_rule = query["rule"]

            if query["correct"]:
                total_correct += 1
                correct_run += 1

                if correct_run >= 3:
                    conceptual_response = True
                if correct_run >= 5:
                    total_complete += 1
            else:
                correct_run = 0

                if conceptual_response:
                    total_fms += 1

                try:
                    ans = int(query["model_ans"])

                    if ans == perserverated_response:
                        total_perseverated += 1
                    perserverated_response = ans
                except:
                    continue

        n_query = len(trial)
        scores["accuracy"].append(total_correct / n_query)
        scores["perserverative_error"].append(total_perseverated / n_query)
        scores["cat_complete"].append(total_complete)
        scores["failure_set"].append(total_fms / n_query)
        # Always append a value for first_cat_trials (use n_query+1 if never completed)
        if first_cat_trial_idx is not None:
            scores["first_cat_trials"].append(first_cat_trial_idx)
        else:
            scores["first_cat_trials"].append(n_query + 1)

    for metric in scores:
        scores[metric] = np.mean(scores[metric])  # type: ignore

    return scores


def get_setup_type(filename):
    """Get the setup type from filename"""
    if "card-random" in filename:
        return "card-random"
    elif "string" in filename:
        return "string"
    else:
        return "card"


def analyze_results(data_type: str = "image"):
    """Analyze WCST results.

    Parameters
    ----------
    data_type : str
        Either 'image' or 'text'. Determines which subdirectory under WCST/data/ to read.
    """
    if data_type not in {"image", "text"}:
        raise ValueError("data_type must be 'image' or 'text'")

    data_dir = Path(f"./WCST/data/{data_type}")
    if not data_dir.exists():
        raise FileNotFoundError(f"WCST data directory not found: {data_dir}")

    # Dictionary to store results by setup type and category
    # Base categories always tracked; additional dynamic categories will be added on demand
    base_categories = ["non_cot", "cot", "few_shot", "few_shot_cot"]
    results = {
        "card": {c: {} for c in base_categories},
        "card-random": {c: {} for c in base_categories},
        "string": {c: {} for c in base_categories},
    }

    # Process files
    for model_dir in data_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == "old":
            continue

        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        for stats_file in model_dir.glob("*.json"):
            if (
                "old" in str(stats_file)
                or "history" in str(stats_file)
                or "reasoning" in str(stats_file)
            ):
                continue

            setup_type = get_setup_type(stats_file.stem)
            if not setup_type:
                continue

            stem = stats_file.stem
            is_cot = "_cot" in stem or stem.endswith("cot")
            is_few_shot = "_few_shot" in stem

            # Detect background mode variants (appear after e.g. image_ )
            background_mode = None
            # We look for patterns _bg_first, _off, _rest OR image_bg_first / image_off / image_rest
            if any(key in stem for key in ["bg_first", "_off", "_rest"]):
                if "bg_first" in stem:
                    background_mode = "bg_first"
                elif "_off" in stem:
                    background_mode = "off"
                elif "_rest" in stem:
                    background_mode = "rest"

            if is_few_shot and is_cot:
                if background_mode:
                    category = f"few_shot_cot_{background_mode}"
                else:
                    category = "few_shot_cot"
            elif is_few_shot:
                # Keep existing behaviour (no background variant requested by user for plain few_shot)
                category = "few_shot"
            elif is_cot:
                if background_mode:
                    category = f"cot_{background_mode}"
                else:
                    category = "cot"
            else:
                category = "non_cot"

            # Dynamically add new category containers if needed
            if category not in results[setup_type]:
                results[setup_type][category] = {}

            print(f"  Processing {setup_type} file: {stats_file.name}")

            try:
                stats = load_run_stats(stats_file)
                if not isinstance(stats, dict):
                    continue

                # Use the new complete_score function for each run
                metric_lists = {
                    "accuracy": [],
                    "perserverative_error": [],
                    "cat_complete": [],
                    "first_cat_trials": [],
                    "failure_set": [],
                }
                for run_id, run_data in stats.items():
                    try:
                        run_metrics = complete_score([run_data])
                        for k in metric_lists:
                            metric_lists[k].append(run_metrics[k])
                    except Exception as e:
                        print(f"  Error processing run {run_id}: {e}")

                # Only add if we have at least one run
                if any(len(metric_lists[k]) > 0 for k in metric_lists):
                    if model_name not in results[setup_type][category]:
                        results[setup_type][category][model_name] = {
                            "accuracy": [],
                            "perserverative_error": [],
                            "cat_complete": [],
                            "first_cat_trials": [],
                            "failure_set": [],
                            "n_files": 0,
                        }
                    for k in metric_lists:
                        results[setup_type][category][model_name][k].extend(
                            metric_lists[k]
                        )
                    results[setup_type][category][model_name]["n_files"] += 1

            except Exception as e:
                print(f"  Error processing {stats_file.name}: {e}")
                continue

    # Calculate statistics and generate plots for each setup type
    for setup_type in results:
        # Calculate final statistics for each metric
        for category in list(results[setup_type].keys()):
            for model_name in list(results[setup_type][category].keys()):
                model_data = results[setup_type][category][model_name]
                metrics = [k for k in model_data if k != "n_files"]
                stats_dict = {}
                n_runs = len(model_data[metrics[0]]) if metrics else 0
                for metric in metrics:
                    arr = np.array(model_data[metric])
                    if arr.size > 0:
                        stats_dict[f"avg_{metric}"] = np.mean(arr)
                        stats_dict[f"std_{metric}"] = np.std(arr)
                        stats_dict[f"max_{metric}"] = np.max(arr)
                        stats_dict[f"min_{metric}"] = np.min(arr)
                    else:
                        stats_dict[f"avg_{metric}"] = 0
                        stats_dict[f"std_{metric}"] = 0
                        stats_dict[f"max_{metric}"] = 0
                        stats_dict[f"min_{metric}"] = 0
                stats_dict["n_runs"] = n_runs
                stats_dict["n_files"] = model_data["n_files"]
                if n_runs > 0:
                    results[setup_type][category][model_name] = stats_dict
                else:
                    del results[setup_type][category][model_name]

        # Create figure with 2 subplots: accuracy and perserverative_error
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        all_models = set()
        for category_dict in results[setup_type].values():
            all_models.update(category_dict.keys())
        all_models = sorted(list(all_models))

        # Determine categories to plot: keep base order, then any dynamic extensions sorted
        dynamic_categories = [c for c in results[setup_type].keys() if c not in base_categories]
        # Stable sort for reproducibility
        dynamic_categories = sorted(dynamic_categories)
        categories_to_plot = base_categories + dynamic_categories

        x = np.arange(len(all_models))
        n_cat = len(categories_to_plot)
        width = min(0.8 / max(n_cat, 1), 0.18)  # keep bars from overlapping

        # Plot accuracy (top subplot)
        for i, category in enumerate(categories_to_plot):
            accuracies = []
            min_accuracies = []
            max_accuracies = []
            for model in all_models:
                if model in results[setup_type][category]:
                    model_data = results[setup_type][category][model]
                    accuracies.append(model_data.get("avg_accuracy", 0))
                    min_accuracies.append(model_data.get("min_accuracy", 0))
                    max_accuracies.append(model_data.get("max_accuracy", 0))
                else:
                    accuracies.append(0)
                    min_accuracies.append(0)
                    max_accuracies.append(0)

            yerr = np.array(
                [
                    [s - min_s for s, min_s in zip(accuracies, min_accuracies)],
                    [max_s - s for s, max_s in zip(accuracies, max_accuracies)],
                ]
            )

            offset = (i - (n_cat - 1) / 2) * width
            bars = ax1.bar(
                x + offset,
                accuracies,
                width,
                yerr=yerr,
                label=category.replace("_", " ").title(),
            )

            for idx, rect in enumerate(bars):
                height = rect.get_height()
                if height > 0:
                    ax1.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        height,
                        f"{accuracies[idx]:.2f}",
                        ha="center",
                        va="bottom",
                    )

        ax1.set_xlabel("Models")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(
            f"WCST {setup_type.title()} - Average Accuracy by Model and Category"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_models, rotation=45, ha="right")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        # Plot perserverative_error (bottom subplot)
        for i, category in enumerate(categories_to_plot):
            errors = []
            min_errors = []
            max_errors = []
            for model in all_models:
                if model in results[setup_type][category]:
                    model_data = results[setup_type][category][model]
                    errors.append(model_data.get("avg_perserverative_error", 0))
                    min_errors.append(model_data.get("min_perserverative_error", 0))
                    max_errors.append(model_data.get("max_perserverative_error", 0))
                else:
                    errors.append(0)
                    min_errors.append(0)
                    max_errors.append(0)

            yerr = np.array(
                [
                    [s - min_s for s, min_s in zip(errors, min_errors)],
                    [max_s - s for s, max_s in zip(errors, max_errors)],
                ]
            )

            offset = (i - (n_cat - 1) / 2) * width
            bars = ax2.bar(
                x + offset,
                errors,
                width,
                yerr=yerr,
                label=category.replace("_", " ").title(),
            )

            for idx, rect in enumerate(bars):
                height = rect.get_height()
                if height > 0:
                    ax2.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        height,
                        f"{errors[idx]:.2f}",
                        ha="center",
                        va="bottom",
                    )

        ax2.set_xlabel("Models")
        ax2.set_ylabel("Perserverative Error")
        ax2.set_title(
            f"WCST {setup_type.title()} - Average Perserverative Error by Model and Category"
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_models, rotation=45, ha="right")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        plots_dir = Path("./WCST/data/plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(
            f"WCST/data/plots/wcst_analysis_{setup_type}.png", bbox_inches="tight"
        )
        plt.close()

    # Save summary statistics
    with open("WCST/data/wcst_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze WCST results.")
    parser.add_argument(
        "--data-type",
        dest="data_type",
        choices=["image", "text"],
        default="image",
        help="Dataset type directory under WCST/data/ (default: image)",
    )
    args = parser.parse_args()
    results = analyze_results(args.data_type)
    # Print compact summary for all categories present (limit to card setup for brevity)
    print("\nWCST Analysis Results (setup=card):")
    for category in results["card"].keys():
        print(f"\nCategory: {category}")
        for model, stats in results["card"][category].items():
            print(f"  Model: {model}")
            for metric, value in stats.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
