import os
import json
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
    elif "card" in filename:
        return "card"
    elif "string" in filename:
        return "string"
    return None


def analyze_results():
    data_dir = Path("./wcst_data/text")
    if not data_dir.exists():
        raise FileNotFoundError("WCST data directory not found")

    # Dictionary to store results by setup type and category
    results = {
        "card": {"non_cot": {}, "cot": {}, "few_shot": {}, "few_shot_cot": {}},
        "card-random": {"non_cot": {}, "cot": {}, "few_shot": {}, "few_shot_cot": {}},
        "string": {"non_cot": {}, "cot": {}, "few_shot": {}, "few_shot_cot": {}},
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

            is_cot = "_cot" in stats_file.stem
            is_few_shot = "_few_shot" in stats_file.stem

            if is_few_shot and is_cot:
                category = "few_shot_cot"
            elif is_few_shot:
                category = "few_shot"
            elif is_cot:
                category = "cot"
            else:
                category = "non_cot"

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
        for category in results[setup_type]:
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        all_models = set()
        for category in results[setup_type].values():
            all_models.update(category.keys())
        all_models = sorted(list(all_models))

        x = np.arange(len(all_models))
        width = 0.2

        # Plot accuracy (top subplot)
        for i, category in enumerate(["non_cot", "cot", "few_shot", "few_shot_cot"]):
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

            bars = ax1.bar(
                x + (i - 1.5) * width,
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
        for i, category in enumerate(["non_cot", "cot", "few_shot", "few_shot_cot"]):
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

            bars = ax2.bar(
                x + (i - 1.5) * width,
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

        plots_dir = Path("./wcst_data/plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(
            f"wcst_data/plots/wcst_analysis_{setup_type}.png", bbox_inches="tight"
        )
        plt.close()

    # Save summary statistics
    with open("wcst_data/wcst_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    results = analyze_results()
    # Print summary
    print("\nWCST Analysis Results:")
    for setup_type in ["card"]:
        print(f"\n{setup_type.upper()}:")
        for category in ["non_cot", "cot"]:
            print(f"\n{category.upper()}:")
            for model, stats in results[setup_type][category].items():
                print(f"\n{model}:")
                for metric, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
