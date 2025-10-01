import csv
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_run_stats(stats_file):
    """Load run statistics from a JSON file"""
    with open(stats_file, "r") as f:
        return json.load(f)


def load_run_history(history_file):
    """Load run history from a JSON file if it exists"""
    if not history_file.exists():
        return None
    with open(history_file, "r") as f:
        return json.load(f)


TOKEN_FOUND_PATTERN = re.compile(
    r"Token\s+\S+\s+found\s+(?:in|at)\s+(?:box|coordinate)", re.IGNORECASE
)


def _extract_text_segments(payload):
    """Recursively extract text segments from history payload entries"""
    segments = []

    if isinstance(payload, str):
        segments.append(payload)
        return segments

    if isinstance(payload, dict):
        # Prioritise common keys that may contain text content
        for key in ("content", "text", "message", "summary"):
            if key in payload:
                segments.extend(_extract_text_segments(payload[key]))
        return segments

    if isinstance(payload, (list, tuple)):
        for item in payload:
            segments.extend(_extract_text_segments(item))
        return segments

    return segments


def count_tokens_found(history_entries):
    """Count how many times a token was found in a run history and return the matches"""
    if not isinstance(history_entries, list):
        return 0, []

    matches = []
    for entry in history_entries:
        if isinstance(entry, dict) and entry.get("role") != "user":
            continue
        for segment in _extract_text_segments(entry):
            matches.extend(TOKEN_FOUND_PATTERN.findall(segment))

    return len(matches), matches


def count_tokens_found_structured(structured_entries):
    """Count tokens found from structured history entries"""
    if not isinstance(structured_entries, list):
        return 0, []

    matches = []
    for entry in structured_entries:
        if isinstance(entry, dict) and entry.get("found") is True:
            matches.append(entry)

    return len(matches), matches


def _flatten_model_stats(model_stats):
    """Flatten nested metric dictionaries into a single-level mapping"""
    flat = {}
    for key, value in model_stats.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


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


def total_tokens_required(setup):
    """Calculate the total number of tokens that need to be found for a setup"""
    if not setup:
        return None

    try:
        n_boxes, n_tokens = map(int, setup.split("_"))
    except (ValueError, TypeError):
        return None

    return n_boxes * n_tokens


def analyze_results():
    source_dirs = [
        ("image", Path("./SWM/data/image")),
        ("image-text", Path("./SWM/data/image-text")),
        ("text", Path("./SWM/data/text")),
    ]
    data_sources = [(name, path) for name, path in source_dirs if path.exists()]
    if not data_sources:
        raise FileNotFoundError("Data directories not found")

    # Dictionary to store results per source, setup, and cot/non-cot
    results = {
        source: {"cot": {}, "non_cot": {}} for source, _ in data_sources
    }

    # Process all run_stats.json files grouped by source
    for source, data_dir in data_sources:
        for stats_file in data_dir.rglob("*run_stats.json"):
            if "old" in str(stats_file):
                continue

            model_name = stats_file.parent.name
            setup, is_cot = parse_setup(stats_file.stem)
            if not setup:
                continue

            total_needed_tokens = total_tokens_required(setup)
            if not total_needed_tokens:
                continue

            category = "cot" if is_cot else "non_cot"
            if setup not in results[source][category]:
                results[source][category][setup] = {}

            stats = load_run_stats(stats_file)
            history_file = stats_file.with_name(stats_file.name.replace("run_stats", "run_history"))
            history_data = load_run_history(history_file)

            structured_file = stats_file.with_name(
                stats_file.name.replace("run_stats", "run_structured_history")
            )
            structured_data = load_run_history(structured_file)

            if history_data is None and structured_data is None:
                print(
                    "Warning: no history or structured history file found for "
                    f"{stats_file.name}. Skipping."
                )
                continue

            # Gather all stats for aggregation
            scores = []
            guesses = []
            illegal = []
            invalid = []
            repeated = []
            nobox = []
            tokens_found = []
            tokens_scores = []
            scores_w_penalty = []

            for run_name, run in stats.items():
                try:
                    run_tokens_found = None
                    token_matches = []
                    history_source_name = None
                    used_structured = False

                    if structured_data and run_name in structured_data:
                        run_tokens_found, token_matches = count_tokens_found_structured(
                            structured_data[run_name]
                        )
                        history_source_name = structured_file.name
                        used_structured = True
                    elif history_data and run_name in history_data:
                        run_tokens_found, token_matches = count_tokens_found(history_data[run_name])
                        history_source_name = history_file.name
                    else:
                        missing_sources = []
                        if structured_data is not None:
                            missing_sources.append(structured_file.name)
                        if history_data is not None:
                            missing_sources.append(history_file.name)
                        missing_desc = ", ".join(missing_sources) if missing_sources else "history files"
                        print(
                            "Warning: run '"
                            f"{run_name}' not present in {missing_desc}. Skipping run."
                        )
                        continue

                    if run_tokens_found > total_needed_tokens:
                        if used_structured:
                            match_text = (
                                "\n".join(json.dumps(match, ensure_ascii=False) for match in token_matches)
                                if token_matches
                                else "<no matches captured>"
                            )
                        else:
                            match_text = "\n".join(token_matches) if token_matches else "<no matches captured>"

                        error_msg = (
                            "Token detection error: found more tokens than expected.\n"
                            f"  Source stats file: {stats_file}\n"
                            f"  Source history file: {history_file}\n"
                        )
                        if structured_data is not None:
                            error_msg += f"  Source structured history file: {structured_file}\n"
                        error_msg += (
                            f"  Run: {run_name}\n"
                            f"  Tokens required: {total_needed_tokens}\n"
                            f"  Tokens detected: {run_tokens_found}\n"
                            "  Matched text segments:\n"
                            f"{match_text}"
                        )
                        raise RuntimeError(error_msg)

                    tokens_found.append(run_tokens_found)

                    tokens_score = (
                        run_tokens_found / total_needed_tokens if total_needed_tokens else 0.0
                    )
                    tokens_score = min(1.0, tokens_score)
                    tokens_scores.append(tokens_score)

                    guesses_value = run.get("guesses", 0)
                    worst_case = run.get("worst_case_guesses", guesses_value)

                    if guesses_value != worst_case:
                        finished_run = run.get("finished_run")
                        tokens_incomplete = (run_tokens_found is not None) and (tokens_score < 1.0)
                        if (finished_run is False) or tokens_incomplete:
                            diff = worst_case - guesses_value
                            if diff > 0:
                                run["invalid"] = run.get("invalid", 0) + diff
                                run["guesses"] = worst_case
                                guesses_value = worst_case

                    score = calculate_score(run)
                    scores.append(score)
                    scores_w_penalty.append(score * tokens_score)
                    guesses.append(run.get("guesses", guesses_value))
                    illegal.append(run.get("illegal", 0))
                    invalid.append(run.get("invalid", 0))
                    repeated.append(run.get("repeated", 0))
                    nobox.append(run.get("nobox", 0))
                except (TypeError, KeyError) as e:
                    print(f"Error processing {stats_file}: {e}")
                    continue

            if scores:  # Only add results if we have valid scores
                avg_score = float(np.mean(scores))
                std_score = float(np.std(scores))
                max_score = float(np.max(scores))
                min_score = float(np.min(scores))

                avg_tokens_score = float(np.mean(tokens_scores)) if tokens_scores else 0.0
                std_tokens_score = float(np.std(tokens_scores)) if tokens_scores else 0.0
                max_tokens_score = float(np.max(tokens_scores)) if tokens_scores else 0.0
                min_tokens_score = float(np.min(tokens_scores)) if tokens_scores else 0.0

                avg_score_w_penalty = (
                    float(np.mean(scores_w_penalty)) if scores_w_penalty else 0.0
                )
                std_score_w_penalty = (
                    float(np.std(scores_w_penalty)) if scores_w_penalty else 0.0
                )
                max_score_w_penalty = (
                    float(np.max(scores_w_penalty)) if scores_w_penalty else 0.0
                )
                min_score_w_penalty = (
                    float(np.min(scores_w_penalty)) if scores_w_penalty else 0.0
                )

                results[source][category][setup][model_name] = {
                    "avg_score": avg_score,
                    "std_score": std_score,
                    "max_score": max_score,
                    "min_score": min_score,
                    "avg_tokens_score": avg_tokens_score,
                    "std_tokens_score": std_tokens_score,
                    "max_tokens_score": max_tokens_score,
                    "min_tokens_score": min_tokens_score,
                    "avg_score_w_penalty": avg_score_w_penalty,
                    "std_score_w_penalty": std_score_w_penalty,
                    "max_score_w_penalty": max_score_w_penalty,
                    "min_score_w_penalty": min_score_w_penalty,
                    "n_runs": len(scores),
                    "tokens_found": {
                        "avg": float(np.mean(tokens_found)) if tokens_found else 0.0,
                        "std": float(np.std(tokens_found)) if tokens_found else 0.0,
                        "min": int(np.min(tokens_found)) if tokens_found else 0,
                        "max": int(np.max(tokens_found)) if tokens_found else 0,
                    },
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
                    "total_tokens_required": total_needed_tokens,
                }

    # Generate plots for each setup
    for source, source_results in results.items():
        setups = set()
        for category in source_results.values():
            setups.update(category.keys())

        for setup in sorted(setups):
            plt.figure(figsize=(12, 6))

            all_models = set()
            for category in ["non_cot", "cot"]:
                if setup in source_results[category]:
                    all_models.update(source_results[category][setup].keys())
            all_models = sorted(list(all_models))

            x = np.arange(len(all_models))
            width = 0.35

            for i, category in enumerate(["non_cot", "cot"]):
                scores = []
                min_scores = []
                max_scores = []
                for model in all_models:
                    if setup in source_results[category] and model in source_results[category][setup]:
                        model_data = source_results[category][setup][model]
                        scores.append(model_data["avg_score"])
                        min_scores.append(model_data["min_score"])
                        max_scores.append(model_data["max_score"])
                    else:
                        scores.append(0)
                        min_scores.append(0)
                        max_scores.append(0)

                yerr = np.array(
                    [
                        [s - min_s for s, min_s in zip(scores, min_scores)],
                        [max_s - s for s, max_s in zip(scores, max_scores)],
                    ]
                )

                bars = plt.bar(
                    x + (i - 0.5) * width,
                    scores,
                    width,
                    yerr=yerr,
                    label=category.replace("_", " ").title(),
                )

                for idx, rect in enumerate(bars):
                    height = rect.get_height()
                    if height > 0:
                        plt.text(
                            rect.get_x() + rect.get_width() / 2.0,
                            height,
                            f"{scores[idx]:.2f}",
                            ha="center",
                            va="bottom",
                        )

            plt.xlabel("Models")
            plt.ylabel("Score")
            plt.title(f"{source.title()} - Setup {setup} (Boxes_Tokens)")
            plt.xticks(x, all_models, rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()

            plots_dir = Path("./data/plots") / source
            plots_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plots_dir / f"analysis_{source}_setup_{setup}.png")
            plt.close()

    # Save CSV summaries per source
    csv_root = Path("./data/csv")
    csv_root.mkdir(parents=True, exist_ok=True)
    for source, source_results in results.items():
        rows = []
        for category in ["non_cot", "cot"]:
            setups = source_results.get(category, {})
            for setup, models in setups.items():
                for model_name, model_stats in models.items():
                    row = {
                        "source": source,
                        "category": category,
                        "setup": setup,
                        "model": model_name,
                    }
                    row.update(_flatten_model_stats(model_stats))
                    rows.append(row)

        if not rows:
            continue

        fieldnames = ["source", "category", "setup", "model"]
        extra_fields = sorted(
            {key for row in rows for key in row.keys() if key not in fieldnames}
        )
        fieldnames.extend(extra_fields)

        csv_path = csv_root / f"swm_summary_{source}.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    # Save summary statistics
    with open("data/swm_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    results = analyze_results()
    print("\nAnalysis Results:")
    for source in results:
        print(f"\n=== SOURCE: {source.upper()} ===")
        for category in ["non_cot", "cot"]:
            if not results[source][category]:
                continue
            print(f"\n{category.upper()}:")
            for setup, models in results[source][category].items():
                print(f"\nSetup {setup}:")
                for model, stats in models.items():
                    print(f"  {model}:")
                    for metric, value in stats.items():
                        if isinstance(value, float):
                            print(f"    {metric}: {value:.4f}")
                        else:
                            print(f"    {metric}: {value}")
