import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Directories relative to this script
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "SWM" / "data"
TARGET_SOURCES = ("image", "image-text")
OUTPUT_PATH = DATA_ROOT / "swm_image_errors_metrics.json"


TOKEN_FOUND_PATTERN = re.compile(
    r"Token\s+\S+\s+found\s+(?:in|at)\s+(?:box|coordinate)", re.IGNORECASE
)


def _normalize_selection(value):
    """Normalize various coordinate/box representations into a comparable form."""
    if isinstance(value, dict):
        if {"row", "col"}.issubset(value.keys()):
            return (value["row"], value["col"])
        if {"x", "y"}.issubset(value.keys()):
            return (value["x"], value["y"])
        return tuple((k, _normalize_selection(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        if len(value) == 1:
            return _normalize_selection(value[0])
        if all(isinstance(v, (int, float, str)) for v in value):
            return tuple(value)
        return tuple(_normalize_selection(v) for v in value)
    return value


def _extract_text_segments(payload):
    """Recursively extract text segments from history payload entries."""
    segments: List[str] = []

    if isinstance(payload, str):
        segments.append(payload)
        return segments

    if isinstance(payload, dict):
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
    """Count how many times a token was found in unstructured history."""
    if not isinstance(history_entries, list):
        return 0, []

    matches: List[str] = []
    for entry in history_entries:
        if isinstance(entry, dict) and entry.get("role") != "user":
            continue
        for segment in _extract_text_segments(entry):
            matches.extend(TOKEN_FOUND_PATTERN.findall(segment))

    return len(matches), matches


def count_tokens_found_structured(structured_entries):
    """Count tokens found from structured history entries."""
    if not isinstance(structured_entries, list):
        return 0, []

    total_found = 0
    matches = []

    for entry in structured_entries:
        if not isinstance(entry, dict) or entry.get("found") is not True:
            continue

        chosen = None
        if entry.get("chosen_box") is not None:
            chosen = _normalize_selection(entry["chosen_box"])
        elif entry.get("chosen_coord") is not None:
            chosen = _normalize_selection(entry["chosen_coord"])
        token_box = entry.get("token_box")

        tokens_found_here = 0
        if chosen is not None and isinstance(token_box, list):
            for coord in token_box:
                if _normalize_selection(coord) == chosen:
                    tokens_found_here += 1

        if tokens_found_here <= 0:
            tokens_found_here = 1

        total_found += tokens_found_here
        matches.append(
            {
                "chosen_box": entry.get("chosen_box"),
                "chosen_coord": entry.get("chosen_coord"),
                "token_box": token_box,
                "tokens_found_here": tokens_found_here,
            }
        )

    return total_found, matches


def load_run_stats(stats_file: Path):
    with open(stats_file, "r") as handle:
        return json.load(handle)


def load_run_history(history_file: Path):
    if not history_file.exists():
        return None
    with open(history_file, "r") as handle:
        return json.load(handle)


def parse_setup(filename: str) -> Tuple[str, bool]:
    parts = filename.split("_")
    for idx, part in enumerate(parts):
        if part.isdigit() and idx + 1 < len(parts):
            try:
                int(parts[idx + 1])
            except ValueError:
                continue
            n_boxes = int(part)
            n_tokens = int(parts[idx + 1])
            is_cot = "_cot_" in filename
            return f"{n_boxes}_{n_tokens}", is_cot
    return None, False


def total_tokens_required(setup: str) -> int:
    if not setup:
        return 0
    try:
        n_boxes, n_tokens = map(int, setup.split("_"))
    except (TypeError, ValueError):
        return 0
    return n_boxes * n_tokens


def compute_metric_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    array = np.array(values, dtype=float)
    return {
        "avg": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_error_metrics():
    metrics_by_model: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "guesses": [],
            "illegal": [],
            "invalid": [],
            "repeated": [],
            "nobox": [],
        }
    )

    total_runs: Dict[str, int] = defaultdict(int)

    for source in TARGET_SOURCES:
        data_dir = DATA_ROOT / source
        if not data_dir.exists():
            continue

        for stats_file in data_dir.rglob("*run_stats.json"):
            if "old" in str(stats_file):
                continue
            if "_notes" in stats_file.stem:
                continue

            model_name = stats_file.parent.name
            setup, _ = parse_setup(stats_file.stem)
            total_needed = total_tokens_required(setup)
            if total_needed <= 0:
                continue

            stats = load_run_stats(stats_file)
            history_file = stats_file.with_name(stats_file.name.replace("run_stats", "run_history"))
            history_data = load_run_history(history_file)
            structured_file = stats_file.with_name(
                stats_file.name.replace("run_stats", "run_structured_history")
            )
            structured_data = load_run_history(structured_file)

            if history_data is None and structured_data is None:
                # Skip files with no history information; we cannot validate completeness.
                continue

            for run_name, run in stats.items():
                if not isinstance(run, dict):
                    continue

                try:
                    run_tokens_found = None
                    if structured_data and run_name in structured_data:
                        run_tokens_found, _ = count_tokens_found_structured(structured_data[run_name])
                    elif history_data and run_name in history_data:
                        run_tokens_found, _ = count_tokens_found(history_data[run_name])

                    guesses_value = run.get("guesses", 0)
                    worst_case = run.get("worst_case_guesses", guesses_value)
                    invalid_value = run.get("invalid", 0)

                    if guesses_value != worst_case:
                        finished_run = run.get("finished_run")
                        tokens_incomplete = (
                            run_tokens_found is not None
                            and total_needed > 0
                            and run_tokens_found < total_needed
                        )
                        if (finished_run is False) or tokens_incomplete:
                            diff = worst_case - guesses_value
                            if diff > 0:
                                guesses_value = worst_case
                                invalid_value += diff

                    metrics = metrics_by_model[model_name]
                    metrics["guesses"].append(float(guesses_value))
                    metrics["illegal"].append(float(run.get("illegal", 0)))
                    metrics["invalid"].append(float(invalid_value))
                    metrics["repeated"].append(float(run.get("repeated", 0)))
                    metrics["nobox"].append(float(run.get("nobox", 0)))
                    total_runs[model_name] += 1
                except (TypeError, ValueError):
                    continue

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name in sorted(metrics_by_model.keys()):
        model_metrics = metrics_by_model[model_name]
        model_result = {
            metric: compute_metric_summary(values)
            for metric, values in model_metrics.items()
        }
        model_result["n_runs"] = total_runs.get(model_name, 0)
        results[model_name] = model_result

    return results


def main():
    results = aggregate_error_metrics()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as handle:
        json.dump(results, handle, indent=4)

    print(f"Saved error metrics summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
