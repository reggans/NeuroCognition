import json
import statistics
from pathlib import Path


def summarize_model_stats(models_dict):
    out = {}
    for model, stats in models_dict.items():
        # stats may already be aggregated (avg/std) or per-run lists depending on source
        # We expect stats to contain numeric fields like 'guesses','illegal','invalid','repeated','nobox'
        def get_vals(key):
            v = stats.get(key)
            if v is None:
                return []
            # If already aggregated dict with avg/std/min/max, return single-value list of avg
            if isinstance(v, dict) and "avg" in v:
                return [v["avg"]]
            # If it's a single number
            if isinstance(v, (int, float)):
                return [v]
            return list(v)

        guesses = get_vals("guesses")
        illegal = get_vals("illegal")
        invalid = get_vals("invalid")
        repeated = get_vals("repeated")
        nobox = get_vals("nobox")

        def stats_of(lst):
            if not lst:
                return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            avg = float(statistics.mean(lst))
            try:
                std = float(statistics.pstdev(lst))
            except Exception:
                std = 0.0
            mn = float(min(lst))
            mx = float(max(lst))
            return {"avg": avg, "std": std, "min": mn, "max": mx}

        out[model] = {
            "guesses": stats_of(guesses),
            "illegal": stats_of(illegal),
            "invalid": stats_of(invalid),
            "repeated": stats_of(repeated),
            "nobox": stats_of(nobox),
            "n_runs": (
                stats.get("n_runs", 1)
                if isinstance(stats.get("n_runs", 1), int)
                else int(stats.get("n_runs", 1))
            ),
        }

    return out


def main():
    summary_path = Path("SWM/data/swm_summary.json")
    if not summary_path.exists():
        print("SWM/data/swm_summary.json not found. Run swm_analysis.py first.")
        return

    data = json.loads(summary_path.read_text())

    # We'll aggregate across both 'image' and 'image-text' modalities
    modalities = ["image", "image-text"]
    accum = {}

    # Only include categories that do NOT contain notes (exclude '*-notes')
    for mod in modalities:
        mod_data = data.get(mod, {})
        for category, setups in mod_data.items():
            if "notes" in category:
                continue
            for setup, models in setups.items():
                for model, stats in models.items():
                    if model not in accum:
                        accum[model] = {
                            # For each metric we accumulate sum_x, sum_x2 and min/max and total runs
                            "guesses": {
                                "sum_x": 0.0,
                                "sum_x2": 0.0,
                                "min": None,
                                "max": None,
                            },
                            "illegal": {
                                "sum_x": 0.0,
                                "sum_x2": 0.0,
                                "min": None,
                                "max": None,
                            },
                            "invalid": {
                                "sum_x": 0.0,
                                "sum_x2": 0.0,
                                "min": None,
                                "max": None,
                            },
                            "repeated": {
                                "sum_x": 0.0,
                                "sum_x2": 0.0,
                                "min": None,
                                "max": None,
                            },
                            "nobox": {
                                "sum_x": 0.0,
                                "sum_x2": 0.0,
                                "min": None,
                                "max": None,
                            },
                            "n_runs": 0,
                        }

                    model_n = int(stats.get("n_runs", 1))

                    for key in ("guesses", "illegal", "invalid", "repeated", "nobox"):
                        val = stats.get(key)
                        # Normalize metric to (avg,std,min,max) if possible
                        if isinstance(val, dict) and "avg" in val:
                            avg = float(val.get("avg", 0.0))
                            std = float(val.get("std", 0.0))
                            mn = val.get("min", None)
                            mx = val.get("max", None)
                        elif isinstance(val, (int, float)):
                            avg = float(val)
                            std = 0.0
                            mn = avg
                            mx = avg
                        else:
                            # Missing metric: skip
                            continue

                        # accumulate sum and sum of squares for pooling
                        accum_entry = accum[model][key]
                        accum_entry["sum_x"] += avg * model_n
                        accum_entry["sum_x2"] += (std**2 + avg**2) * model_n
                        # update min/max
                        if mn is not None:
                            if accum_entry["min"] is None or mn < accum_entry["min"]:
                                accum_entry["min"] = mn
                        if mx is not None:
                            if accum_entry["max"] is None or mx > accum_entry["max"]:
                                accum_entry["max"] = mx

                    accum[model]["n_runs"] += model_n

    # Build summarized output using pooled statistics across setups/modalities
    summarized = {}
    import math

    for model, data_acc in accum.items():
        total_runs = int(data_acc.get("n_runs", 0))
        model_out = {}
        for key in ("guesses", "illegal", "invalid", "repeated", "nobox"):
            entry = data_acc[key]
            if total_runs <= 0:
                model_out[key] = {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                continue

            mean = entry["sum_x"] / total_runs
            mean_sq = entry["sum_x2"] / total_runs
            var = mean_sq - (mean**2)
            if var < 0 and var > -1e-8:
                var = 0.0
            std = math.sqrt(var) if var > 0 else 0.0
            mn = entry["min"] if entry["min"] is not None else 0.0
            mx = entry["max"] if entry["max"] is not None else 0.0

            model_out[key] = {
                "avg": mean,
                "std": std,
                "min": float(mn),
                "max": float(mx),
            }

        model_out["n_runs"] = total_runs
        summarized[model] = model_out

    out_path = Path("SWM/data/swm_image_errors_metrics.json")
    out_path.write_text(json.dumps(summarized, indent=4))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
