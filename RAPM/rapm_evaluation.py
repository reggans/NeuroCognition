#!/usr/bin/env python3
"""RAPM evaluation (image & text) with optional OpenAI Batch API.
All prompt construction, formatting, parsing, batch request building, and scoring helpers
live in `RAPM/rapm_utils.py` to keep this driver lean.
"""
import argparse, json, os, re, sys, shutil
from collections import defaultdict

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # broad except ok for optional util

    def tqdm(it, desc=None):  # type: ignore
        if desc:
            print(desc)
        return it


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.model_wrapper import ModelWrapper
from RAPM.rapm_utils import (
    build_image_system_prompt,
    build_text_system_prompt,
    format_text_item_prompt,
    reconstruct_cell_constraint,
    build_image_batch_requests,
    build_text_batch_requests,
    score_image,
    score_text,
    parse_image_answer,
    parse_text_mc,
)

# ---------------- Data loading ---------------- #


def load_evaluation_data(path, limit_per_type=None):
    with open(path, "r") as f:
        data = json.load(f)
    questions = data["questions"]
    if limit_per_type is not None:
        by_type = defaultdict(list)
        for q in questions:
            by_type[q["dataset_type"]].append(q)
        limited = []
        for dt, lst in by_type.items():
            limited.extend(lst[:limit_per_type])
            print(f"Selected {min(len(lst), limit_per_type)} questions from {dt}")
        questions = limited
        print(f"Total questions after limiting: {len(questions)}")
    return questions


def load_text_rapm_jsonl(path):
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
            items.append(
                {
                    "id": obj.get("id") or f"text_rapm_{len(items)}",
                    "question_grid": obj.get("question_grid") or obj.get("full_grid"),
                    "options": obj.get("options", []),
                    "correct_index": obj.get("correct_index"),
                    "raw": obj,
                }
            )
    return items


# ---------------- Synchronous (image) ---------------- #


def run_rapm_evaluation(args):
    limit = args.limit_per_type if args.limit_per_type > 0 else None
    questions = load_evaluation_data(args.eval_data, limit_per_type=limit)
    print(f"Loaded {len(questions)} questions")
    dataset_types = set(q["dataset_type"] for q in questions)
    model = ModelWrapper(
        args.model,
        args.model_source,
        api_key=args.api_key,
        max_new_tokens=args.max_tokens,
        think_budget=args.think_budget,
        image_input=True,
        image_path=os.path.dirname(args.eval_data),
    )
    system_prompt = build_image_system_prompt(args)
    results = []
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    reasoning_traces = []
    current_image_path = None
    for i, q in enumerate(tqdm(questions, desc="Processing questions")):
        model.init_chat(system_prompt)
        src = os.path.join(os.path.dirname(args.eval_data), q["full_image"])
        current_image_path = os.path.join(model.image_path, "current.png")  # type: ignore
        shutil.copy2(src, current_image_path)
        resp = model.send_message("", cot=args.cot, truncate_history=True)
        m = re.search(r"<answer>(.*?)</answer>", resp)
        pred = parse_image_answer(m.group(1).strip() if m else None)
        correct = q["correct_answer"]
        is_correct = pred is not None and (pred - 1) == correct
        dt = q["dataset_type"]
        total_by_type[dt] += 1
        if is_correct:
            correct_by_type[dt] += 1
        results.append(
            {
                "id": q["id"],
                "dataset_type": dt,
                "correct_answer": correct,
                "predicted_answer": pred,
                "is_correct": is_correct,
                "response": resp,
                "image_path": q["full_image"],
            }
        )
        if args.cot and model.reasoning_trace:
            reasoning_traces.append(
                {"question_id": q["id"], "reasoning": model.reasoning_trace[-1]}
            )
        if (i + 1) % 100 == 0:
            acc = sum(r["is_correct"] for r in results) / len(results)
            print(f"Progress: {i+1}/{len(questions)} accuracy {acc:.3f}")
    if current_image_path and os.path.exists(current_image_path):
        os.remove(current_image_path)
    total_correct = sum(r["is_correct"] for r in results)
    total_questions = len(results)
    overall_acc = total_correct / total_questions if total_questions else 0
    print("\n=== RAPM IMAGE RESULTS ===")
    print(f"Model: {args.model}")
    print(f"Overall accuracy: {overall_acc:.3f} ({total_correct}/{total_questions})")
    for dt in sorted(dataset_types):
        acc = correct_by_type[dt] / total_by_type[dt] if total_by_type[dt] else 0
        print(f"  {dt}: {acc:.3f} ({correct_by_type[dt]}/{total_by_type[dt]})")
    summary = {
        "model": args.model,
        "model_source": args.model_source,
        "total_questions": total_questions,
        "total_correct": total_correct,
        "overall_accuracy": overall_acc,
        "accuracy_by_type": {
            dt: {
                "correct": correct_by_type[dt],
                "total": total_by_type[dt],
                "accuracy": (
                    correct_by_type[dt] / total_by_type[dt] if total_by_type[dt] else 0
                ),
            }
            for dt in dataset_types
        },
        "args": vars(args),
    }
    # history no longer persisted externally
    return results, summary, [], reasoning_traces


# ---------------- Synchronous (text) ---------------- #


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
    system_prompt = build_text_system_prompt(args)
    results = []
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    reasoning_traces = []
    for item in tqdm(items, desc="Processing text RAPM"):
        model.init_chat(system_prompt)
        prompt = format_text_item_prompt(item, args.answer_mode)
        resp = model.send_message(prompt, cot=args.cot, truncate_history=True)
        m = re.search(r"<answer>(.*?)</answer>", resp, re.DOTALL)
        predicted = None
        is_correct = False
        matches_gold = False
        constraint_valid = False
        if args.answer_mode == "mc":
            predicted = parse_text_mc(m.group(1).strip() if m else None)
            ci = item["correct_index"]
            is_correct = predicted is not None and (predicted - 1) == ci
        else:
            gold = item["raw"].get("answer")
            cdesc = (item["raw"].get("cell_constraints") or {}).get("2,2")
            cell_constraint = reconstruct_cell_constraint(cdesc) if cdesc else None
            if m:
                gen = m.group(1).strip()
                if gen.startswith('"') and gen.endswith('"') and len(gen) >= 2:
                    gen = gen[1:-1]
                predicted = gen
                if cell_constraint and gen:
                    from RAPM.text_rapm.validator import cell_satisfies

                    constraint_valid = cell_satisfies(gen, cell_constraint)
                matches_gold = gen == gold
                is_correct = constraint_valid
            ci = None
        cats = (
            item["raw"].get("credited_categories")
            or item["raw"].get("assigned_categories")
            or [item["raw"].get("primary_category")]
            if item["raw"].get("primary_category")
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
                "correct_index": ci,
                **(
                    {"matches_gold": matches_gold, "constraint_valid": constraint_valid}
                    if args.answer_mode == "gen"
                    else {}
                ),
                "is_correct": is_correct,
                "response": resp,
                "categories": cats,
            }
        )
        if args.cot and model.reasoning_trace:
            reasoning_traces.append(
                {"id": item["id"], "reasoning": model.reasoning_trace[-1]}
            )
    total_correct = sum(r["is_correct"] for r in results)
    total = len(results)
    overall_acc = total_correct / total if total else 0
    print("\n=== RAPM TEXT RESULTS ===")
    print(f"Model: {args.model}")
    print(f"Overall accuracy: {overall_acc:.3f} ({total_correct}/{total})")
    if cat_total:
        for c in sorted(cat_total):
            acc = cat_correct[c] / cat_total[c] if cat_total[c] else 0
            print(f"  {c}: {acc:.3f} ({cat_correct[c]}/{cat_total[c]})")
    summary = {
        "model": args.model,
        "mode": "text",
        "answer_mode": args.answer_mode,
        "total": total,
        "correct": total_correct,
        "overall_accuracy": overall_acc,
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
    return results, summary, [], reasoning_traces


# ---------------- Batch submit / collect ---------------- #


def batch_submit_rapm(args):
    if args.model_source != "openai":
        raise SystemExit("Batch mode only for openai source")
    model = ModelWrapper(
        args.model,
        args.model_source,
        api_key=args.api_key,
        max_new_tokens=args.max_tokens,
        think_budget=args.think_budget,
        image_input=(args.mode == "image"),
        # Provide image directory so ModelWrapper validation passes for image mode
        image_path=(os.path.dirname(args.eval_data) if args.mode == "image" else None),
    )
    if args.mode == "image":
        questions = load_evaluation_data(
            args.eval_data,
            limit_per_type=(args.limit_per_type if args.limit_per_type > 0 else None),
        )
        system_prompt = build_image_system_prompt(args)
        requests = build_image_batch_requests(args, questions, system_prompt)
        meta = {"task": "rapm_image", "count": str(len(questions)), "mode": "image"}
    else:
        items = load_text_rapm_jsonl(args.eval_data)
        system_prompt = build_text_system_prompt(args)
        requests = build_text_batch_requests(args, items, system_prompt)
        meta = {"task": f"rapm_text_{args.answer_mode}", "count": str(len(items)), "mode": "text"}
    model.batch_create_requests(requests, args.batch_requests_path)
    batch_id = model.batch_submit(
        args.batch_requests_path,
        completion_window=args.batch_completion_window,
        metadata=meta,
    )
    with open(args.batch_id_path, "w") as f:
        f.write(batch_id)
    print(f"Submitted batch id: {batch_id}")
    return batch_id


def batch_collect_rapm(args):
    if args.model_source != "openai":
        raise SystemExit("Batch mode only for openai source")
    # Inspect batch metadata to ensure mode matches submission
    temp_model = ModelWrapper(
        args.model,
        args.model_source,
        api_key=args.api_key,
        max_new_tokens=args.max_tokens,
        think_budget=args.think_budget,
        image_input=False,
    )
    try:
        status_obj = temp_model.batch_status(args.batch_id)
        meta = getattr(status_obj, "metadata", {}) or {}
        submitted_mode = meta.get("mode") if isinstance(meta, dict) else None
        if submitted_mode and submitted_mode != args.mode:
            raise SystemExit(
                f"Batch {args.batch_id} was submitted for mode '{submitted_mode}' but collect requested with mode '{args.mode}'. Rerun with --mode {submitted_mode}."
            )
    except Exception:
        pass  # Non-fatal; proceed
    if args.mode == "image":
        questions = load_evaluation_data(
            args.eval_data,
            limit_per_type=(args.limit_per_type if args.limit_per_type > 0 else None),
        )
    else:
        items = load_text_rapm_jsonl(args.eval_data)
    model = ModelWrapper(
        args.model,
        args.model_source,
        api_key=args.api_key,
        max_new_tokens=args.max_tokens,
        think_budget=args.think_budget,
    image_input=(args.mode == "image"),
    image_path=(os.path.dirname(args.eval_data) if args.mode == "image" else None),
    )
    parsed = model.batch_collect(
        args.batch_id, output_jsonl_path=args.batch_output_jsonl
    )
    if isinstance(parsed, dict):  # status summary, not ready
        print(parsed.get("message"))
        # Return empty results with summary status so caller can decide to retry later
        status_summary = {"batch_status": parsed, "args": vars(args)}
        return [], status_summary, [], []
    responses = {}
    for obj in parsed:
        cid = obj.get("custom_id")
        if not cid:
            continue
        try:
            idx = int(cid.rsplit("_", 1)[1])
        except Exception:
            continue
        body = (obj.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            responses[idx] = {"content": None}
            continue
        content = choices[0].get("message", {}).get("content")
        responses[idx] = {"content": content}
    if args.mode == "image":
        results, s, reasoning_traces = score_image(questions, responses, args)
        s["args"] = vars(args)
        summary = s
    else:
        results, s, reasoning_traces = score_text(items, responses, args)
        s["args"] = vars(args)
        summary = s
    return results, summary, [], reasoning_traces


# ---------------- CLI ---------------- #


def main():
    p = argparse.ArgumentParser(description="Run RAPM evaluation (image or text)")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--model_source", default="openrouter", choices=["openai", "openrouter", "vllm"]
    )
    p.add_argument("--mode", default="image", choices=["image", "text"])
    p.add_argument("--eval_data", required=True)
    p.add_argument("--cot", action="store_true")
    p.add_argument("--patterns", action="store_true")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--think_budget", type=int, default=256)
    p.add_argument("--api_key")
    p.add_argument("--output_dir", default="rapm_data")
    p.add_argument("--limit_per_type", type=int, default=100)
    p.add_argument(
        "--answer_mode",
        default="mc",
        choices=["mc", "gen"],
        help="Text mode answer type",
    )
    # Batch
    p.add_argument("--batch_mode", default="off", choices=["off", "submit", "collect"])
    p.add_argument("--batch_requests_path", default="rapm_batch_requests.jsonl")
    p.add_argument("--batch_id", default=None)
    p.add_argument("--batch_id_path", default="rapm_batch_id.txt")
    p.add_argument("--batch_output_jsonl", default="rapm_batch_output.jsonl")
    p.add_argument("--batch_completion_window", default="24h")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = f"{args.model_source}_{args.model.replace('/', '-')}_rapm_{args.mode}"
    if args.mode == "text" and args.answer_mode == "gen":
        base += "_gen"
    if args.patterns:
        base += "_pat"
    if args.cot:
        base += "_cot"
    results_path = os.path.join(args.output_dir, f"{base}_results.json")
    summary_path = os.path.join(args.output_dir, f"{base}_summary.json")
    reasoning_path = os.path.join(args.output_dir, f"{base}_reasoning.json")

    if os.path.exists(results_path) and args.batch_mode == "off":
        print(f"Results already exist at {results_path}")
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if args.mode == "image":
            print(
                f"Overall accuracy: {summary['overall_accuracy']:.3f} ({summary['total_correct']}/{summary['total_questions']})"
            )
        else:
            print(
                f"Overall accuracy: {summary['overall_accuracy']:.3f} ({summary['correct']}/{summary['total']})"
            )
        return

    if args.batch_mode == "submit":
        bid = batch_submit_rapm(args)
        print(f"Batch submitted. ID: {bid}")
        return
    if args.batch_mode == "collect":
        if not args.batch_id:
            if os.path.exists(args.batch_id_path):
                with open(args.batch_id_path, "r") as f:
                    args.batch_id = f.read().strip()
            else:
                raise SystemExit("--batch_id not provided and batch id file missing")
        print(f"Collecting batch {args.batch_id} ...")
        results, summary, history, reasoning_traces = batch_collect_rapm(args)
        base += "_batch"
        results_path = os.path.join(args.output_dir, f"{base}_results.json")
        summary_path = os.path.join(args.output_dir, f"{base}_summary.json")
        reasoning_path = os.path.join(args.output_dir, f"{base}_reasoning.json")
    else:
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
    if reasoning_traces:
        with open(reasoning_path, "w") as f:
            json.dump(reasoning_traces, f, indent=2)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
