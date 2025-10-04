#!/usr/bin/env python3
"""RAPM evaluation (image & text) with optional OpenAI Batch API.
All prompt construction, formatting, parsing, batch request building, and scoring helpers
live in `RAPM/rapm_utils.py` to keep this driver lean.
"""
import argparse, json, os, re, sys, shutil
from collections import defaultdict, OrderedDict
from tempfile import NamedTemporaryFile

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


def atomic_write_json(path, data):
    if path is None:
        return
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp", encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def compute_image_summary(results, args):
    total_questions = len(results)
    total_correct = sum(1 for r in results if r.get("is_correct"))
    dataset_types = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        dt = r.get("dataset_type")
        if dt is None:
            continue
        dataset_types[dt]["total"] += 1
        if r.get("is_correct"):
            dataset_types[dt]["correct"] += 1
    summary = {
        "model": args.model,
        "model_source": args.model_source,
        "total_questions": total_questions,
        "total_correct": total_correct,
        "overall_accuracy": (total_correct / total_questions) if total_questions else 0,
        "accuracy_by_type": {
            dt: {
                "correct": vals["correct"],
                "total": vals["total"],
                "accuracy": (vals["correct"] / vals["total"]) if vals["total"] else 0,
            }
            for dt, vals in dataset_types.items()
        },
        "args": vars(args),
    }
    return summary


def compute_text_summary(results, args):
    total = len(results)
    total_correct = sum(1 for r in results if r.get("is_correct"))
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    failed_items = []
    for r in results:
        cats = r.get("categories") or []
        for c in cats:
            if not c:
                continue
            cat_total[c] += 1
            if r.get("is_correct"):
                cat_correct[c] += 1
        if r.get("failure_reason") or r.get("response") is None:
            identifier = r.get("id") or r.get("question_id")
            if identifier:
                failed_items.append(identifier)
    summary = {
        "model": args.model,
        "mode": "text",
        "answer_mode": args.answer_mode,
        "total": total,
        "correct": total_correct,
        "overall_accuracy": (total_correct / total) if total else 0,
        "accuracy_by_category": {
            c: {
                "correct": cat_correct[c],
                "total": cat_total[c],
                "accuracy": (cat_correct[c] / cat_total[c]) if cat_total[c] else 0,
            }
            for c in sorted(cat_total.keys())
        },
        "failed_items": failed_items,
        "args": vars(args),
    }
    return summary


def save_progress(result_map, reasoning_map, args, results_path, summary_path, reasoning_path, mode):
    results_list = list(result_map.values())
    if mode == "image":
        summary = compute_image_summary(results_list, args)
    else:
        summary = compute_text_summary(results_list, args)
    atomic_write_json(results_path, results_list)
    atomic_write_json(summary_path, summary)
    if reasoning_map:
        reasoning_list = list(reasoning_map.values())
        atomic_write_json(reasoning_path, reasoning_list)


# ---------------- Synchronous (image) ---------------- #


def run_rapm_evaluation(
    args,
    existing_results=None,
    existing_reasoning=None,
    existing_summary=None,
    results_path=None,
    summary_path=None,
    reasoning_path=None,
):
    existing_results = existing_results or []
    existing_reasoning = existing_reasoning or []
    existing_summary = existing_summary or {}
    retry_ids = set()
    if isinstance(existing_summary, dict):
        failed_from_summary = existing_summary.get("failed_items") or []
        for fid in failed_from_summary:
            if fid is not None:
                retry_ids.add(fid)
    limit = args.limit_per_type if args.limit_per_type > 0 else None
    questions = load_evaluation_data(args.eval_data, limit_per_type=limit)
    print(f"Loaded {len(questions)} questions")
    dataset_types = set(q["dataset_type"] for q in questions)
    results_map = OrderedDict()
    for res in existing_results:
        rid = res.get("id")
        if rid is None:
            continue
        if res.get("failure_reason") or res.get("response") is None:
            retry_ids.add(rid)
            continue
        if rid in retry_ids or rid in results_map:
            continue
        results_map[rid] = res
    reasoning_map = OrderedDict()
    for trace in existing_reasoning:
        key = trace.get("question_id") or trace.get("id")
        if key is None or key in retry_ids or key in reasoning_map:
            continue
        reasoning_map[key] = trace
    if retry_ids:
        print(f"Will retry {len(retry_ids)} previously failed image items.")
    processed_existing = len(results_map)
    skipped = 0
    if results_path and summary_path:
        save_progress(results_map, reasoning_map, args, results_path, summary_path, reasoning_path, "image")
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
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    for res in results_map.values():
        dt = res.get("dataset_type")
        if dt is None:
            continue
        total_by_type[dt] += 1
        if res.get("is_correct"):
            correct_by_type[dt] += 1
    reasoning_traces = []
    current_image_path = None
    for i, q in enumerate(tqdm(questions, desc="Processing questions")):
        if q["id"] in results_map:
            skipped += 1
            continue
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
        result_entry = {
            "id": q["id"],
            "dataset_type": dt,
            "correct_answer": correct,
            "predicted_answer": pred,
            "is_correct": is_correct,
            "response": resp,
            "image_path": q["full_image"],
        }
        results_map[q["id"]] = result_entry
        if args.cot and model.reasoning_trace:
            reasoning_map[q["id"]] = {
                "question_id": q["id"],
                "reasoning": model.reasoning_trace[-1],
            }
        if results_path and summary_path:
            save_progress(results_map, reasoning_map, args, results_path, summary_path, reasoning_path, "image")
        if (i + 1) % 100 == 0:
            processed = len(results_map) - processed_existing
            recent = list(results_map.values())[-processed:] if processed else []
            acc = (
                sum(r["is_correct"] for r in recent) / processed
                if processed
                else 0
            )
            print(
                f"Progress: {i+1}/{len(questions)} (new processed: {processed}) accuracy {acc:.3f}"
            )
    if current_image_path and os.path.exists(current_image_path):
        os.remove(current_image_path)
    all_results = list(results_map.values())
    summary = compute_image_summary(all_results, args)
    total_correct = summary.get("total_correct", 0)
    total_questions = summary.get("total_questions", 0)
    overall_acc = summary.get("overall_accuracy", 0)
    print("\n=== RAPM IMAGE RESULTS ===")
    print(f"Model: {args.model}")
    print(f"Overall accuracy: {overall_acc:.3f} ({total_correct}/{total_questions})")
    for dt in sorted(summary.get("accuracy_by_type", {})):
        stats = summary["accuracy_by_type"][dt]
        print(
            f"  {dt}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})"
        )
    if results_path and summary_path:
        save_progress(results_map, reasoning_map, args, results_path, summary_path, reasoning_path, "image")
    reasoning_traces = list(reasoning_map.values())
    if skipped:
        print(f"Skipped {skipped} questions already present in results.")
    # history no longer persisted externally
    return all_results, summary, [], reasoning_traces


# ---------------- Synchronous (text) ---------------- #


def run_text_rapm_evaluation(
    args,
    existing_results=None,
    existing_reasoning=None,
    existing_summary=None,
    results_path=None,
    summary_path=None,
    reasoning_path=None,
):
    existing_results = existing_results or []
    existing_reasoning = existing_reasoning or []
    existing_summary = existing_summary or {}
    retry_ids = set()
    if isinstance(existing_summary, dict):
        failed_from_summary = existing_summary.get("failed_items") or []
        for fid in failed_from_summary:
            if fid is not None:
                retry_ids.add(fid)
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
    results_map = OrderedDict()
    for res in existing_results:
        rid = res.get("id")
        if rid is None:
            continue
        if res.get("failure_reason") or res.get("response") is None:
            retry_ids.add(rid)
            continue
        if rid in retry_ids or rid in results_map:
            continue
        results_map[rid] = res
    reasoning_map = OrderedDict()
    for trace in existing_reasoning:
        key = trace.get("id") or trace.get("question_id")
        if key is None or key in retry_ids or key in reasoning_map:
            continue
        reasoning_map[key] = trace
    if retry_ids:
        print(f"Will retry {len(retry_ids)} previously failed text items.")
    skipped = 0
    if results_path and summary_path:
        save_progress(results_map, reasoning_map, args, results_path, summary_path, reasoning_path, "text")
    for item in tqdm(items, desc="Processing text RAPM"):
        if item["id"] in results_map:
            skipped += 1
            continue
        model.init_chat(system_prompt)
        prompt = format_text_item_prompt(item, args.answer_mode)
        resp = model.send_message(prompt, cot=args.cot, truncate_history=True)
        if resp is None:
            # Record failure and continue without crashing
            result_entry = {
                "id": item["id"],
                "predicted_answer": None,
                "correct_index": item.get("correct_index"),
                **(
                    {}
                    if args.answer_mode == "mc"
                    else {"matches_gold": False, "constraint_valid": False}
                ),
                "is_correct": False,
                "response": None,
                "categories": item["raw"].get("credited_categories")
                or item["raw"].get("assigned_categories")
                or ([item["raw"].get("primary_category")] if item["raw"].get("primary_category") else []),
                "failure_reason": "no_response",
            }
            results_map[item["id"]] = result_entry
            if results_path and summary_path:
                save_progress(results_map, reasoning_map, args, results_path, summary_path, reasoning_path, "text")
            continue
        # NOTE: If failures correlate with very long outputs, consider lowering max_new_tokens
        # adaptively (e.g., halve after a None) or trimming the prompt. Placeholder left here
        # for a future adaptive token budget feature.
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
        result_entry = {
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
        results_map[item["id"]] = result_entry
        if args.cot and model.reasoning_trace:
            reasoning_map[item["id"]] = {
                "id": item["id"],
                "reasoning": model.reasoning_trace[-1],
            }
        if results_path and summary_path:
            save_progress(results_map, reasoning_map, args, results_path, summary_path, reasoning_path, "text")
    all_results = list(results_map.values())
    print("\n=== RAPM TEXT RESULTS ===")
    print(f"Model: {args.model}")
    summary = compute_text_summary(all_results, args)
    total_correct = summary.get("correct", 0)
    total = summary.get("total", 0)
    overall_acc = summary.get("overall_accuracy", 0)
    print(f"Overall accuracy: {overall_acc:.3f} ({total_correct}/{total})")
    if summary.get("failed_items"):
        fi = summary["failed_items"]
        print(f"Failed items (no response) : {len(fi)} -> {fi[:10]}{'...' if len(fi) > 10 else ''}")
    if summary.get("accuracy_by_category"):
        for c in sorted(summary["accuracy_by_category"]):
            stats = summary["accuracy_by_category"][c]
            print(f"  {c}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
    reasoning_traces = list(reasoning_map.values())
    if skipped:
        print(f"Skipped {skipped} items already present in results.")
    return all_results, summary, [], reasoning_traces


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

    existing_results = []
    existing_reasoning = []
    existing_summary = None
    if os.path.exists(results_path) and args.batch_mode == "off":
        try:
            with open(results_path, "r") as f:
                existing_results = json.load(f)
            print(f"Found {len(existing_results)} existing results at {results_path}. Will resume/skip duplicates.")
        except Exception as exc:
            print(f"Warning: failed to load existing results ({exc}). Starting fresh.")
            existing_results = []
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    existing_summary = json.load(f)
            except Exception:
                existing_summary = None
        if os.path.exists(reasoning_path):
            try:
                with open(reasoning_path, "r") as f:
                    existing_reasoning = json.load(f)
            except Exception:
                existing_reasoning = []
        if existing_summary:
            if args.mode == "image":
                print(
                    f"Existing accuracy: {existing_summary.get('overall_accuracy', 0):.3f} "
                    f"({existing_summary.get('total_correct', 0)}/{existing_summary.get('total_questions', 0)})"
                )
            else:
                print(
                    f"Existing accuracy: {existing_summary.get('overall_accuracy', 0):.3f} "
                    f"({existing_summary.get('correct', 0)}/{existing_summary.get('total', 0)})"
                )

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
            results, summary, history, reasoning_traces = run_rapm_evaluation(
                args,
                existing_results=existing_results if args.batch_mode == "off" else None,
                existing_reasoning=existing_reasoning if args.batch_mode == "off" else None,
                existing_summary=existing_summary if args.batch_mode == "off" else None,
                results_path=results_path,
                summary_path=summary_path,
                reasoning_path=reasoning_path,
            )
        else:
            print("Starting text RAPM evaluation...")
            results, summary, history, reasoning_traces = run_text_rapm_evaluation(
                args,
                existing_results=existing_results if args.batch_mode == "off" else None,
                existing_reasoning=existing_reasoning if args.batch_mode == "off" else None,
                existing_summary=existing_summary if args.batch_mode == "off" else None,
                results_path=results_path,
                summary_path=summary_path,
                reasoning_path=reasoning_path,
            )

    atomic_write_json(results_path, results)
    atomic_write_json(summary_path, summary)
    if reasoning_traces:
        atomic_write_json(reasoning_path, reasoning_traces)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
