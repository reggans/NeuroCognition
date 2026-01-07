#!/usr/bin/env python3
"""Utility for slicing RAPM datasets by dataset_type.

Given an input dataset (image JSON or text JSONL), this script trims the
number of items per dataset type (mirroring the behaviour in
``load_evaluation_data``) and writes the reduced dataset to a new location.

If the dataset references image assets, the required image files are also
copied alongside the new dataset so that relative paths remain valid.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


IMAGE_KEYS = ["problem_matrix_image", "answer_choices_image", "full_image"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limit RAPM dataset size per dataset_type"
    )
    parser.add_argument(
        "--data_input",
        required=True,
        help="Path to RAPM dataset (JSON for image, JSONL for text)",
    )
    parser.add_argument(
        "--data_output", required=True, help="Destination path for the filtered dataset"
    )
    parser.add_argument(
        "--limit_per_type",
        type=int,
        default=0,
        help="Maximum number of items to keep for each dataset_type. Use 0 (default) to keep all items.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute counts without writing output or copying assets.",
    )
    return parser.parse_args()


def limit_items(
    items: Iterable[Dict[str, Any]],
    limit: int,
    key_fn,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Return items limited per key along with the retained counts."""
    if limit <= 0:
        retained = list(items)
        counts = defaultdict(int)
        for item in retained:
            counts[key_fn(item)] += 1
        return retained, counts

    counts = defaultdict(int)
    limited: List[Dict[str, Any]] = []
    for item in items:
        dtype = key_fn(item)
        if counts[dtype] >= limit:
            continue
        limited.append(item)
        counts[dtype] += 1
    return limited, counts


def detect_dataset_type(record: Dict[str, Any]) -> str:
    """Best-effort dataset-type extraction for text RAPM JSONL."""
    for key in ("dataset_type", "primary_category"):
        value = record.get(key)
        if value:
            return str(value)
    raw = record.get("raw")
    if isinstance(raw, dict):
        for key in ("dataset_type", "primary_category"):
            value = raw.get(key)
            if value:
                return str(value)
    return "default"


def copy_image_assets(
    records: Iterable[Dict[str, Any]],
    input_root: str,
    output_root: str,
    dry_run: bool = False,
) -> None:
    seen_paths = set()
    for entry in records:
        for key in IMAGE_KEYS:
            rel_path = entry.get(key)
            if not rel_path or rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            src = os.path.normpath(os.path.join(input_root, rel_path))
            dst = os.path.normpath(os.path.join(output_root, rel_path))
            if dry_run:
                if not os.path.exists(src):
                    print(f"[WARN] Missing image (dry-run): {src}")
                continue
            if not os.path.exists(src):
                print(f"[WARN] Missing image: {src}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def process_image_dataset(args: argparse.Namespace) -> None:
    with open(args.data_input, "r", encoding="utf-8") as f:
        payload = json.load(f)
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise SystemExit("Expected 'questions' list in image dataset JSON")

    limited_questions, counts = limit_items(
        questions,
        args.limit_per_type,
        key_fn=lambda item: item.get("dataset_type", "unknown"),
    )

    print("Retained items per dataset_type:")
    for dtype, count in sorted(counts.items()):
        print(f"  {dtype}: {count}")
    print(f"Total retained questions: {len(limited_questions)}")

    if args.dry_run:
        return

    input_root = os.path.dirname(os.path.abspath(args.data_input))
    output_root = os.path.dirname(os.path.abspath(args.data_output))

    os.makedirs(output_root, exist_ok=True)
    payload["questions"] = limited_questions
    with open(args.data_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    copy_image_assets(
        limited_questions,
        input_root=input_root,
        output_root=output_root,
        dry_run=args.dry_run,
    )


def process_text_dataset(args: argparse.Namespace) -> None:
    retained_items: List[Dict[str, Any]] = []
    counts: Dict[str, int]

    with open(args.data_input, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    retained_items, counts = limit_items(
        items,
        args.limit_per_type,
        key_fn=detect_dataset_type,
    )

    print("Retained items per dataset_type:")
    for dtype, count in sorted(counts.items()):
        print(f"  {dtype}: {count}")
    print(f"Total retained items: {len(retained_items)}")

    if args.dry_run:
        return

    os.makedirs(
        os.path.dirname(os.path.abspath(args.data_output)) or ".", exist_ok=True
    )
    with open(args.data_output, "w", encoding="utf-8") as f:
        for item in retained_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    input_ext = os.path.splitext(args.data_input)[1].lower()
    if input_ext == ".json":
        process_image_dataset(args)
    elif input_ext == ".jsonl":
        process_text_dataset(args)
    else:
        raise SystemExit(
            "Unsupported input format. Expected .json for image or .jsonl for text dataset."
        )


if __name__ == "__main__":
    main()
