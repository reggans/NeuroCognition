from __future__ import annotations
import argparse
import json
import sys
from typing import List, Optional
from .api import generate_text_rapm_item


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate text RAPM items and save to JSON / JSONL."
    )
    p.add_argument(
        "--count", type=int, default=1, help="Number of items to generate (default: 1)"
    )
    # --seed deprecated: retained for backward compatibility but ignored.
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(Deprecated / Ignored) Previously used as a base seed; now randomness is always fresh.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="-",
        help="Output file path or - for stdout (default: -)",
    )
    p.add_argument(
        "--jsonl",
        action="store_true",
        help="Write JSON Lines (one JSON object per line) instead of a single JSON array",
    )
    p.add_argument(
        "--debug", action="store_true", help="Include debug information in each item"
    )
    p.add_argument(
        "--pretty", action="store_true", help="Pretty print (indent=2) the JSON output"
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    items = []
    for _ in range(args.count):
        # Pass seed=None to trigger fresh randomness each generation.
        item = generate_text_rapm_item(seed=None, debug=args.debug)
        # No _seed stored anymore (removed per user request for fully random output)
        items.append(item)

    out_stream = sys.stdout
    close_needed = False
    if args.output != "-":
        out_stream = open(args.output, "w", encoding="utf-8")
        close_needed = True

    try:
        if args.jsonl:
            for obj in items:
                json.dump(obj, out_stream, ensure_ascii=False)
                out_stream.write("\n")
        else:
            if args.pretty:
                json.dump(items, out_stream, ensure_ascii=False, indent=2)
            else:
                json.dump(items, out_stream, ensure_ascii=False)
            if args.pretty:
                out_stream.write("\n")
    finally:
        if close_needed:
            out_stream.close()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
