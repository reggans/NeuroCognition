"""Compare token usage and cost of one multi-turn chat with prompt caching on vs off.

Calls the real API and spends credits: two arms x --turns requests, each resending
a ~--prompt_tokens system prompt plus the growing history.

Example:
  python compare_prompt_cache.py --model_source openrouter --model anthropic/claude-sonnet-5
"""

import argparse
import json
import logging
import uuid

from shared.model_wrapper import ModelWrapper

COLORS = ["red", "green", "blue", "yellow", "purple", "orange"]
TOKENS_PER_FACT = 20  # rough; only sizes the prompt


def build_system_prompt(run_tag: str, n_facts: int) -> str:
    # Unique tag up front so arms (and earlier runs) never share a cached prefix
    lines = [
        f"Session {run_tag}.",
        "You are a warehouse assistant. Answer each question using only the facts below.",
        "Reply with the answer only, no explanation.",
        "",
    ]
    for i in range(n_facts):
        lines.append(
            f"Fact {i}: item-{i} is stored in box {(i * 7919) % 97} and is colored {COLORS[i % len(COLORS)]}."
        )
    return "\n".join(lines)


def question(turn: int, n_facts: int) -> str:
    return f"Which box holds item-{(turn * 37) % n_facts}?"


def run_arm(args, prompt_cache: bool):
    model = ModelWrapper(
        args.model,
        args.model_source,
        max_new_tokens=args.max_new_tokens,
        n_retry=2,
        prompt_cache=prompt_cache,
    )
    n_facts = max(1, args.prompt_tokens // TOKENS_PER_FACT)
    model.init_chat(build_system_prompt(uuid.uuid4().hex, n_facts))

    rows = []
    for t in range(args.turns):
        if model.send_message(question(t, n_facts)) is None:
            raise RuntimeError(f"cache={'on' if prompt_cache else 'off'}: turn {t + 1} failed")
        if model.last_usage is None:
            raise RuntimeError("Provider returned no usage data; cannot compare")
        rows.append({"turn": t + 1, **model.last_usage})
    return rows, dict(model.usage_totals)


def print_report(on_rows, on_tot, off_rows, off_tot):
    print(f"\n{'turn':>4} | {'on: prompt':>10} {'cached':>8} {'cost $':>10} | {'off: prompt':>11} {'cached':>8} {'cost $':>10}")
    for a, b in zip(on_rows, off_rows):
        print(
            f"{a['turn']:>4} | {a['prompt_tokens']:>10} {a['cached_tokens']:>8} {a['cost']:>10.5f} | "
            f"{b['prompt_tokens']:>11} {b['cached_tokens']:>8} {b['cost']:>10.5f}"
        )
    print("-" * 78)
    for name, tot in (("on", on_tot), ("off", off_tot)):
        share = tot["cached_tokens"] / tot["prompt_tokens"] if tot["prompt_tokens"] else 0.0
        print(
            f"cache {name:>3}: prompt={tot['prompt_tokens']} cached={tot['cached_tokens']} "
            f"({share:.0%}) completion={tot['completion_tokens']} cost=${tot['cost']:.5f}"
        )

    if on_tot["cost"] and off_tot["cost"]:
        saved = off_tot["cost"] - on_tot["cost"]
        print(f"\nsaved ${saved:.5f} ({saved / off_tot['cost']:.0%} of the uncached cost)")
    else:
        print("\nProvider reported no cost; compare cached tokens against its pricing page.")
    if off_tot["cached_tokens"]:
        print(
            "Note: the 'off' arm still got cache hits. This provider caches automatically; "
            "'off' only drops this repo's cache hints."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_source", required=True, choices=["openai", "openrouter", "google", "vllm"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--prompt_tokens", type=int, default=6000, help="Approx. system prompt size")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--out", help="Optional path to save per-turn results as JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    on_rows, on_tot = run_arm(args, prompt_cache=True)
    off_rows, off_tot = run_arm(args, prompt_cache=False)
    print_report(on_rows, on_tot, off_rows, off_tot)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"args": vars(args), "cache_on": {"turns": on_rows, "totals": on_tot},
                 "cache_off": {"turns": off_rows, "totals": off_tot}},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
