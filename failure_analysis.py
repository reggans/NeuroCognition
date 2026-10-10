#!/usr/bin/env python3
"""Per-turn failure taxonomy and failure-depth analysis for WCST and SWM logs.

Works on existing result files; no reruns needed.

    python failure_analysis.py wcst --data_dir WCST/data/text
    python failure_analysis.py swm --data_dir SWM/data/text

WCST error labels (one per incorrect turn, first matching wins):
    format        answer missing or not an option index
    unclassified  chosen card's attributes unknown (e.g. "empty" variant)
    switch        first turn of a new rule, chosen card follows the previous rule (expected)
    set_loss      error after the rule was acquired (>= `acquisition` consecutive correct)
    perseverative chosen card follows the previous rule after feedback said it's wrong
    no_match      chosen card shares no attribute with the given card
    non_update    chosen card only matches attributes already disconfirmed for this rule
    exploratory   tests an attribute not yet disconfirmed (expected)

SWM labels: found, empty (expected); illegal, repeated, format, nobox (failures).
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional, Sequence

WCST_UNEXPECTED = frozenset(
    {"format", "set_loss", "perseverative", "no_match", "non_update"}
)
SWM_UNEXPECTED = frozenset({"format", "nobox", "illegal", "repeated"})

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
COLORS = frozenset({"red", "green", "blue", "yellow"})
SHAPES = frozenset({"circle", "triangle", "star", "square"})
# Image WCST options are fixed (not shuffled)
IMAGE_OPTIONS = [
    {"number": "one", "color": "red", "shape": "circle", "background": "red"},
    {"number": "two", "color": "green", "shape": "triangle", "background": "green"},
    {"number": "three", "color": "blue", "shape": "star", "background": "blue"},
    {"number": "four", "color": "yellow", "shape": "square", "background": "yellow"},
]

CARD_RULE_PATTERNS = {
    "color": r"(?<!background )\bcolou?r",
    "shape": r"\bshape",
    "number": r"\b(?:number|count)",
}
BACKGROUND_PATTERN = r"\bbackground"
STRING_RULE_PATTERNS = {
    "length": r"\blength",
    "vowels": r"\bvowel",
    "consonants": r"\bconsonant",
}


@dataclass
class WcstEvent:
    file: str
    run: str
    turn: int
    rule: str
    segment: int
    correct: bool
    label: str
    unexpected: bool
    matched: Optional[str]  # "|"-joined attributes shared by given and chosen card
    model_ans: Optional[str]
    true_ans: Optional[str]
    stated_rule: Optional[str]
    dissociation: Optional[bool]  # stated rule not among the chosen card's matches
    prompt_tokens: Optional[int] = None
    context_chars: Optional[int] = None


@dataclass
class SwmEvent:
    file: str
    run: str
    turn: int
    search: int  # tokens found before this turn
    label: str
    unexpected: bool
    choice: Optional[str]
    p_error_chance: float  # error probability of a uniformly random box pick
    prompt_tokens: Optional[int] = None
    context_chars: Optional[int] = None


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return 0


def context_chars(messages: Sequence[dict], stateless: bool = False) -> list[int]:
    """Text chars in context at each API call (images not counted).

    Stateless runs only send the system prompt and the current user turn.
    """
    if stateless:
        system = _text_len(messages[0]["content"])
        return [system + _text_len(m["content"]) for m in messages if m["role"] == "user"]
    out, total = [], 0
    for message in messages:
        total += _text_len(message["content"])
        if message["role"] == "user":
            out.append(total)
    return out


def align(values: Optional[list], n: int, allow_trailing: bool) -> Optional[list]:
    """Per-turn values, or None when call counts don't match turns (e.g. continuations)."""
    if values is None:
        return None
    if len(values) == n or (allow_trailing and len(values) == n + 1):
        return list(values[:n])
    return None


def kaplan_meier(durations: Sequence[int], events: Sequence[bool]) -> list[tuple[int, float]]:
    """Survival curve as [(turn, S(turn))] at each turn with an event."""
    pairs = sorted(zip(durations, events))
    at_risk, survival, curve, i = len(pairs), 1.0, [], 0
    while i < len(pairs):
        t, n_events, n_leaving = pairs[i][0], 0, 0
        while i < len(pairs) and pairs[i][0] == t:
            n_events += int(pairs[i][1])
            n_leaving += 1
            i += 1
        if n_events:
            survival *= 1 - n_events / at_risk
            curve.append((t, survival))
        at_risk -= n_leaving
    return curve


def summarize(events: Sequence[Any], bin_width: int) -> dict:
    """Label counts, first-failure survival and per-depth error rates."""
    by_run: dict[str, list] = defaultdict(list)
    for event in events:
        by_run[event.run].append(event)

    durations, observed, first_labels, first_tokens = [], [], Counter(), []
    for run_events in by_run.values():
        first = next((e for e in run_events if e.unexpected), None)
        if first is None:
            durations.append(len(run_events))
            observed.append(False)
        else:
            durations.append(first.turn)
            observed.append(True)
            first_labels[first.label] += 1
            if first.prompt_tokens is not None:
                first_tokens.append(first.prompt_tokens)

    curve = kaplan_meier(durations, observed)
    bins: dict[int, list] = defaultdict(list)
    for event in events:
        bins[(event.turn - 1) // bin_width].append(event)

    return {
        "n_runs": len(by_run),
        "n_turns": len(events),
        "labels": dict(Counter(e.label for e in events)),
        "unexpected_rate": (
            sum(e.unexpected for e in events) / len(events) if events else None
        ),
        "first_failure": {
            "turns": durations,
            "observed": observed,
            "labels": dict(first_labels),
            "prompt_tokens": first_tokens,
            "median_turn": next((t for t, s in curve if s <= 0.5), None),
            "km_curve": curve,
        },
        "by_depth": [
            {
                "turns": [b * bin_width + 1, (b + 1) * bin_width],
                "n": len(group),
                "unexpected_rate": sum(e.unexpected for e in group) / len(group),
                "labels": dict(Counter(e.label for e in group)),
            }
            for b, group in sorted(bins.items())
        ],
    }


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _optional_run_field(path: Path, run: str, key: Optional[str] = None) -> Any:
    if not path.exists():
        return None
    data = _load_json(path).get(run)
    if data is None or key is None:
        return data
    return data.get(key)


def _depth_columns(
    path_usage: Path, path_history: Path, run: str, n: int, allow_trailing: bool
) -> tuple[list, list]:
    calls = _optional_run_field(path_usage, run, "calls")
    tokens = align(
        [c["prompt_tokens"] for c in calls] if calls is not None else None,
        n,
        allow_trailing,
    )
    if calls is not None and tokens is None:
        print(f"  warn: {path_usage.name} [{run}] has {len(calls)} calls for {n} turns; prompt_tokens dropped")

    messages = _optional_run_field(path_history, run)
    stateless = "_stateless" in path_history.stem
    chars = align(context_chars(messages, stateless) if messages else None, n, allow_trailing)
    if messages and chars is None:
        print(f"  warn: {path_history.name} [{run}] doesn't align with {n} turns; context_chars dropped")

    return tokens or [None] * n, chars or [None] * n


def write_outputs(task: str, events: list, summaries: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"{task}_events.csv"
    summary_path = out_dir / f"{task}_summary.json"
    if events:
        with open(events_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[x.name for x in fields(events[0])])
            writer.writeheader()
            writer.writerows(asdict(e) for e in events)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {len(events)} events to {events_path} and {len(summaries)} summaries to {summary_path}")


# --------------------------------------------------------------------------- #
# WCST
# --------------------------------------------------------------------------- #


def wcst_variant(stem: str) -> str:
    for variant in ("card-random", "string", "empty", "image", "card"):
        if f"_{variant}_" in stem:
            return variant
    raise ValueError(f"Cannot infer WCST variant from {stem!r}")


def parse_card(text: str, background: bool) -> dict[str, str]:
    """Attributes of a text card, in canonical or (without background) shuffled order."""
    words = []
    for word in text.lower().split():
        if word.endswith("s") and word[:-1] in SHAPES:
            word = word[:-1]
        words.append(word)

    if background:
        # Color and background share vocabulary, so only canonical order is parseable
        if (
            len(words) != 4
            or words[0] not in NUMBER_WORDS.values()
            or words[1] not in COLORS
            or words[2] not in SHAPES
            or words[3] not in COLORS
        ):
            raise ValueError(f"Unparseable background card: {text!r}")
        return dict(zip(("number", "color", "shape", "background"), words))

    card: dict[str, str] = {}
    for word in words:
        if word in NUMBER_WORDS.values():
            key = "number"
        elif word in COLORS:
            key = "color"
        elif word in SHAPES:
            key = "shape"
        else:
            raise ValueError(f"Unparseable card: {text!r}")
        if key in card:
            raise ValueError(f"Duplicate {key} in card: {text!r}")
        card[key] = word
    if len(card) != 3:
        raise ValueError(f"Incomplete card: {text!r}")
    return card


def _string_attrs(s: str) -> dict[str, int]:
    vowels = sum(c in "aeiouAEIOU" for c in s)
    return {"length": len(s), "vowels": vowels, "consonants": len(s) - vowels}


def parse_question(question: str) -> tuple[str, list[str]]:
    given = re.search(r"Given:\s*(.+)", question)
    options = re.findall(r"^\d\.\s*(.+)$", question, re.M)
    if given is None or len(options) != 4:
        raise ValueError(f"Unparseable WCST question: {question!r}")
    return given.group(1).strip(), [o.strip() for o in options]


def has_background(row: dict, variant: str) -> bool:
    if variant == "image":
        return "background" in row["given_card"]
    if variant in ("card", "card-random"):
        return len(parse_question(row["question"])[0].split()) == 4
    return False


def answer_index(model_ans: Any) -> Optional[int]:
    try:
        idx = int(str(model_ans).strip())
    except ValueError:
        return None
    return idx if 1 <= idx <= 4 else None


def matched_attributes(
    row: dict, variant: str, idx: int, background: bool
) -> Optional[frozenset[str]]:
    """Attributes the chosen option shares with the given card."""
    if variant == "empty":
        return None
    if variant == "image":
        raw = row["given_card"]
        given = {
            "number": NUMBER_WORDS[raw["count"]],
            "color": raw["color"],
            "shape": raw["shape"],
        }
        if background:
            given["background"] = raw["background"]
        option = IMAGE_OPTIONS[idx - 1]
        return frozenset(k for k in given if given[k] == option[k])

    given_text, options = parse_question(row["question"])
    if variant == "string":
        given_s, option_s = _string_attrs(given_text), _string_attrs(options[idx - 1])
        return frozenset(k for k in given_s if given_s[k] == option_s[k])

    given = parse_card(given_text, background)
    option = parse_card(options[idx - 1], background)
    return frozenset(k for k in given if given[k] == option[k])


def stated_rule(response: Optional[str], patterns: dict[str, str]) -> Optional[str]:
    """The single rule named in the post-reasoning summary, else None.

    Heuristic: negated mentions ("not color") still count as mentions.
    """
    if not response:
        return None
    text = re.split(r"</think(?:ing)?>", response)[-1]
    text = re.sub(r"<answer>.*?</answer>", "", text, flags=re.S)
    hits = {rule for rule, p in patterns.items() if re.search(p, text, re.I)}
    return hits.pop() if len(hits) == 1 else None


def classify_wcst_run(
    rows: Sequence[dict],
    variant: str,
    file: str = "",
    run: str = "",
    acquisition: int = 3,
) -> list[WcstEvent]:
    background = bool(rows) and has_background(rows[0], variant)
    if variant == "string":
        patterns = STRING_RULE_PATTERNS
    elif variant == "empty":
        patterns = {}
    else:
        patterns = dict(CARD_RULE_PATTERNS)
        if background:
            patterns["background"] = BACKGROUND_PATTERN

    events = []
    prev_rule = cur_rule = None
    segment, streak, acquired, eliminated = -1, 0, False, set()

    for turn, row in enumerate(rows, 1):
        rule = str(row["rule"])
        segment_start = rule != cur_rule
        if segment_start:
            prev_rule, cur_rule = cur_rule, rule
            segment, streak, acquired, eliminated = segment + 1, 0, False, set()

        correct = bool(row["correct"])
        idx = answer_index(row.get("model_ans"))
        matched = (
            matched_attributes(row, variant, idx, background) if idx is not None else None
        )

        if correct:
            label = "correct"
        elif idx is None:
            label = "format"
        elif matched is None:
            label = "unclassified"
        elif segment_start and prev_rule in matched:
            label = "switch"
        elif acquired:
            label = "set_loss"
        elif prev_rule is not None and prev_rule in matched:
            label = "perseverative"
        elif not matched:
            label = "no_match"
        elif matched <= eliminated:
            label = "non_update"
        else:
            label = "exploratory"

        stated = stated_rule(row.get("response"), patterns)
        events.append(
            WcstEvent(
                file=file,
                run=run,
                turn=turn,
                rule=rule,
                segment=segment,
                correct=correct,
                label=label,
                unexpected=label in WCST_UNEXPECTED,
                matched="|".join(sorted(matched)) if matched is not None else None,
                model_ans=None if row.get("model_ans") is None else str(row["model_ans"]),
                true_ans=None if row.get("true_ans") is None else str(row["true_ans"]),
                stated_rule=stated,
                dissociation=(
                    stated not in matched
                    if stated is not None and matched is not None
                    else None
                ),
            )
        )

        if correct:
            streak += 1
            acquired = acquired or streak >= acquisition
        else:
            streak = 0
            if matched:
                eliminated |= matched

    return events


def wcst_extras(events: Sequence[WcstEvent]) -> dict:
    """Knowing-doing dissociation and answer-position bias."""
    stated = [e for e in events if e.dissociation is not None]
    errors = [e for e in events if e.unexpected]
    return {
        "dissociation": {
            "n_stated": len(stated),
            "rate": sum(e.dissociation for e in stated) / len(stated) if stated else None,
        },
        "positions": {
            "true": dict(Counter(e.true_ans for e in events)),
            "chosen": dict(Counter(e.model_ans for e in events)),
            "chosen_on_unexpected": dict(Counter(e.model_ans for e in errors)),
        },
    }


def analyze_wcst_file(path: Path, acquisition: int) -> list[WcstEvent]:
    variant = wcst_variant(path.stem)
    usage_path = path.with_name(path.stem + "_usage.json")
    history_path = path.with_name(path.stem + "_history.json")
    events = []
    for run, rows in _load_json(path).items():
        run_events = classify_wcst_run(rows, variant, str(path), run, acquisition)
        tokens, chars = _depth_columns(usage_path, history_path, run, len(rows), False)
        for event, t, c in zip(run_events, tokens, chars):
            event.prompt_tokens, event.context_chars = t, c
        events.extend(run_events)
    return events


def _is_wcst_result(path: Path) -> bool:
    return not re.search(r"_(history|reasoning|usage)$", path.stem)


# --------------------------------------------------------------------------- #
# SWM
# --------------------------------------------------------------------------- #


def _box_key(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _token_positions(
    token_box: Sequence[Any], contained: Sequence[set], n_boxes: int
) -> dict[int, Any]:
    n_tokens = len(contained)
    if len(token_box) == n_tokens:
        token_ids: Sequence[int] = range(n_tokens)
    else:
        # Logged list drops token types completed in earlier searches
        token_ids = [t for t in range(n_tokens) if len(contained[t]) < n_boxes]
        if len(token_ids) != len(token_box):
            raise ValueError(
                f"token_box {token_box} doesn't match {len(token_ids)} active token types"
            )
    return {t: _box_key(p) for t, p in zip(token_ids, token_box) if p is not None}


def replay_swm_run(
    entries: Sequence[dict],
    n_boxes: int,
    n_tokens: int,
    file: str = "",
    run: str = "",
) -> list[SwmEvent]:
    """Re-derive per-turn SWM error labels from the structured history."""
    contained: list[set] = [set() for _ in range(n_tokens)]
    opened: set = set()
    search = 0
    events = []

    for turn, entry in enumerate(entries, 1):
        status = entry["status"]
        box = _box_key(entry["chosen_box"] if "chosen_box" in entry else entry.get("chosen_coord"))
        search_before = search
        illegal_boxes = set.intersection(*contained)
        p_chance = len(illegal_boxes | opened) / n_boxes

        if status == "invalid":
            label = "format"
        elif status == "nobox":
            label = "nobox"
        elif status in ("valid", "box"):
            in_grid = not isinstance(box, int) or 1 <= box <= n_boxes
            if not in_grid or box in illegal_boxes:
                label = "illegal"
            elif box in opened:
                label = "repeated"
            else:
                label = "found" if entry["found"] else "empty"
            if entry["found"] and label != "found":
                raise ValueError(f"{run} turn {turn}: token found in a {label} box")

            if in_grid:
                opened.add(box)
            if entry["found"]:
                positions = _token_positions(entry["token_box"], contained, n_boxes)
                hits = [t for t, pos in positions.items() if pos == box]
                if not hits:
                    raise ValueError(f"{run} turn {turn}: found=True but no token at {box}")
                for t in hits:
                    contained[t].add(box)
                opened = set()
                search += 1
        else:
            raise ValueError(f"{run} turn {turn}: unknown SWM status {status!r}")

        events.append(
            SwmEvent(
                file=file,
                run=run,
                turn=turn,
                search=search_before,
                label=label,
                unexpected=label in SWM_UNEXPECTED,
                choice=None if box is None else str(box),
                p_error_chance=p_chance,
            )
        )
    return events


def swm_extras(events: Sequence[SwmEvent], bin_width: int) -> dict:
    """Memory-error rate vs. random-pick baseline, by turn depth."""
    bins: dict[int, list] = defaultdict(list)
    for e in events:
        if e.label in ("found", "empty", "illegal", "repeated"):
            bins[(e.turn - 1) // bin_width].append(e)
    rows = []
    for b, group in sorted(bins.items()):
        rate = sum(e.label in ("illegal", "repeated") for e in group) / len(group)
        chance = sum(e.p_error_chance for e in group) / len(group)
        rows.append(
            {
                "turns": [b * bin_width + 1, (b + 1) * bin_width],
                "n": len(group),
                "memory_error_rate": rate,
                "chance_rate": chance,
                "excess_over_chance": rate - chance,
            }
        )
    return {"memory_vs_chance_by_depth": rows}


def analyze_swm_file(path: Path) -> list[SwmEvent]:
    match = re.search(r"_(\d+)_(\d+)_run_structured_history$", path.stem)
    if match is None:
        raise ValueError(f"Cannot infer n_boxes/n_tokens from {path.name}")
    n_boxes, n_tokens = int(match.group(1)), int(match.group(2))
    prefix = path.name[: -len("run_structured_history.json")]
    usage_path = path.with_name(prefix + "run_usage.json")
    history_path = path.with_name(prefix + "run_history.json")

    events = []
    for run, entries in _load_json(path).items():
        run_events = replay_swm_run(entries, n_boxes, n_tokens, str(path), run)
        # The last response may be unevaluated, so one extra call is allowed
        tokens, chars = _depth_columns(usage_path, history_path, run, len(entries), True)
        for event, t, c in zip(run_events, tokens, chars):
            event.prompt_tokens, event.context_chars = t, c
        events.extend(run_events)
    return events


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("task", choices=["wcst", "swm"])
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/failure_analysis"))
    parser.add_argument("--bin_width", type=int, default=10, help="Turns per depth bin")
    parser.add_argument(
        "--acquisition", type=int, default=3, help="WCST: consecutive correct to count a rule as acquired"
    )
    args = parser.parse_args()

    if args.task == "wcst":
        paths = [p for p in sorted(args.data_dir.rglob("*.json")) if _is_wcst_result(p)]
    else:
        paths = sorted(args.data_dir.rglob("*_run_structured_history.json"))

    all_events, summaries = [], {}
    for path in paths:
        print(f"Processing {path}")
        try:
            if args.task == "wcst":
                events = analyze_wcst_file(path, args.acquisition)
            else:
                events = analyze_swm_file(path)
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            print(f"  skipped: {e}")
            continue
        summary = summarize(events, args.bin_width)
        if args.task == "wcst":
            summary.update(wcst_extras(events))
        else:
            summary.update(swm_extras(events, args.bin_width))
        summaries[str(path)] = summary
        all_events.extend(events)

    write_outputs(args.task, all_events, summaries, args.out_dir)


if __name__ == "__main__":
    main()
