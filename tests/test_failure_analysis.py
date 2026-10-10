import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from failure_analysis import (  # noqa: E402
    align,
    analyze_swm_file,
    analyze_wcst_file,
    classify_wcst_run,
    context_chars,
    kaplan_meier,
    matched_attributes,
    parse_card,
    replay_swm_run,
    summarize,
    wcst_extras,
)

# Each option shares exactly one attribute with the given card, except option 4
QUESTION = (
    "Given: two red star\nOptions:\n"
    "1. one red circle\n2. two green triangle\n3. three blue star\n4. four yellow square"
)
TRUE_ANS = {"color": 1, "shape": 3}


def _row(rule, ans, response=None):
    return {
        "rule": rule,
        "correct": str(ans) == str(TRUE_ANS[rule]),
        "question": QUESTION,
        "response": response or f"<answer>{ans}</answer>",
        "model_ans": str(ans),
        "true_ans": TRUE_ANS[rule],
    }


WCST_ROWS = [
    _row("color", 2),  # exploratory
    _row("color", 2),  # non_update: number already disconfirmed
    _row("color", 4),  # no_match
    _row("color", "x"),  # format
    _row("color", 1),
    _row("color", 1),
    _row("color", 1),  # acquired
    _row("color", 3),  # set_loss
    _row("color", 1),
    _row("shape", 1),  # switch
    _row("shape", 1, "<think>hmm</think> Trying shape now. <answer>1</answer>"),  # perseverative
    _row("shape", 3, "<think>x</think> Sorting by shape. <answer>3</answer>"),
]


def test_wcst_labels():
    events = classify_wcst_run(WCST_ROWS, "card")
    assert [e.label for e in events] == [
        "exploratory",
        "non_update",
        "no_match",
        "format",
        "correct",
        "correct",
        "correct",
        "set_loss",
        "correct",
        "switch",
        "perseverative",
        "correct",
    ]
    assert [e.segment for e in events] == [0] * 9 + [1] * 3
    assert not events[0].unexpected and not events[9].unexpected
    assert events[1].unexpected and events[7].unexpected


def test_wcst_dissociation_and_positions():
    events = classify_wcst_run(WCST_ROWS, "card")
    assert events[10].stated_rule == "shape" and events[10].dissociation is True
    assert events[11].stated_rule == "shape" and events[11].dissociation is False
    assert events[0].stated_rule is None and events[0].dissociation is None

    extras = wcst_extras(events)
    assert extras["dissociation"] == {"n_stated": 2, "rate": 0.5}
    assert extras["positions"]["chosen_on_unexpected"] == {"2": 1, "4": 1, "x": 1, "3": 1, "1": 1}


def test_parse_card_shuffled_and_plural():
    assert parse_card("star two red", False) == {"number": "two", "color": "red", "shape": "star"}
    assert parse_card("two red stars", False)["shape"] == "star"
    assert parse_card("two red star blue", True)["background"] == "blue"
    with pytest.raises(ValueError):
        parse_card("red two star blue", True)  # shuffled background card is ambiguous


def test_matched_attributes_image_and_string():
    image_row = {"given_card": {"count": 3, "color": "red", "shape": "star", "background": "blue"}}
    assert matched_attributes(image_row, "image", 3, True) == {"number", "shape", "background"}

    string_row = {"question": "Given: abc\nOptions:\n1. xyz\n2. ae\n3. bac\n4. q"}
    assert matched_attributes(string_row, "string", 1, False) == {"length"}
    assert matched_attributes(string_row, "string", 3, False) == {"length", "vowels", "consonants"}


def test_swm_replay_text():
    entries = [
        {"token_box": [2], "chosen_box": 1, "found": False, "status": "valid"},
        {"token_box": [2], "chosen_box": 1, "found": False, "status": "valid"},
        {"token_box": [2], "chosen_box": None, "found": False, "status": "invalid"},
        {"token_box": [2], "chosen_box": 2, "found": True, "status": "valid"},
        {"token_box": [3], "chosen_box": 2, "found": False, "status": "valid"},
        {"token_box": [3], "chosen_box": 5, "found": False, "status": "valid"},
        {"token_box": [3], "chosen_box": 3, "found": True, "status": "valid"},
    ]
    events = replay_swm_run(entries, n_boxes=3, n_tokens=1)
    assert [e.label for e in events] == [
        "empty", "repeated", "format", "found", "illegal", "illegal", "found"
    ]
    assert [e.search for e in events] == [0, 0, 0, 0, 1, 1, 1]
    assert [round(e.p_error_chance, 3) for e in events] == [0, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333]


def test_swm_replay_multi_token_image_with_dropped_tokens():
    entries = [
        {"token_box": [[0, 0], [1, 1]], "chosen_coord": [0, 0], "found": True, "status": "box"},
        {"token_box": [[1, 1], [1, 1]], "chosen_coord": [1, 1], "found": True, "status": "box"},
        # Token type A is complete, so only B's position is logged
        {"token_box": [[0, 0]], "chosen_coord": [1, 1], "found": False, "status": "box"},
        {"token_box": [[0, 0]], "chosen_coord": [0, 0], "found": True, "status": "box"},
    ]
    events = replay_swm_run(entries, n_boxes=2, n_tokens=2)
    assert [e.label for e in events] == ["found", "found", "illegal", "found"]


def test_swm_replay_rejects_inconsistent_log():
    entries = [
        {"token_box": [1], "chosen_box": 1, "found": True, "status": "valid"},
        {"token_box": [1], "chosen_box": 1, "found": True, "status": "valid"},
    ]
    with pytest.raises(ValueError):
        replay_swm_run(entries, n_boxes=2, n_tokens=1)


def test_kaplan_meier():
    curve = kaplan_meier([2, 3, 3, 5], [True, True, False, True])
    assert curve == [(2, 0.75), (3, 0.5), (5, 0.0)]


def test_align_and_context_chars():
    assert align([1, 2, 3], 3, False) == [1, 2, 3]
    assert align([1, 2, 3, 4], 3, True) == [1, 2, 3]
    assert align([1, 2, 3, 4], 3, False) is None
    messages = [
        {"role": "system", "content": "abc"},
        {"role": "user", "content": "de"},
        {"role": "assistant", "content": "f"},
        {"role": "user", "content": [{"type": "text", "text": "gh"}, {"type": "image_url"}]},
    ]
    assert context_chars(messages) == [5, 8]
    assert context_chars(messages, stateless=True) == [5, 5]


def test_summarize_first_failure():
    events = classify_wcst_run(WCST_ROWS, "card", run="run_1")
    summary = summarize(events, bin_width=5)
    assert summary["first_failure"]["turns"] == [2]
    assert summary["first_failure"]["labels"] == {"non_update": 1}
    assert [b["n"] for b in summary["by_depth"]] == [5, 5, 2]


def test_analyze_files_attach_depth(tmp_path, capsys):
    wcst_path = tmp_path / "openrouter_m_card_64-5_off.json"
    wcst_path.write_text(json.dumps({"run_1": WCST_ROWS[:2], "run_2": WCST_ROWS[:2]}))
    calls = [{"prompt_tokens": 10}, {"prompt_tokens": 20}]
    (tmp_path / "openrouter_m_card_64-5_off_usage.json").write_text(
        json.dumps({"run_1": {"calls": calls}, "run_2": {"calls": calls[:1]}})
    )
    events = analyze_wcst_file(wcst_path, acquisition=3)
    assert [e.prompt_tokens for e in events] == [10, 20, None, None]
    assert "run_2" in capsys.readouterr().out

    swm_path = tmp_path / "openrouter_m_3_1_run_structured_history.json"
    swm_path.write_text(json.dumps({"run_1": [
        {"token_box": [2], "chosen_box": 1, "found": False, "status": "valid"},
    ]}))
    history = [{"role": "system", "content": "ab"}, {"role": "user", "content": "c"},
               {"role": "assistant", "content": "d"}, {"role": "user", "content": "e"}]
    (tmp_path / "openrouter_m_3_1_run_history.json").write_text(json.dumps({"run_1": history}))
    events = analyze_swm_file(swm_path)
    assert events[0].context_chars == 3
