import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swm_analysis import CATEGORY_ORDER, setup_category  # noqa: E402
from wcst_analysis import _normalize_category_for_hard_easy  # noqa: E402


@pytest.mark.parametrize(
    "category, expected",
    [
        ("cot", "cot_easy"),
        ("cot_bg_first", "cot_hard"),
        ("cot_notes", "cot_easy_notes"),
        ("cot_stateless", "cot_easy_stateless"),
        ("few_shot_cot_bg_rest_notes_stateless", "few_shot_cot_hard_notes_stateless"),
    ],
)
def test_wcst_hard_easy_keeps_condition_suffixes(category, expected):
    assert _normalize_category_for_hard_easy(category) == expected


@pytest.mark.parametrize(
    "stem, is_cot, expected",
    [
        ("openrouter_m_6_1_run_stats", False, "non_cot"),
        ("openrouter_m_cot_6_1_run_stats", True, "cot"),
        ("openrouter_m_cot_stateless_6_1_run_stats", True, "cot-stateless"),
        ("openrouter_m_notes_stateless_6_1_run_stats", False, "non_cot-notes-stateless"),
    ],
)
def test_swm_setup_category(stem, is_cot, expected):
    category = setup_category(stem, is_cot)
    assert category == expected
    assert category in CATEGORY_ORDER
