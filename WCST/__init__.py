"""Wisconsin Card Sorting Test (WCST) module."""

from .wcst import run_wcst, run_wcst_with_env
from .wcst_mt_env import load_environment, WCSTMultiTurnEnv, create_wcst_dataset
from .wcst_rubric import WCSTRubric
from .utils import wcst_generator, string_generator, check_rule_ambiguity

# Optional image generation (requires PIL)
try:
    from .image import draw_five_cards

    __all__ = [
        "run_wcst",
        "run_wcst_with_env",
        "load_environment",
        "WCSTMultiTurnEnv",
        "create_wcst_dataset",
        "WCSTRubric",
        "wcst_generator",
        "string_generator",
        "check_rule_ambiguity",
        "draw_five_cards",
    ]
except ImportError:
    __all__ = [
        "run_wcst",
        "run_wcst_with_env",
        "load_environment",
        "WCSTMultiTurnEnv",
        "create_wcst_dataset",
        "WCSTRubric",
        "wcst_generator",
        "string_generator",
        "check_rule_ambiguity",
    ]
