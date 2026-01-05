"""
Wisconsin Card Sorting Test (WCST) module.
"""

from .wcst import run_wcst, run_wcst_with_env
from .wcst_env import WCSTEnv, WCSTTrial

# Optional image generation (requires PIL)
try:
    from .image import draw_five_cards
    __all__ = ['run_wcst', 'run_wcst_with_env', 'WCSTEnv', 'WCSTTrial', 'draw_five_cards']
except ImportError:
    __all__ = ['run_wcst', 'run_wcst_with_env', 'WCSTEnv', 'WCSTTrial']
