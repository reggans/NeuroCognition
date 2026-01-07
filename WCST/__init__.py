"""
Wisconsin Card Sorting Test (WCST) module.
"""

from .wcst import run_wcst
from .wcst_env import WCSTEnv, WCSTTrial

# Optional image generation (requires PIL)
try:
    from .image import draw_five_cards
    __all__ = ['run_wcst', 'WCSTEnv', 'WCSTTrial', 'draw_five_cards']
except ImportError:
    __all__ = ['run_wcst', 'WCSTEnv', 'WCSTTrial']
