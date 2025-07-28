"""
CognitiveEval: A package for running cognitive evaluation tests.

This package includes:
- Wisconsin Card Sorting Test (WCST)
- Spatial Working Memory (SWM) test
"""

__version__ = "1.0.0"
__author__ = "CognitiveEval Team"

# Import main functions for easy access
from .WCST.wcst import run_wcst
from .SWM.main import swm_main

__all__ = ['run_wcst', 'swm_main']