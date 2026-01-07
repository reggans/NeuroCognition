"""
Raven's Progressive Matrices (RAPM) module.
"""

from .rapm_evaluation import run_rapm_evaluation, run_text_rapm_evaluation
from .rapm_env import RAPMEnv, RAPMQuestion, RAPMDatasetEnv

__all__ = [
    'run_rapm_evaluation',
    'run_text_rapm_evaluation',
    'RAPMEnv',
    'RAPMQuestion',
    'RAPMDatasetEnv',
]
