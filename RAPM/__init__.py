"""
Raven's Progressive Matrices (RAPM) module.
"""

from .rapm_evaluation import run_rapm_evaluation, run_text_rapm_evaluation, run_rapm_with_env
from .rapm_env import RAPMEnv, RAPMQuestion, RAPMDatasetEnv

__all__ = [
    'run_rapm_evaluation',
    'run_text_rapm_evaluation',
    'run_rapm_with_env',
    'RAPMEnv',
    'RAPMQuestion',
    'RAPMDatasetEnv',
]
