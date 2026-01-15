"""
Shared utilities for CognitiveEval package.

This module provides:
- Environment classes for RL training (Environment, MultiTurnEnv)
- Trainer classes (GRPOEnvTrainer, MTGRPOEnvTrainer)
- Rubric base class for reward computation
- XMLParser for parsing model outputs
- Model utilities for loading models and tokenizers
- Configuration utilities for GRPO training
- ModelWrapper for API-based model interaction
- CognitiveEnv base class for cognitive task environments
"""

# Base environment classes (for evaluation/API-based testing)
from .model_wrapper import ModelWrapper, encode_image_to_base64
from .base_env import CognitiveEnv, StepResult, ActionStatus, EnvState

# RL training environment classes
from .envs import Environment, MultiTurnEnv

# Trainers
from .trainers import GRPOEnvTrainer, MTGRPOEnvTrainer

# Rubrics and parsers
from .rubrics import Rubric, equals_reward_func
from .parsers import XMLParser

# Model and config utilities
from .model_utils import get_model, get_tokenizer, get_model_and_tokenizer
from .config import get_default_grpo_config

__all__ = [
    # API/Evaluation utilities
    "ModelWrapper",
    "encode_image_to_base64",
    "CognitiveEnv",
    "StepResult",
    "ActionStatus",
    "EnvState",
    # RL Environment classes
    "Environment",
    "MultiTurnEnv",
    # Trainers
    "GRPOEnvTrainer",
    "MTGRPOEnvTrainer",
    # Rubrics and parsers
    "Rubric",
    "equals_reward_func",
    "XMLParser",
    # Model utilities
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "get_default_grpo_config",
]
