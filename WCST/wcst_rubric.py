"""
WCST Rubric for multi-turn RL training.
Defines turn-level and outcome-level reward functions compatible with TRL.
"""

import os
import sys
import re
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from shared.rubrics import Rubric
    from shared.parsers import XMLParser
except ImportError:
    # Fallback - define minimal base classes
    class Rubric:
        def __init__(self):
            self.turn_reward_funcs = []
            self.outcome_reward_funcs = []

    class XMLParser:
        def __init__(self, fields=None):
            self.fields = fields or []


# Reward constants (matching MT-GRPO paper reward design)
# Turn-level: +0.1 for correct format (encourage exploration with valid format)
# Outcome: +0.2 for correct format but wrong answer
REWARD_CORRECT = 1.0
REWARD_INCORRECT = 0.0
REWARD_VALID_FORMAT = 0.1  # Small positive for correct format (MT-GRPO Section 5.2)
REWARD_VALID_CHOICE = 0.1  # Small positive for valid choice range
REWARD_INVALID_FORMAT = -1.0
REWARD_INVALID_CHOICE = -1.0
REWARD_PERSEVERATIVE_ERROR = -0.5
REWARD_FAILURE_TO_MAINTAIN_SET = -0.5
REWARD_REPEATED = -1.0


class WCSTRubric(Rubric):
    """
    Rubric for Wisconsin Card Sorting Test.

    Supports:
    - Turn-level rewards: per-choice correctness and format
    - Outcome rewards: final accuracy and rule detection
    """

    def __init__(self):
        """Initialize with XML parser for structured outputs."""
        super().__init__()
        self.parser = XMLParser(fields=["thinking", "choice"])

    def _parse_choice(self, text: str) -> tuple:
        """Parse choice (1-4) from <answer> tags. Returns (choice, status)."""
        if not text:
            return None, "invalid_format"

        # Try to find <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            return None, "invalid_format"

        answer_text = match.group(1).strip()
        nums = re.findall(r"\d+", answer_text)
        if not nums:
            return None, "invalid_format"

        try:
            choice = int(nums[0])
            if 1 <= choice <= 4:
                return choice, "valid"
            else:
                return None, "invalid_choice"
        except ValueError:
            return None, "invalid_format"

    # ========== TURN-LEVEL REWARDS ==========

    def turn_format_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """Reward for format compliance - summed across ALL turns.

        Per-turn rewards:
        - +0.1 for each turn with valid <answer> tags
        - -1.0 for each turn with missing/malformed tags

        Returns sum of per-turn rewards for each trajectory.
        """
        rewards = []
        for trajectory in completions:
            total_reward = 0.0
            has_assistant = False

            for msg in trajectory:
                if msg["role"] == "assistant":
                    has_assistant = True
                    _, status = self._parse_choice(msg["content"])
                    if status == "invalid_format":
                        total_reward += REWARD_INVALID_FORMAT  # -1.0
                    else:
                        total_reward += REWARD_VALID_FORMAT  # +0.1

            if not has_assistant:
                rewards.append(REWARD_INVALID_FORMAT)
            else:
                rewards.append(total_reward)

        return rewards

    def turn_choice_validity_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """Reward for valid choice range (1-4) - summed across ALL turns.

        Per-turn rewards:
        - +0.1 for each turn with valid choice (1-4)
        - -1.0 for each turn with out-of-range choice
        - 0.0 for format errors (handled by turn_format_reward)

        Returns sum of per-turn rewards for each trajectory.
        """
        rewards = []
        for trajectory in completions:
            total_reward = 0.0

            for msg in trajectory:
                if msg["role"] == "assistant":
                    _, status = self._parse_choice(msg["content"])
                    if status == "valid":
                        total_reward += REWARD_VALID_CHOICE  # +0.1
                    elif status == "invalid_choice":
                        total_reward += REWARD_INVALID_CHOICE  # -1.0
                    # status == "invalid_format" -> 0.0 (handled by format reward)

            rewards.append(total_reward)

        return rewards

    def turn_repeat_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """Penalty for repeating a choice within the current rule/trial.

        Expects `seen_options` (set) or `prev_choices_in_trial` (list) in kwargs.
        Returns -1.0 if repeated, else 0.0.
        """
        rewards = []
        seen_options = kwargs.get("seen_options", set())
        prev_choices = kwargs.get("prev_choices_in_trial", [])

        for idx, trajectory in enumerate(completions):
            last_assistant = None
            for msg in reversed(trajectory):
                if msg["role"] == "assistant":
                    last_assistant = msg["content"]
                    break

            if not last_assistant:
                rewards.append(0.0)
                continue

            choice, status = self._parse_choice(last_assistant)
            if status != "valid" or choice is None:
                rewards.append(0.0)
                continue

            # Allow per-example state if lists are provided
            if isinstance(seen_options, list):
                seen_set = seen_options[idx] if idx < len(seen_options) else set()
            else:
                seen_set = seen_options

            if (
                isinstance(prev_choices, list)
                and prev_choices
                and isinstance(prev_choices[0], list)
            ):
                prev_list = prev_choices[idx] if idx < len(prev_choices) else []
            else:
                prev_list = prev_choices

            already_seen = choice in seen_set or choice in prev_list
            rewards.append(REWARD_REPEATED if already_seen else 0.0)

        return rewards

    def turn_correctness_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """Base correctness reward plus WCST-specific penalties.

        - +1.0 if correct
        - 0.0 if incorrect (base), with additional penalties:
            * -0.5 failure-to-maintain-set if consecutive_correct >= 2 (state) or flag provided
            * -0.5 perseverative error if flagged

        Expects either `answer` list of correct indices or `correct_answers` kwarg.
        Also consumes state/flags: consecutive_correct, is_failure_to_maintain, is_perseverative_error.
        """
        rewards = []
        correct_answers = kwargs.get("correct_answers", answer)
        consecutive_correct_state = kwargs.get("consecutive_correct", 0)
        is_failure_flag = kwargs.get("is_failure_to_maintain", False)
        is_persev_flag = kwargs.get("is_perseverative_error", False)

        # Handle None or missing correct_answers
        if correct_answers is None:
            correct_answers = [None] * len(completions)
        elif not isinstance(correct_answers, (list, tuple)):
            correct_answers = [correct_answers] * len(completions)

        # Ensure correct_answers matches completions length
        if len(correct_answers) < len(completions):
            correct_answers = list(correct_answers) + [None] * (
                len(completions) - len(correct_answers)
            )

        for idx, (trajectory, expected) in enumerate(zip(completions, correct_answers)):
            # Skip if no expected answer
            if expected is None:
                rewards.append(0.0)
                continue

            last_assistant = None
            for msg in reversed(trajectory):
                if msg["role"] == "assistant":
                    last_assistant = msg["content"]
                    break

            if not last_assistant:
                rewards.append(0.0)
                continue

            choice, status = self._parse_choice(last_assistant)
            if status != "valid" or choice is None:
                rewards.append(0.0)
                continue

            try:
                expected_int = int(expected)
            except (ValueError, TypeError):
                rewards.append(0.0)
                continue

            if choice == expected_int:
                rewards.append(REWARD_CORRECT)
                continue

            # Incorrect path
            reward = REWARD_INCORRECT

            # Failure to maintain set: env uses >=2 consecutive correct before the error
            if isinstance(consecutive_correct_state, list):
                cc_val = (
                    consecutive_correct_state[idx]
                    if idx < len(consecutive_correct_state)
                    else 0
                )
            else:
                cc_val = consecutive_correct_state

            failure_flag = (
                is_failure_flag[idx]
                if isinstance(is_failure_flag, list) and idx < len(is_failure_flag)
                else is_failure_flag
            )
            persev_flag = (
                is_persev_flag[idx]
                if isinstance(is_persev_flag, list) and idx < len(is_persev_flag)
                else is_persev_flag
            )

            if failure_flag or cc_val >= 2:
                reward += REWARD_FAILURE_TO_MAINTAIN_SET

            # Perseverative error: rely on flag passed from state (env should detect)
            if persev_flag:
                reward += REWARD_PERSEVERATIVE_ERROR

            rewards.append(reward)

        return rewards

    # ========== OUTCOME REWARDS ==========

    def outcome_completed_categories(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """Outcome reward: fraction of fully completed categories/rules.

        Uses state: completed_categories, rules (to infer total categories = 2 full cycles).
        Returns value in [0.0, 1.0].
        """
        completed = kwargs.get("completed_categories", 0)
        rules = kwargs.get("rules", [])
        num_rules = len(rules) if rules else kwargs.get("num_rules", 0)
        total_categories = max(1, num_rules * 2)  # two full cycles to finish task

        reward = min(1.0, completed / total_categories) if total_categories else 0.0
        return [reward] * len(completions)

    def outcome_error_ratio(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """Outcome penalty: -(perseverative_errors + failure_to_maintain_set) / n_trials.

        Returns range [-1.0, 0.0].
        """
        perseverative_errors = kwargs.get("perseverative_errors", 0)
        failure_to_maintain = kwargs.get("failure_to_maintain_set", 0)
        n_trials = kwargs.get("n_trials", 0)

        if n_trials == 0:
            ratio = 0.0
        else:
            ratio = (perseverative_errors + failure_to_maintain) / n_trials

        return [-ratio] * len(completions)

    @property
    def turn_reward_funcs(self) -> List:
        """
        Turn-level reward functions.
        Called after each step to provide immediate feedback.
        """
        return [
            self.turn_format_reward,
            self.turn_choice_validity_reward,
            self.turn_repeat_reward,
            self.turn_correctness_reward,
        ]

    @property
    def outcome_reward_funcs(self) -> List:
        """
        Outcome-level reward functions.
        Called at episode end to evaluate overall performance.
        """
        return [
            self.outcome_completed_categories,
            self.outcome_error_ratio,
        ]
