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

    Turn-level rewards (4 functions) - SUMMED across all turns:
    1. turn_format_reward: +0.1 per valid format, -1.0 per invalid
    2. turn_choice_validity_reward: +0.1 per valid choice (1-4), -1.0 per out-of-range
    3. turn_repeat_reward: -1.0 per repeated INCORRECT choice in current rule cycle
    4. turn_correctness_reward: +1.0 per correct, -0.5 per failure-to-maintain-set,
       -0.5 per perseverative error

    State reconstruction: Functions 3-4 parse USER feedback messages in the trajectory
    to reconstruct per-turn state:
    - Track consecutive_correct for failure-to-maintain-set (error after 2+ correct)
    - Track rule changes (5 consecutive correct)
    - Track previous rule's correct choice for perseverative error detection

    Outcome-level rewards:
    - outcome_completed_categories: fraction of completed rule cycles
    - outcome_error_ratio: -(perseverative_errors + failures) / n_trials
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
        """Penalty for repeating a choice within the current rule cycle - summed across ALL turns.

        A repeated choice is only penalized when:
        1. The same option was chosen before in the current rule cycle
        2. The previous same choice was INCORRECT

        State reconstruction from trajectory:
        - Track seen_options_this_rule: set of choices made in current rule
        - Reset when we detect a rule change (5 consecutive correct)
        - Reset when a choice is correct (fresh exploration context per WCST rules)

        Per-turn rewards:
        - -1.0 for repeating a choice that was previously incorrect in current rule
        - 0.0 otherwise

        Returns sum of per-turn rewards for each trajectory.
        """
        rewards = []
        num_correct_for_rule_change = kwargs.get("num_correct", 5)

        for trajectory in completions:
            total_reward = 0.0

            # Track choices seen in current rule cycle that were INCORRECT
            seen_incorrect_choices: set = set()
            consecutive_correct = 0

            # Process trajectory turn by turn
            for i, msg in enumerate(trajectory):
                if msg["role"] != "assistant":
                    continue

                choice, status = self._parse_choice(msg["content"])
                if status != "valid" or choice is None:
                    consecutive_correct = 0  # Reset on invalid
                    continue

                # Look at the NEXT message (USER response) to determine outcome
                if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "user":
                    user_response = trajectory[i + 1]["content"].lower()

                    is_correct = "correct!" in user_response and "incorrect" not in user_response

                    # Check if this is a repeat of a previously incorrect choice
                    if choice in seen_incorrect_choices:
                        total_reward += REWARD_REPEATED  # -1.0

                    if is_correct:
                        consecutive_correct += 1
                        # Reset seen choices on correct (fresh exploration context)
                        seen_incorrect_choices = set()

                        # Check for rule change
                        if consecutive_correct >= num_correct_for_rule_change:
                            consecutive_correct = 0
                            seen_incorrect_choices = set()
                    else:
                        # Incorrect - add to seen choices, reset consecutive
                        seen_incorrect_choices.add(choice)
                        consecutive_correct = 0

            rewards.append(total_reward)

        return rewards

    def turn_correctness_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """Correctness reward with WCST-specific error penalties - summed across ALL turns.

        Per-turn rewards:
        - +1.0 for correct answer
        - 0.0 for incorrect answer (base)
        - -0.5 for failure-to-maintain-set (error after 2+ consecutive correct)
        - -0.5 for perseverative error (repeating previous "correct pattern" after rule change)

        State reconstruction from trajectory:
        - Track consecutive_correct count
        - Detect rule changes (5 consecutive correct)
        - Track last_correct_choice_before_rule_change for perseverative detection

        Note on perseverative error detection:
        Since we don't have explicit rule info in trajectory, we approximate by checking
        if the model repeats the same choice that was "working" (last correct) before
        the rule changed, and that choice is now incorrect.

        Returns sum of per-turn rewards for each trajectory.
        """
        rewards = []
        num_correct_for_rule_change = kwargs.get("num_correct", 5)

        for trajectory in completions:
            total_reward = 0.0

            consecutive_correct = 0
            last_correct_choice = None  # The last choice that was correct
            previous_rule_correct_choice = None  # Choice that was correct under previous rule
            rule_just_changed = False

            # Process trajectory turn by turn
            for i, msg in enumerate(trajectory):
                if msg["role"] != "assistant":
                    continue

                choice, status = self._parse_choice(msg["content"])
                if status != "valid" or choice is None:
                    consecutive_correct = 0
                    continue

                # Look at the NEXT message (USER response) to determine outcome
                if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "user":
                    user_response = trajectory[i + 1]["content"].lower()

                    is_correct = "correct!" in user_response and "incorrect" not in user_response

                    if is_correct:
                        total_reward += REWARD_CORRECT  # +1.0
                        consecutive_correct += 1
                        last_correct_choice = choice
                        rule_just_changed = False

                        # Check for rule change
                        if consecutive_correct >= num_correct_for_rule_change:
                            # Rule is about to change
                            previous_rule_correct_choice = last_correct_choice
                            consecutive_correct = 0
                            rule_just_changed = True
                    else:
                        # Incorrect answer - check for WCST-specific errors

                        # Failure to Maintain Set: error after 2+ consecutive correct
                        if consecutive_correct >= 2:
                            total_reward += REWARD_FAILURE_TO_MAINTAIN_SET  # -0.5

                        # Perseverative Error: repeating previous rule's correct choice after rule change
                        # Skip penalty on the FIRST trial after rule change (model hasn't had feedback yet)
                        if (
                            previous_rule_correct_choice is not None
                            and choice == previous_rule_correct_choice
                            and not rule_just_changed
                        ):
                            total_reward += REWARD_PERSEVERATIVE_ERROR  # -0.5

                        consecutive_correct = 0
                        rule_just_changed = False

            rewards.append(total_reward)

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
