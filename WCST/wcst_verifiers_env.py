"""
Verifiers environment for WCST (Wisconsin Card Sorting Test).
Mirrors the RL reward design from wcst_env.py without modifying it.

Supports:
- Variants: card, card-random, card-image, string, empty
- Rewards: +1.0 correct; -1.0 invalid; -0.5 perseverative; -0.5 FMS; -1.0 repeated
"""

import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import Dataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WCST.utils import (
    check_rule_ambiguity,
    string_generator,
    wcst_generator,
)
from WCST.wcst_rubric import WCSTRubric

try:
    import verifiers as vf
except ImportError:
    vf = None  # type: ignore

# Reward constants (matching wcst_env.py)
REWARD_CORRECT = 1.0
REWARD_INVALID_FORMAT = -1.0
REWARD_INVALID_CHOICE = -1.0
REWARD_PERSEVERATIVE_ERROR = -0.5
REWARD_FAILURE_TO_MAINTAIN_SET = -0.5
REWARD_REPEATED = -1.0
RULE_CHANGE_THRESHOLD = 5  # Align with wcst_env num_correct default

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Card WCST rules
CARD_RULES = ["number", "color", "shape", "background"]
# String WCST rules
STRING_RULES = ["length", "vowels", "consonants"]


@dataclass
class WCSTTrial:
    """Single WCST trial."""

    given: str
    options: List[str]
    rule: str
    correct: int  # 1-indexed
    trial_num: int


def _generate_wcst_trial(
    variant: str,
    rule: str,
    trial_num: int,
    randomize_rule: bool = False,
    bg_color: bool = False,
) -> WCSTTrial:
    """Generate a single WCST trial using existing generators."""
    if variant in ["card", "card-random", "card-image"]:
        # Use card generator - returns strings, not dicts
        bg_color_value = "white" if not bg_color else None  # None means random
        ambiguous = False
        given_str, options_str = wcst_generator(
            rule, randomize_rule, bg_color_value, ambiguous
        )

        # The generator returns string descriptions, and first option is always correct
        # given_str: e.g., "one red circle" or "two green triangle"
        # options_str: list of strings like ["one red circle", "two green triangle", ...]
        correct_idx = 0  # First option is always correct by design of wcst_generator

    elif variant == "string":
        # Use string generator
        max_length = 10
        given_str, options_str = string_generator(rule, max_length)

        # Find correct option
        if rule == "length":
            target = len(given_str)
            correct_idx = next(
                (i for i, opt in enumerate(options_str) if len(opt) == target), 0
            )
        elif rule == "vowels":
            target = sum(1 for c in given_str if c.lower() in "aeiou")
            correct_idx = next(
                (
                    i
                    for i, opt in enumerate(options_str)
                    if sum(1 for c in opt if c.lower() in "aeiou") == target
                ),
                0,
            )
        elif rule == "consonants":
            target = sum(
                1 for c in given_str if c.isalpha() and c.lower() not in "aeiou"
            )
            correct_idx = next(
                (
                    i
                    for i, opt in enumerate(options_str)
                    if sum(1 for c in opt if c.isalpha() and c.lower() not in "aeiou")
                    == target
                ),
                0,
            )
        else:
            correct_idx = 0

    else:
        # Empty variant: no cards/strings
        given_str = ""
        options_str = ["", "", "", ""]
        correct_idx = 0

    return WCSTTrial(
        given=given_str,
        options=options_str,
        rule=rule,
        correct=correct_idx + 1,  # 1-indexed
        trial_num=trial_num,
    )


def _generate_wcst_episode(
    variant: str,
    max_trials: int,
    bg_color: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a full WCST episode with rule cycles."""
    if seed is not None:
        random.seed(seed)

    # Determine rules based on variant
    if variant in ["card", "card-random", "card-image"]:
        rules = CARD_RULES.copy()
        if not bg_color:
            # Remove background from rules if not enabled
            rules = [r for r in rules if r != "background"]
    elif variant == "string":
        rules = STRING_RULES.copy()
    else:
        rules = ["rule"]  # Empty variant

    # Shuffle rules
    random.shuffle(rules)

    # Generate trials - convert to dicts for Arrow compatibility
    trials = []
    current_rule_idx = 0
    for trial_num in range(max_trials):
        rule = rules[current_rule_idx % len(rules)]
        randomize = variant == "card-random"
        trial = _generate_wcst_trial(variant, rule, trial_num, randomize, bg_color)
        # Convert dataclass to dict for serialization
        trials.append(
            {
                "given": trial.given,
                "options": trial.options,
                "rule": trial.rule,
                "correct": trial.correct,
                "trial_num": trial.trial_num,
            }
        )

    return {
        "variant": variant,
        "rules": rules,
        "trials": trials,
        "max_trials": max_trials,
    }


def _create_wcst_dataset(
    variant: str = "card",
    max_trials: int = 128,
    bg_color: bool = False,
    num_episodes: int = 10,
    seed: Optional[int] = None,
) -> Dataset:
    """Create dataset with procedurally generated WCST episodes."""
    rows: List[Dict[str, Any]] = []
    for i in range(num_episodes):
        episode_seed = (seed + i) if seed is not None else None
        episode_data = _generate_wcst_episode(
            variant, max_trials, bg_color, episode_seed
        )

        rows.append(
            {
                "example_id": f"wcst_{variant}_{i}",
                "prompt": [
                    {
                        "role": "user",
                        "content": "Complete the Wisconsin Card Sorting Test by matching cards/strings based on hidden rules.",
                    }
                ],
                "answer": "",  # No single answer for WCST
                "info": episode_data,
            }
        )
    return Dataset.from_list(rows)


def _parse_choice_answer(text: str) -> Tuple[Optional[int], str]:
    """Parse choice (1-4) from answer tags. Returns (choice, status)."""
    match = ANSWER_TAG_RE.search(text or "")
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


def load_environment(
    variant: str = "card",
    max_trials: int = 128,
    bg_color: bool = False,
    image_mode: bool = False,
    num_episodes: int = 10,
    seed: Optional[int] = None,
    **kwargs,
):
    """
    Load a verifiers MultiTurnEnv for WCST.

    Args:
        variant: "card", "card-random", "card-image", "string", or "empty"
        max_trials: Maximum trials per episode
        bg_color: Whether to include background color as a matching rule
        image_mode: Whether to use image-based cards (sets variant to "card-image")
        num_episodes: Number of episodes in dataset
        seed: Random seed for episode generation
        **kwargs: Additional args passed to MultiTurnEnv

    Returns:
        vf.Environment instance
    """
    if vf is None:
        raise ImportError("verifiers is not installed; use in a verifiers workspace")

    # Adjust variant if image_mode is requested
    if image_mode and variant == "card":
        variant = "card-image"

    dataset = _create_wcst_dataset(variant, max_trials, bg_color, num_episodes, seed)

    # Determine rules for system prompt
    if variant in ["card", "card-random", "card-image"]:
        if bg_color:
            rule_list = "number, color, shape, or background"
        else:
            rule_list = "number, color, or shape"
    elif variant == "string":
        rule_list = "length, number of vowels, or number of consonants"
    else:
        rule_list = "unknown rules"

    system_prompt = f"""You will be performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card/string and 4 options.
You must choose which option matches the given card/string based on a hidden rule.
The rule can be one of: {rule_list}.
After several correct responses, the rule will change without warning.
You must detect the rule change and adapt your strategy.

Your final answer should be 1, 2, 3, or 4 (the index of the matching option), wrapped with <answer> and </answer>"""

    parser = vf.Parser()

    # Rubric: compute reward based on correctness and errors
    async def compute_reward(parser, completion, answer, state, **_):
        """
        Compute reward matching CognitiveEval WCST design:
        - +1.0 correct
        - -1.0 invalid format/choice
        - -0.5 perseverative error
        - -0.5 failure to maintain set
        - -1.0 repeated choice in same trial
        """
        info = state.get("info", {})
        trials = info.get("trials", [])
        trial_idx = state.get("trial_idx", 0)

        if trial_idx >= len(trials):
            return 0.0

        trial = trials[trial_idx]

        # Parse choice
        last_completion = completion or state.get("completion") or ""
        choice, status = _parse_choice_answer(str(last_completion))

        if status != "valid" or choice is None:
            if status == "invalid_format":
                return REWARD_INVALID_FORMAT
            elif status == "invalid_choice":
                return REWARD_INVALID_CHOICE
            return REWARD_INVALID_FORMAT

        # Check if repeated
        prev_choices = state.get("prev_choices_in_trial", [])
        if choice in prev_choices:
            return REWARD_REPEATED

        # Check correctness
        is_correct = choice == trial.correct

        if is_correct:
            return REWARD_CORRECT
        else:
            # Check for perseverative error
            prev_rule = state.get("prev_rule", None)
            if prev_rule is not None and prev_rule != trial.rule:
                # Rule changed; check if still using old rule
                consecutive_correct = state.get("consecutive_correct_before_change", 0)
                if consecutive_correct >= RULE_CHANGE_THRESHOLD:
                    # Likely perseverative error
                    return REWARD_PERSEVERATIVE_ERROR

            # Check for failure to maintain set (error after 2+ correct)
            consecutive_correct = state.get("consecutive_correct", 0)
            if consecutive_correct >= 2:
                # Had streak but failed before completing category
                return REWARD_FAILURE_TO_MAINTAIN_SET

            # Regular incorrect response
            return REWARD_PERSEVERATIVE_ERROR  # Default penalty

    rubric = vf.Rubric(funcs=[compute_reward], parser=parser)

    class WCSTVerifiersEnv(vf.MultiTurnEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        async def setup_state(self, state: vf.State) -> vf.State:
            """Initialize episode state from info."""
            info = state.get("info", {})
            state["trial_idx"] = 0
            state["current_rule_idx"] = 0
            state["consecutive_correct"] = 0
            state["completed_categories"] = 0
            state["rule_cycle"] = 0
            state["perseverative_errors"] = 0
            state["failure_to_maintain_set"] = 0
            state["prev_rule"] = None
            state["consecutive_correct_before_change"] = 0
            state["prev_choices_in_trial"] = []
            state["trials"] = info.get("trials", [])
            state["rules"] = info.get("rules", [])
            state["max_trials"] = info.get("max_trials", 128)
            return state

        @vf.stop
        async def max_trials_reached(self, state: vf.State) -> bool:
            """Stop when max trials reached."""
            trial_idx = state.get("trial_idx", 0)
            max_trials = state.get("max_trials", 128)
            return trial_idx >= max_trials

        @vf.stop
        async def two_cycles_completed(self, state: vf.State) -> bool:
            """Stop when 2 full rule cycles completed."""
            rule_cycle = state.get("rule_cycle", 0)
            return rule_cycle >= 2

        async def env_response(
            self, messages: vf.Messages, state: vf.State, **kwargs
        ) -> vf.Messages:
            """Provide feedback after each choice."""
            info = state.get("info", {})
            trials = state.get("trials", [])
            trial_idx = state.get("trial_idx", 0)
            rules = state.get("rules", [])

            if trial_idx >= len(trials):
                return [{"role": "user", "content": "Episode complete!"}]

            trial = trials[trial_idx]

            last = state.get("completion") or ""
            choice, status = _parse_choice_answer(str(last))

            # Update prev_choices for this trial
            prev_choices = state.get("prev_choices_in_trial", [])
            if choice is not None:
                prev_choices.append(choice)
            state["prev_choices_in_trial"] = prev_choices

            if status != "valid" or choice is None:
                if status == "invalid_format":
                    feedback = "Please answer with 1, 2, 3, or 4 inside <answer> tags."
                else:
                    feedback = "Invalid choice. Please choose 1, 2, 3, or 4."
                return [{"role": "user", "content": feedback}]

            # Check if repeated
            if choice in prev_choices[:-1]:  # Exclude last (current) choice
                feedback = f"You already chose option {choice} in this trial. Please try a different option."
                return [{"role": "user", "content": feedback}]

            # Check correctness
            is_correct = choice == trial.correct

            if is_correct:
                state["consecutive_correct"] = state.get("consecutive_correct", 0) + 1

                # Check if completed category (threshold matches wcst_env)
                if state["consecutive_correct"] >= RULE_CHANGE_THRESHOLD:
                    state["completed_categories"] = (
                        state.get("completed_categories", 0) + 1
                    )
                    state["consecutive_correct"] = 0

                    # Advance rule
                    state["current_rule_idx"] = state.get("current_rule_idx", 0) + 1
                    if state["current_rule_idx"] >= len(rules):
                        state["rule_cycle"] = state.get("rule_cycle", 0) + 1
                        state["current_rule_idx"] = 0

                    state["consecutive_correct_before_change"] = RULE_CHANGE_THRESHOLD
                    state["prev_rule"] = trial.rule

                    feedback = "Correct! The rule has changed. Continue matching."
                else:
                    feedback = "Correct! Continue matching."

                # Clear prev_choices for next trial
                state["prev_choices_in_trial"] = []
                state["trial_idx"] = trial_idx + 1

            else:
                # Check for perseverative error
                consecutive_correct = state.get("consecutive_correct", 0)
                if consecutive_correct >= 2:
                    state["failure_to_maintain_set"] = (
                        state.get("failure_to_maintain_set", 0) + 1
                    )

                prev_rule = state.get("prev_rule", None)
                if prev_rule is not None and prev_rule != trial.rule:
                    consecutive_before_change = state.get(
                        "consecutive_correct_before_change", 0
                    )
                    if consecutive_before_change >= RULE_CHANGE_THRESHOLD:
                        state["perseverative_errors"] = (
                            state.get("perseverative_errors", 0) + 1
                        )

                state["consecutive_correct"] = 0
                feedback = "Incorrect. Please try again."

            # Add next trial if available
            next_trial_idx = state.get("trial_idx", 0)
            if next_trial_idx < len(trials):
                next_trial = trials[next_trial_idx]
                feedback += f"\n\nTrial {next_trial_idx + 1}:\n"
                feedback += f"Given: {next_trial.given}\n"
                feedback += f"Options:\n"
                for i, opt in enumerate(next_trial.options, 1):
                    feedback += f"  {i}. {opt}\n"
                feedback += f"\nWhich option matches the given?"

            return [{"role": "user", "content": feedback}]

        def get_rubric(self):
            """Return the rubric instance for this environment."""
            return self._wcst_rubric

        def compute_reward_with_state(
            self, completions: List[List[dict]], state: vf.State
        ) -> List[float]:
            """Compute turn + outcome rewards using rubric with state context."""
            rubric = self.get_rubric()

            info = state.get("info", {})
            trials = info.get("trials", [])
            trial_idx = state.get("trial_idx", 0)

            # Prepare per-example correct answers
            if 0 <= trial_idx < len(trials):
                correct = trials[trial_idx].get("correct") or trials[trial_idx].get(
                    "correct_idx", None
                )
            else:
                correct = None
            correct_answers = [correct] * len(completions)

            # State for repeat detection
            prev_choices = state.get("prev_choices_in_trial", [])
            seen_options = prev_choices  # reuse for repeat checks

            # Flags/metrics
            consecutive_correct = state.get("consecutive_correct", 0)
            prev_rule = state.get("prev_rule", None)
            consecutive_before_change = state.get(
                "consecutive_correct_before_change", 0
            )
            # Determine current rule
            current_rule = (
                trials[trial_idx]["rule"] if 0 <= trial_idx < len(trials) else None
            )

            # Perseverative flag mirrors verifiers compute_reward logic
            is_persev = False
            if (
                prev_rule is not None
                and current_rule is not None
                and prev_rule != current_rule
                and consecutive_before_change >= RULE_CHANGE_THRESHOLD
            ):
                is_persev = True

            # Failure-to-maintain flag: we align to env logic (>=2 before error)
            is_failure = consecutive_correct >= 2

            reward_kwargs = {
                "correct_answers": correct_answers,
                "prev_choices_in_trial": (
                    [prev_choices] * len(completions)
                    if isinstance(prev_choices, list)
                    and prev_choices
                    and not isinstance(prev_choices[0], list)
                    else prev_choices
                ),
                "seen_options": (
                    [seen_options] * len(completions)
                    if isinstance(seen_options, list)
                    and seen_options
                    and not isinstance(seen_options[0], list)
                    else seen_options
                ),
                "consecutive_correct": (
                    [consecutive_correct] * len(completions)
                    if not isinstance(consecutive_correct, list)
                    else consecutive_correct
                ),
                "is_perseverative_error": [is_persev] * len(completions),
                "is_failure_to_maintain": [is_failure] * len(completions),
            }

            # Turn-level rewards
            turn_rewards = []
            for func in rubric.turn_reward_funcs:
                turn_rewards.append(func(completions, correct_answers, **reward_kwargs))

            total_turn = [
                sum(r[i] for r in turn_rewards) for i in range(len(completions))
            ]

            # Outcome rewards (use final state counters)
            outcome_kwargs = {
                "completed_categories": state.get("completed_categories", 0),
                "rules": info.get("rules", []),
                "num_rules": len(info.get("rules", [])),
                "perseverative_errors": state.get("perseverative_errors", 0),
                "failure_to_maintain_set": state.get("failure_to_maintain_set", 0),
                "n_trials": state.get("trial_idx", 0),
            }

            outcome_rewards = [0.0] * len(completions)
            for func in rubric.outcome_reward_funcs:
                o = func(completions, correct_answers, **outcome_kwargs)
                for i in range(len(completions)):
                    outcome_rewards[i] += o[i]

            return [total_turn[i] + outcome_rewards[i] for i in range(len(completions))]

    # Create rubric instance
    wcst_rubric = WCSTRubric()

    env = WCSTVerifiersEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
    env._wcst_rubric = wcst_rubric
    return env
