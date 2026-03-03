"""
Multi-Turn Environment for WCST using Multi-Turn-RL-Agent's verifiers.
Simpler synchronous API compared to the async verifiers package.
"""

import os
import random
import re
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.envs import MultiTurnEnv
from WCST.utils import wcst_generator, string_generator, check_rule_ambiguity

# MODULE-LEVEL GLOBAL STATE STORAGE
# These persist across environment instance recreation during training
_GLOBAL_EPISODE_STATES: Dict[str, Dict[str, Any]] = {}
_GLOBAL_PROMPT_TO_INFO: Dict[str, Dict[str, Any]] = {}
_GLOBAL_STATE_LOCK = threading.Lock()  # Thread-safety for concurrent access

# Reward constants
REWARD_CORRECT = 1.0
REWARD_INVALID_FORMAT = -1.0
REWARD_INVALID_CHOICE = -1.0
REWARD_PERSEVERATIVE_ERROR = -0.5
REWARD_FAILURE_TO_MAINTAIN_SET = -0.5
REWARD_REPEATED = -1.0
RULE_CHANGE_THRESHOLD = 5

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_config_from_prompt(prompt_content: str) -> Dict[str, Any]:
    """Extract rules and other config from the initial prompt text.

    This allows state recovery when info is not passed to env_response.
    """
    config = {}

    # Extract rules from "The rule can be one of:" section
    rules_match = re.search(r"The rule can be one of: ([^.]+)", prompt_content)
    if rules_match:
        rules_str = rules_match.group(1).strip()
        # Parse rules like "number, color, shape, or background" or "number, color, or shape"
        rules_str = rules_str.replace(" or ", ", ")
        config["rules"] = [r.strip() for r in rules_str.split(",") if r.strip()]

    # Check for background color hint
    if "background" in prompt_content.lower():
        if "rules" not in config or "background" not in config["rules"]:
            config["rules"] = config.get("rules", ["number", "color", "shape"]) + [
                "background"
            ]

    return config


# Card WCST rules
CARD_RULES = ["number", "color", "shape", "background"]
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
    randomize: bool = False,
    bg_color: bool = False,
) -> WCSTTrial:
    """Generate a single WCST trial with shuffled options."""
    if variant in ["card", "card-random", "card-image"]:
        # Use wcst_generator for card-based WCST
        given_str, options_str = wcst_generator(
            rule=rule, randomize=randomize, bg_color=bg_color
        )
        # wcst_generator puts correct answer at index 0, we'll shuffle below
        correct_idx = 0
    elif variant == "string":
        # Use string_generator for string-based WCST
        given_str, options_str = string_generator()
        # Determine correct based on rule
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
        given_str = ""
        options_str = ["", "", "", ""]
        correct_idx = 0

    # Shuffle options and track the new position of the correct answer
    correct_option = options_str[correct_idx]
    indexed_options = list(enumerate(options_str))
    random.shuffle(indexed_options)
    shuffled_options = [opt for _, opt in indexed_options]
    # Find where the correct option ended up
    new_correct_idx = next(
        i for i, opt in enumerate(shuffled_options) if opt == correct_option
    )

    return WCSTTrial(
        given=given_str,
        options=shuffled_options,
        rule=rule,
        correct=new_correct_idx + 1,  # 1-indexed
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

    if variant in ["card", "card-random", "card-image"]:
        rules = CARD_RULES.copy()
        if not bg_color:
            rules = [r for r in rules if r != "background"]
    elif variant == "string":
        rules = STRING_RULES.copy()
    else:
        rules = ["rule"]

    random.shuffle(rules)

    trials = []
    for trial_num in range(max_trials):
        rule = rules[trial_num % len(rules)]
        randomize = variant == "card-random"
        trial = _generate_wcst_trial(variant, rule, trial_num, randomize, bg_color)
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


def _parse_choice_answer(text: str) -> Tuple[Optional[int], str]:
    """Parse choice (1-4) from answer tags."""
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


def create_wcst_dataset(
    variant: str = "card",
    max_trials: int = 64,
    bg_color: bool = False,
    num_episodes: int = 10,
    seed: Optional[int] = None,
    image_mode: bool = False,
    enable_thinking: bool = False,  # Whether to include thinking instructions
) -> Dataset:
    """Create dataset with procedurally generated WCST episodes.

    Args:
        variant: "card", "card-random", "card-image", or "string"
        max_trials: Maximum trials per episode
        bg_color: Whether to include background color attribute
        num_episodes: Number of episodes to generate
        seed: Random seed for reproducibility
        image_mode: Whether this is image modality (stored in info)
    """
    rows = []

    if variant in ["card", "card-random", "card-image"]:
        if bg_color:
            rule_list = "number, color, shape, or background"
        else:
            rule_list = "number, color, or shape"
    elif variant == "string":
        rule_list = "length, number of vowels, or number of consonants"
    else:
        rule_list = "unknown rules"

    # Determine modality for the example_id
    modality = "image" if image_mode or variant == "card-image" else "text"
    bg_str = "bg" if bg_color else "nobg"

    for i in range(num_episodes):
        episode_seed = (seed + i) if seed is not None else None
        episode_data = _generate_wcst_episode(
            variant, max_trials, bg_color, episode_seed
        )

        # Store modality info in episode_data
        episode_data["modality"] = modality
        episode_data["bg_color"] = bg_color

        # Get first trial for initial prompt
        first_trial = episode_data["trials"][0]

        initial_prompt = f"""You will be performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card/string and 4 options.
You must choose which option matches the given card/string based on a hidden rule.
The rule can be one of: {rule_list}.
After several correct responses, the rule will change without warning.
You must detect the rule change and adapt your strategy.

Your final answer should be 1, 2, 3, or 4 (the index of the matching option), wrapped with <answer> and </answer>

Here is your first trial:
Given: {first_trial['given']}
Options:
1. {first_trial['options'][0]}
2. {first_trial['options'][1]}
3. {first_trial['options'][2]}
4. {first_trial['options'][3]}

Respond with <answer>1</answer>, <answer>2</answer>, <answer>3</answer>, or <answer>4</answer>."""

        # Add thinking instruction or /no_think suffix
        if enable_thinking:
            initial_prompt += "\n\nThink step-by-step about the problem in maximum 1000 tokens, wrapped with <think> and </think>. Then provide your final answer."
        else:
            initial_prompt += " /no_think"

        example_id = f"wcst_{modality}_{bg_str}_{i}"
        episode_data["example_id"] = example_id  # Add to info for state tracking

        rows.append(
            {
                "example_id": example_id,
                "prompt": [{"role": "user", "content": initial_prompt}],
                "info": episode_data,
            }
        )

    return Dataset.from_list(rows)


class WCSTMultiTurnEnv(MultiTurnEnv):
    """WCST environment using Multi-Turn-RL-Agent's simpler synchronous API.

    Episode completion logic:
    - Easy: 3 rules (number, color, shape), each must be completed twice = 6 total
    - Hard: 4 rules (+background), each must be completed twice = 8 total
    - Rule is "completed" when N consecutive correct answers (N = RULE_CHANGE_THRESHOLD = 5)
    - Episode ends when all rules completed twice OR max_trials reached

    State is stored in MODULE-LEVEL dicts to persist across environment instances.
    Each parallel generation gets its own isolated state via trajectory-specific keys.
    """

    def __init__(self, dataset: Dataset, enable_thinking: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._dataset = dataset
        self._enable_thinking = enable_thinking

        # PRE-INDEX: Build a mapping from prompt hash to info (use global storage)
        # Use prompt-only key (shared across all generations for same prompt)
        for row in dataset:
            prompt = row.get("prompt", [])
            info = row.get("info", {})
            if prompt:
                key = self._get_prompt_key(prompt)
                _GLOBAL_PROMPT_TO_INFO[key] = info

    def get_dataset(self, **kwargs) -> Dataset:
        return self._dataset

    def get_rubric(self, **kwargs) -> List:
        """Return reward functions."""
        return [self.compute_reward]

    def _get_prompt_key(self, messages: List[Dict[str, str]]) -> str:
        """Get key based on prompt only (for info lookup)."""
        if messages:
            return f"wcst_prompt_{hash(messages[0].get('content', ''))}"
        return "wcst_prompt_default"

    def _get_state_key(
        self, messages: List[Dict[str, str]], info: Dict[str, Any] = None
    ) -> str:
        """Generate a unique key for this specific trajectory's state.

        Uses prompt hash + trajectory signature to ensure each parallel
        generation has its own isolated state.

        For first turn (1 assistant message, no user feedback), use id(messages).
        After that, use hash of all subsequent content.
        """
        if not messages:
            return "wcst_default"

        # Base key from prompt
        prompt_key = hash(messages[0].get("content", ""))

        # Get all messages after the initial prompt
        subsequent_msgs = messages[1:]

        # Count assistant and user messages
        assistant_count = sum(
            1 for m in subsequent_msgs if m.get("role") == "assistant"
        )
        user_feedback_count = sum(1 for m in subsequent_msgs if m.get("role") == "user")

        if assistant_count == 1 and user_feedback_count == 0:
            # First turn: use id(messages) for unique key per generation
            trajectory_sig = f"id_{id(messages)}"
        else:
            # After first feedback: use hash of ALL subsequent content
            all_contents = tuple(m.get("content", "") for m in subsequent_msgs)
            trajectory_sig = f"content_{hash(all_contents)}"

        return f"wcst_{prompt_key}_{trajectory_sig}"

    def _get_info_for_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Look up the original info for this episode from global pre-indexed data."""
        if not messages:
            return {}
        # Use prompt-only key for info lookup (shared across generations)
        key = self._get_prompt_key(messages)
        return _GLOBAL_PROMPT_TO_INFO.get(key, {})

    def _get_state_for_messages(
        self, messages: List[Dict[str, str]], info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get or initialize state for an episode based on messages.

        Each parallel generation gets its own isolated state via trajectory-specific keys.
        State migration handles the transition from id-based keys (turn 1) to content-based keys.
        """
        if not messages:
            return {}

        key = self._get_state_key(messages, info)

        # Check if we have state for this exact trajectory
        if key in _GLOBAL_EPISODE_STATES:
            return _GLOBAL_EPISODE_STATES[key]

        # No state found - need to create new one
        # First, get the info (shared across generations for same prompt)
        if not info:
            info = self._get_info_for_messages(messages)

        # Fall back to prompt extraction only if info still not found
        if not info:
            first_msg = messages[0].get("content", "")
            info = _extract_config_from_prompt(first_msg)

        # Check if there's a "parent" state from earlier in this trajectory
        subsequent_msgs = messages[1:]
        parent_state = None

        # Count current state
        assistant_count = sum(
            1 for m in subsequent_msgs if m.get("role") == "assistant"
        )
        user_count = sum(1 for m in subsequent_msgs if m.get("role") == "user")
        prompt_key = hash(messages[0].get("content", ""))

        if assistant_count >= 1 and user_count >= 1:
            # We're past turn 1, need to find parent state
            # The parent state is from before the last user+assistant exchange

            # Get the first assistant response to identify the trajectory
            first_assistant_content = None
            for m in messages:
                if m.get("role") == "assistant":
                    first_assistant_content = m.get("content", "")
                    break

            if first_assistant_content:
                # For turn 2 (assistant_count=2, user_count=1), parent was turn 1 (id-based)
                # For turn 3+, parent was turn 2+ (content-based)

                if assistant_count == 2 and user_count == 1:
                    # Parent was turn 1 - search by first_response match
                    # Use list() snapshot to avoid "dictionary changed size during iteration"
                    with _GLOBAL_STATE_LOCK:
                        state_snapshot = list(_GLOBAL_EPISODE_STATES.items())
                    for stored_key, stored_state in state_snapshot:
                        if stored_key.startswith(f"wcst_{prompt_key}_id_"):
                            if (
                                stored_state.get("_first_response")
                                == first_assistant_content
                            ):
                                parent_state = stored_state
                                break
                else:
                    # Parent was turn 2+ - reconstruct content-based key
                    # Parent key was based on messages up to and INCLUDING the second-to-last assistant
                    # (because parent stored state after processing that assistant's response)
                    assistant_indices = [
                        i
                        for i, m in enumerate(messages)
                        if m.get("role") == "assistant"
                    ]

                    if len(assistant_indices) >= 2:
                        # Get the second-to-last assistant index
                        second_to_last_idx = assistant_indices[-2]
                        # Parent's subsequent messages were [1:second_to_last_idx+1]
                        prev_subsequent = messages[1 : second_to_last_idx + 1]
                        all_contents = tuple(
                            m.get("content", "") for m in prev_subsequent
                        )
                        prev_key = f"wcst_{prompt_key}_content_{hash(all_contents)}"
                        parent_state = _GLOBAL_EPISODE_STATES.get(prev_key)

        if parent_state:
            # Copy parent state to new key
            import copy

            _GLOBAL_EPISODE_STATES[key] = copy.deepcopy(parent_state)
            return _GLOBAL_EPISODE_STATES[key]

        # No parent state - create fresh initial state
        rules = info.get("rules", ["number", "color", "shape"])
        trials = info.get("trials", [])
        max_trials = info.get("max_trials", len(trials) if trials else 64)
        variant = info.get("variant", "card")
        bg_color = info.get("bg_color", False)

        # Get first trial from pre-generated trials (matches initial prompt)
        first_trial = None
        if trials:
            first_trial = {
                "given": trials[0].get("given", ""),
                "options": trials[0].get("options", []),
                "rule": trials[0].get("rule", ""),
                "correct": trials[0].get("correct", 0),
            }

        _GLOBAL_EPISODE_STATES[key] = {
            "trial_idx": 0,
            "consecutive_correct": 0,
            "current_rule_idx": 0,
            "rule_completions": {
                r: 0 for r in rules
            },  # Each rule must be completed twice
            "prev_choices_in_trial": [],
            "total_trials": 0,
            # Store these for when info is not available
            "rules": rules,
            "max_trials": max_trials,
            "variant": variant,
            "bg_color": bg_color,
            # Current trial (matches the initial prompt)
            "current_trial": first_trial,
            # Store first response for parent lookup
            "_first_response": next(
                (
                    m.get("content", "")
                    for m in messages
                    if m.get("role") == "assistant"
                ),
                None,
            ),
        }
        return _GLOBAL_EPISODE_STATES[key]

    def is_completed(self, messages: List[Dict[str, str]], **kwargs) -> bool:
        """Check if episode is complete."""
        if not messages:
            return False

        last_msg = messages[-1].get("content", "")
        if "Episode complete" in last_msg or "WCST complete" in last_msg:
            return True
        if "Max trials reached" in last_msg:
            return True

        # Check step count
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        return len(assistant_msgs) >= self.max_steps

    def env_response(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        """Provide environment response after agent's choice.

        WCST logic (matching original wcst.py):
        - When INCORRECT: Repeat the SAME given/options (model must try again)
        - When CORRECT: Generate a NEW trial with the same rule
        - After N consecutive correct: Rule changes, generate new trial with new rule
        - Each rule must be completed twice (N consecutive correct twice each)
        """
        if not messages:
            return {"role": "user", "content": "Please make a choice."}

        # Get last assistant message
        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        if not last_assistant:
            return {
                "role": "user",
                "content": "Please make a choice (1, 2, 3, or 4) wrapped in <answer> tags.",
            }

        # Parse choice
        choice, status = _parse_choice_answer(last_assistant)

        no_think_suffix = "" if self._enable_thinking else " /no_think"

        if status != "valid" or choice is None:
            if status == "invalid_format":
                return {
                    "role": "user",
                    "content": f"Please answer with 1, 2, 3, or 4 inside <answer> tags.{no_think_suffix}",
                }
            else:
                return {
                    "role": "user",
                    "content": f"Invalid choice. Please choose 1, 2, 3, or 4.{no_think_suffix}",
                }

        # Get info from kwargs
        info = kwargs.get("info", {})

        # Get state first (pass info for initialization)
        state = self._get_state_for_messages(messages, info)

        # Use state values
        rules = state.get("rules") or info.get("rules", ["number", "color", "shape"])
        max_trials = state.get("max_trials") or info.get("max_trials", 64)
        variant = state.get("variant") or info.get("variant", "card")
        bg_color = state.get("bg_color", False) or info.get("bg_color", False)

        # Initialize current_rule_idx if needed
        if "current_rule_idx" not in state:
            state["current_rule_idx"] = 0

        # Initialize rule completions if not present
        if "rule_completions" not in state or not state["rule_completions"]:
            state["rule_completions"] = {r: 0 for r in rules}

        # Increment total trials
        state["total_trials"] = state.get("total_trials", 0) + 1

        # Check if we've exceeded max trials
        if state["total_trials"] > max_trials:
            return {
                "role": "user",
                "content": f"Episode complete! Max trials reached. WCST complete.{no_think_suffix}",
            }

        # Get current trial info (generated on first attempt or stored for retries)
        current_trial = state.get("current_trial")
        if current_trial is None:
            # This shouldn't happen in normal flow (trial set in initial prompt or after correct answer)
            # Generate one now as fallback
            current_rule = rules[state["current_rule_idx"] % len(rules)]
            randomize = variant == "card-random"
            trial = _generate_wcst_trial(variant, current_rule, 0, randomize, bg_color)
            state["current_trial"] = {
                "given": trial.given,
                "options": trial.options,
                "rule": trial.rule,
                "correct": trial.correct,
            }
            current_trial = state["current_trial"]

        correct_choice = current_trial.get("correct", 0)
        current_rule = current_trial.get("rule", "")
        is_correct = choice == correct_choice

        # Update state based on correctness
        if is_correct:
            state["consecutive_correct"] = state.get("consecutive_correct", 0) + 1
            feedback = "Correct!"

            # Check if rule is completed (N consecutive correct)
            if state["consecutive_correct"] >= RULE_CHANGE_THRESHOLD:
                state["rule_completions"][current_rule] = (
                    state["rule_completions"].get(current_rule, 0) + 1
                )
                state["consecutive_correct"] = 0

                # Check if all rules completed twice
                all_completed = all(
                    state["rule_completions"].get(r, 0) >= 2 for r in rules
                )
                if all_completed:
                    return {
                        "role": "user",
                        "content": f"{feedback} You've mastered all rules! Episode complete! WCST complete.{no_think_suffix}",
                    }

                # Move to next rule
                state["current_rule_idx"] = state.get("current_rule_idx", 0) + 1
                feedback = f"{feedback} Rule learned. The rule will change now."

            # Generate NEW trial (correct answer moves to new trial)
            new_rule = rules[state["current_rule_idx"] % len(rules)]
            randomize = variant == "card-random"
            new_trial = _generate_wcst_trial(
                variant, new_rule, state["total_trials"], randomize, bg_color
            )
            state["current_trial"] = {
                "given": new_trial.given,
                "options": new_trial.options,
                "rule": new_trial.rule,
                "correct": new_trial.correct,
            }

            response = f"""{feedback}

Next trial:
Given: {state["current_trial"]["given"]}
Options:
1. {state["current_trial"]["options"][0]}
2. {state["current_trial"]["options"][1]}
3. {state["current_trial"]["options"][2]}
4. {state["current_trial"]["options"][3]}

Which option matches the given item?{no_think_suffix}"""
        else:
            # INCORRECT: Reset consecutive correct, but keep SAME trial
            state["consecutive_correct"] = 0
            feedback = "Incorrect. Please try again."

            # Repeat the same trial
            response = f"""{feedback}

Given: {current_trial["given"]}
Options:
1. {current_trial["options"][0]}
2. {current_trial["options"][1]}
3. {current_trial["options"][2]}
4. {current_trial["options"][3]}

Which option matches the given item?{no_think_suffix}"""

        return {"role": "user", "content": response}

    def compute_reward(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        """Compute rewards for completions."""
        rewards = []

        for prompt, completion in zip(prompts, completions):
            total_reward = 0.0

            for msg in completion:
                if msg.get("role") != "assistant":
                    continue

                content = msg.get("content", "")
                choice, status = _parse_choice_answer(content)

                if status != "valid":
                    total_reward += REWARD_INVALID_FORMAT
                else:
                    # Without trial info, assume partial reward for valid format
                    total_reward += 0.1  # Small reward for valid response

            rewards.append(total_reward)

        return rewards


def load_environment(
    variant: str = "card",
    max_trials: int = 64,
    bg_color: bool = False,
    image_mode: bool = False,
    num_episodes: int = 10,
    seed: Optional[int] = None,
    max_steps: int = 100,  # Increased: allow up to 100 turns per episode
    max_episode_tokens: int = 32768,  # Total tokens for entire episode
    enable_thinking: bool = False,  # Whether to include thinking instructions
    **kwargs,
) -> WCSTMultiTurnEnv:
    """Load WCST MultiTurn environment.

    Args:
        variant: "card", "card-random", "card-image", or "string"
        max_trials: Maximum trials per episode
        bg_color: Whether to include background color attribute (4th rule for hard mode)
        image_mode: Whether to use image modality
        num_episodes: Number of episodes
        seed: Random seed
        max_steps: Maximum conversation turns (default 100)
        max_episode_tokens: Total token budget for entire episode (default 32768)
        enable_thinking: Whether to include thinking instructions in prompts

    Episode completion:
        - Easy (bg_color=False): 3 rules, each completed twice = 6 total completions needed
        - Hard (bg_color=True): 4 rules, each completed twice = 8 total completions needed
        - Rule complete = RULE_CHANGE_THRESHOLD (5) consecutive correct
    """
    if image_mode and variant == "card":
        variant = "card-image"

    dataset = create_wcst_dataset(
        variant=variant,
        max_trials=max_trials,
        bg_color=bg_color,
        num_episodes=num_episodes,
        seed=seed,
        image_mode=image_mode,
        enable_thinking=enable_thinking,
    )

    return WCSTMultiTurnEnv(
        dataset=dataset,
        enable_thinking=enable_thinking,
        max_steps=max_steps,
        max_episode_tokens=max_episode_tokens,
        **kwargs,
    )
