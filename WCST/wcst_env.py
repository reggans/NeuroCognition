"""
Wisconsin Card Sorting Test (WCST) Environment for RL training.
"""

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.base_env import CognitiveEnv, StepResult, ActionStatus
from WCST.utils import wcst_generator, string_generator, check_rule_ambiguity, count_vowels

# =============================================================================
# REWARD CONFIGURATION - Modify these values to tune reward structure
# =============================================================================
# Basic rewards
REWARD_CORRECT = 1.0           # Correct answer
REWARD_INCORRECT = 0.0         # Wrong answer (valid format)
REWARD_INVALID_FORMAT = -0.5   # Answer not parseable
REWARD_INVALID_ACTION = -0.5   # Answer out of range (not 1-4)

# WCST-specific error penalties (applied in addition to REWARD_INCORRECT)
REWARD_PERSEVERATIVE_ERROR = -0.2   # Repeating same wrong answer after negative feedback
REWARD_FAILURE_TO_MAINTAIN = -0.3   # Error after achieving conceptual response (3+ correct)

# Try to import image generation (optional, requires PIL)
try:
    from WCST.image import draw_five_cards
    HAS_PIL = True
except ImportError:
    draw_five_cards = None  # type: ignore
    HAS_PIL = False

# System prompts
WCST_PROMPTS = {
    "card": """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you have to figure out.
The cards will be described by the following attributes:
1. Number of symbols
2. Color of symbols
3. Shape of symbols
4. Background color of the card

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, you have to figure out the correct rule to match the cards.
If you are correct, you have to stick with the same attribute until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.

""",
    "card-random": """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you have to figure out.
The cards will be described by the following attributes in a random order:
1. Number of symbols
2. Color of symbols
3. Shape of symbols
4. Background color of the card

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, you have to figure out the correct rule to match the cards.
If you are correct, you have to stick with the same attribute until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.

""",
    "card-image": """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown an image containing five cards: four option cards labeled 1-4 and a "Given" card on the right.
Each card displays a certain number of colored shapes.

Your task is to match the Given card to one of the four option cards (1-4) based on an attribute you must figure out.
The possible matching attributes are:
1. Number of symbols on the card
2. Color of the symbols
3. Shape of the symbols
4. Background color of the card (if applicable)

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed.
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, figure out the new correct rule.
If you are correct, stick with the same attribute until you are incorrect.

Your final answer should be a number between 1-4 corresponding to the option card that matches the Given card.

""",
    "string": """You are performing a modified version of the Wisconsin Card Sorting Test (WCST).
You will be shown a given string, and you have to match it with one of four option strings according to a rule that you have to figure out.
The rule is one of the following:
1. Length of the string
2. The number of vowels in the string
3. The number of consonants in the string

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, you have to figure out the correct rule to match the strings.
If you are correct, you have to stick with the same rule until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the string you think is the correct match.

""",
    "empty": """You are performing a modified version of the Wisconsin Card Sorting Test (WCST).
One option among 1, 2, 3, and 4 is the correct answer.
You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the correct answer has changed, you have to figure out the correct answer.
If you are correct, you have to stick with the same answer until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the answer you think is correct.

"""
}


@dataclass
class WCSTTrial:
    """Represents a single trial in the WCST."""
    given: str
    options: List[str]
    correct_idx: int  # 1-indexed
    rule: str  # Can be string for card/string variants or stringified int for empty
    is_ambiguous: Optional[bool] = None
    # For image mode: store card attributes
    given_attrs: Optional[Dict[str, Any]] = None  # {"shape": ..., "color": ..., "count": ..., "background": ...}


class WCSTEnv(CognitiveEnv):
    """
    Wisconsin Card Sorting Test environment for RL training.
    
    The environment presents card matching trials where the model must
    discover the current sorting rule through trial and error.
    
    Reward structure:
    - +1.0 for correct answer
    - -0.1 for incorrect answer
    - -0.5 for invalid format
    
    Episode ends when:
    - max_trials is reached
    - All categories are completed (num_correct consecutive correct per rule)
    """
    
    def __init__(
        self,
        variant: str = "card",
        max_trials: int = 64,
        num_correct: int = 5,
        bg_color: bool = False,
        ambiguous_mode: str = "off",
        cot: bool = False,
        think_budget: int = 64,
        hint: bool = False,
        image_mode: bool = False,
        image_path: Optional[str] = None,
        image_only: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the WCST environment.
        
        Args:
            variant: Type of WCST ("card", "card-random", "string", "empty")
            max_trials: Maximum number of trials before episode ends
            num_correct: Consecutive correct answers needed per rule
            bg_color: Whether to include background color attribute
            ambiguous_mode: Ambiguity control ("off", "first", "rest")
            cot: Whether to request chain-of-thought reasoning
            think_budget: Token budget for reasoning
            hint: Whether to provide hints about the rule
            image_mode: Whether to generate card images (requires PIL)
            image_path: Directory to save generated images
            image_only: If True, observation only indicates image path (for multimodal models)
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        
        self.variant = variant
        self.max_trials = max_trials
        self.num_correct = num_correct
        self.bg_color = bg_color
        self.ambiguous_mode = ambiguous_mode
        self.cot = cot
        self.think_budget = think_budget
        self.hint = hint
        self.image_mode = image_mode
        self.image_path = image_path
        self.image_only = image_only
        
        # Validate image mode
        if image_mode:
            if not HAS_PIL:
                raise ImportError("PIL/Pillow is required for image mode. Install with: pip install Pillow")
            if variant not in ["card", "card-random"]:
                raise ValueError(f"Image mode only supports 'card' and 'card-random' variants, not '{variant}'")
            if image_path is None:
                self.image_path = os.path.join("WCST", "images")
            os.makedirs(self.image_path, exist_ok=True)  # type: ignore
        
        self._current_image_path: Optional[str] = None
        
        # Set up rules based on variant
        if variant in ["card", "card-random"]:
            self.rules = ["color", "shape", "number"]
            if bg_color:
                self.rules.append("background")
        elif variant == "string":
            self.rules = ["length", "vowels", "consonants"]
        elif variant == "empty":
            self.rules = [1, 2, 3, 4]
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Episode state
        self._current_trial: Optional[WCSTTrial] = None
        self._current_rule_idx = 0
        self._consecutive_correct = 0
        self._completed_categories = 0
        self._n_trials = 0
        self._total_correct = 0
        self._force_ambig = False
        self._rule_cycle = 0  # Track how many times we've cycled through rules
        self._feedback = ""
        
        # WCST-specific error tracking
        self._previous_rule_idx: Optional[int] = None  # For perseverative error detection
        self._last_response: Optional[int] = None      # Last response given by model
        self._perseverative_errors = 0                 # Count of perseverative errors
        self._failure_to_maintain_set = 0              # FMS count (error after 3+ correct)
        self._conceptual_responses = 0                 # 3+ consecutive correct in a row
        self._first_after_rule_change = False          # Skip perseverative penalty on first trial after rule change
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for WCST."""
        if self.image_mode:
            prompt = WCST_PROMPTS.get("card-image", WCST_PROMPTS["card"])
        else:
            prompt = WCST_PROMPTS.get(self.variant, WCST_PROMPTS["card"])
        
        if self.cot:
            prompt += f"Explain your thought process regarding the problem and the feedbacks you received in maximum {self.think_budget} tokens wrapped with <think> and </think>. Then, provide a really short summary of your reasoning after the closing </think> tag.\n"
        else:
            prompt += "Answer only with your final answer.\n"
        
        prompt += """State your final answer using the template: "<answer>your answer</answer>"\n"""
        
        return prompt
    
    def reset(self) -> str:
        """Reset the environment and return initial observation."""
        if self.seed is not None:
            random.seed(self.seed)
        
        # Reset state
        self.step_count = 0
        self.history = []
        self._done = False
        self._current_rule_idx = 0
        self._consecutive_correct = 0
        self._completed_categories = 0
        self._n_trials = 0
        self._total_correct = 0
        self._rule_cycle = 0
        self._feedback = ""
        self._force_ambig = self.ambiguous_mode == "first"
        
        # Reset WCST-specific error tracking
        self._previous_rule_idx = None
        self._last_response = None
        self._perseverative_errors = 0
        self._failure_to_maintain_set = 0
        self._conceptual_responses = 0
        self._first_after_rule_change = False
        
        # Generate first trial
        self._current_trial = self._generate_trial()
        
        return self._format_observation()
    
    def _generate_trial(self) -> WCSTTrial:
        """Generate a new trial based on current rule."""
        rule = self.rules[self._current_rule_idx]
        
        if self.variant == "empty":
            # Empty variant: no cards, just numbers
            return WCSTTrial(
                given="",
                options=["", "", "", ""],
                correct_idx=int(rule),
                rule=str(rule),
                is_ambiguous=False
            )
        
        # Generate cards/strings based on variant
        if self.variant in ["card", "card-random"]:
            randomize = self.variant == "card-random"
            if self.ambiguous_mode != "off":
                given, options = wcst_generator(
                    rule, randomize=randomize, bg_color=self.bg_color, ambiguous=self._force_ambig
                )
                # Update ambiguity for next trial
                if self.ambiguous_mode == "rest":
                    self._force_ambig = True
                else:
                    self._force_ambig = False
            else:
                given, options = wcst_generator(rule, randomize=randomize, bg_color=self.bg_color)
            
            # The correct answer is always the first in the original list
            correct_option = options[0]
            random.shuffle(options)
            correct_idx = options.index(correct_option) + 1
            
            # Check ambiguity
            is_ambiguous = None
            if self.ambiguous_mode != "off":
                try:
                    is_ambiguous = check_rule_ambiguity(given, correct_option, bg_color=self.bg_color)
                except:
                    pass
            
            # Parse card attributes for image mode
            given_attrs = None
            if self.image_mode:
                given_attrs = self._parse_card_description(given)
            
            return WCSTTrial(
                given=given,
                options=options,
                correct_idx=correct_idx,
                rule=str(rule),
                is_ambiguous=is_ambiguous,
                given_attrs=given_attrs
            )
        
        elif self.variant == "string":
            given, options = string_generator(rule)
            correct_option = options[0]
            random.shuffle(options)
            correct_idx = options.index(correct_option) + 1
            
            return WCSTTrial(
                given=given,
                options=options,
                correct_idx=correct_idx,
                rule=str(rule),
                is_ambiguous=False
            )
        
        raise ValueError(f"Unknown variant: {self.variant}")
    
    def _parse_card_description(self, description: str) -> Dict[str, Any]:
        """Parse a card description string into attributes for image generation."""
        parts = description.lower().split()
        
        # Number mapping
        number_map = {"one": 1, "two": 2, "three": 3, "four": 4}
        # Color options
        colors = ["red", "green", "blue", "yellow"]
        # Shape options
        shapes = ["circle", "triangle", "star", "square"]
        
        attrs: Dict[str, Any] = {"shape": "circle", "color": "red", "count": 1}
        
        for part in parts:
            if part in number_map:
                attrs["count"] = number_map[part]
            elif part in colors:
                # Could be shape color or background
                if "color" not in attrs or attrs.get("color") == "red":
                    attrs["color"] = part
                else:
                    attrs["background"] = part
            elif part in shapes:
                attrs["shape"] = part
        
        # If bg_color mode and we have a background in the description
        if self.bg_color and len(parts) > 3:
            # The last color mentioned is likely the background
            for part in reversed(parts):
                if part in colors and part != attrs.get("color"):
                    attrs["background"] = part
                    break
        
        return attrs
    
    def _generate_image(self) -> None:
        """Generate card image for the current trial (image mode only)."""
        if not self.image_mode or self._current_trial is None:
            return
        
        if draw_five_cards is None:
            return
        
        trial = self._current_trial
        if trial.given_attrs is None:
            return
        
        # Generate the image
        draw_five_cards(trial.given_attrs, bg_color=self.bg_color)
        
        # Update current image path
        assert self.image_path is not None
        self._current_image_path = os.path.join(self.image_path, "current.png")
    
    def get_current_image_path(self) -> Optional[str]:
        """Get the path to the current image (image mode only)."""
        return self._current_image_path
    
    def _parse_card_attrs(self, card_desc: str) -> Dict[str, str]:
        """
        Parse a card description string into attributes.
        
        Card format: "one red circle" or "two green triangle" etc.
        With bg_color: "one red circle green" (last word is background)
        """
        parts = card_desc.lower().split()
        
        numbers = {"one", "two", "three", "four"}
        colors = {"red", "green", "blue", "yellow"}
        shapes = {"circle", "triangle", "star", "square"}
        
        attrs = {}
        bg_color = None
        
        for part in parts:
            if part in numbers:
                attrs["number"] = part
            elif part in colors:
                # If we already have color, this must be background
                if "color" in attrs:
                    bg_color = part
                else:
                    attrs["color"] = part
            elif part in shapes:
                attrs["shape"] = part
        
        if bg_color:
            attrs["background"] = bg_color
        
        return attrs
    
    def _get_correct_for_rule(self, trial: WCSTTrial, rule_idx: int) -> Optional[int]:
        """
        Get the correct answer index for a specific rule on this trial.
        
        Used for perseverative error detection - determine what answer would
        be correct if the model was using a different rule.
        
        Returns None if cannot determine (e.g., empty variant or parse error).
        """
        if self.variant not in ["card", "card-random"]:
            # For string/empty variants, skip perseverative detection
            return None
        
        rule = self.rules[rule_idx]
        given_attrs = self._parse_card_attrs(trial.given)
        
        if rule not in given_attrs:
            return None
        
        target_value = given_attrs[rule]
        
        # Find which option matches the given card on this rule
        for i, option in enumerate(trial.options, 1):
            option_attrs = self._parse_card_attrs(option)
            if option_attrs.get(rule) == target_value:
                return i
        
        return None
    
    def _format_observation(self) -> str:
        """Format the current trial as an observation string."""
        trial = self._current_trial
        if trial is None:
            return ""
        
        # Generate image for image mode
        if self.image_mode:
            self._generate_image()
            
            if self.image_only:
                # Just return feedback and image indication
                return f"{self._feedback}[Image: {self._current_image_path}]\nSelect option 1-4.".strip()
            else:
                # Return feedback with indication that image is available
                obs = f"{self._feedback}[See image for cards]\nSelect option 1-4 to match the Given card."
                if self.hint:
                    obs += f"\nRule: {trial.rule}"
                return obs.strip()
        
        if self.variant == "empty":
            obs = f"{self._feedback}Options:\n1.\n2.\n3.\n4."
        else:
            obs = f"{self._feedback}Given: {trial.given}\nOptions:\n"
            for i, opt in enumerate(trial.options, 1):
                obs += f"{i}. {opt}\n"
        
        if self.hint:
            obs += f"\nRule: {trial.rule}"
        
        return obs.strip()
    
    def parse_action(self, response: str) -> Tuple[Optional[int], ActionStatus]:
        """Parse the model's response to extract the chosen option."""
        # Look for answer in <answer> tags
        match = re.search(r"<answer>(?s:.*?)</answer>", response)
        
        if match is None:
            return None, ActionStatus.INVALID_FORMAT
        
        answer_text = re.sub(r"<answer>|</answer>", "", match[0]).strip()
        
        try:
            answer = int(answer_text)
            if 1 <= answer <= 4:
                return answer, ActionStatus.VALID
            else:
                return None, ActionStatus.INVALID_ACTION
        except ValueError:
            return None, ActionStatus.INVALID_FORMAT
    
    def step(self, action: str) -> StepResult:
        """
        Take a step in the environment.
        
        Args:
            action: The model's response string
            
        Returns:
            StepResult with observation, reward, done, and info
        """
        if self._done:
            return StepResult(
                observation="",
                reward=0.0,
                done=True,
                info={"error": "Episode already finished"},
                truncated=False
            )
        
        if self._current_trial is None:
            return StepResult(
                observation="",
                reward=0.0,
                done=True,
                info={"error": "No trial available"},
                truncated=False
            )
        
        trial = self._current_trial  # Local reference for type checker
        
        self._n_trials += 1
        self.step_count += 1
        
        # Parse the action
        parsed_action, status = self.parse_action(action)
        
        # Prepare step info
        step_info = {
            "trial_num": self._n_trials,
            "rule": trial.rule,
            "correct_answer": trial.correct_idx,
            "model_answer": parsed_action,
            "raw_response": action,
            "status": status.value,
        }
        
        if trial.is_ambiguous is not None:
            step_info["is_ambiguous"] = trial.is_ambiguous
        
        # Handle invalid format
        if status == ActionStatus.INVALID_FORMAT:
            self._feedback = 'Answer not found. Please state your final answer using the template: "<answer>your answer</answer>"\n'
            self._consecutive_correct = 0
            
            step_info["correct"] = False
            self.history.append(step_info)
            
            return StepResult(
                observation=self._format_observation(),
                reward=REWARD_INVALID_FORMAT,
                done=False,
                info=step_info
            )
        
        # Handle invalid action (number out of range)
        if status == ActionStatus.INVALID_ACTION:
            self._feedback = "Please answer with a number between 1 and 4.\n"
            self._consecutive_correct = 0
            
            step_info["correct"] = False
            self.history.append(step_info)
            
            return StepResult(
                observation=self._format_observation(),
                reward=REWARD_INVALID_ACTION,
                done=False,
                info=step_info
            )
        
        # Check if answer is correct
        correct = parsed_action == trial.correct_idx
        step_info["correct"] = correct
        
        # WCST-specific error detection
        is_perseverative = False
        is_failure_to_maintain = False
        
        # Capture and clear first-after-rule-change flag BEFORE processing
        is_first_after_rule_change = self._first_after_rule_change
        self._first_after_rule_change = False
        
        if correct:
            self._feedback = "Correct!\n"
            self._consecutive_correct += 1
            self._total_correct += 1
            reward = REWARD_CORRECT
            
            # Track conceptual response (3+ consecutive correct)
            if self._consecutive_correct >= 3:
                self._conceptual_responses += 1
            
            # Check if rule is mastered
            if self._consecutive_correct >= self.num_correct:
                self._completed_categories += 1
                
                # Store current rule as previous before changing
                self._previous_rule_idx = self._current_rule_idx
                
                self._consecutive_correct = 0
                self._current_rule_idx += 1
                self._first_after_rule_change = True  # Mark next trial as first after change
                
                # Check if we've completed a full cycle
                if self._current_rule_idx >= len(self.rules):
                    self._current_rule_idx = 0
                    self._rule_cycle += 1
        else:
            # Error - check for WCST-specific error types
            reward = REWARD_INCORRECT
            
            # Failure to Maintain Set: error after achieving conceptual level (3+ correct)
            if self._consecutive_correct >= 3:
                is_failure_to_maintain = True
                self._failure_to_maintain_set += 1
                reward += REWARD_FAILURE_TO_MAINTAIN
            
            # Perseverative Error: response matches previous rule after rule change
            # Only check if there was a previous rule and NOT the first trial after rule change
            if (self._previous_rule_idx is not None and 
                self._previous_rule_idx != self._current_rule_idx and
                parsed_action is not None and
                not is_first_after_rule_change):  # Skip penalty on first trial after change
                # Check if response would be correct under previous rule
                prev_correct = self._get_correct_for_rule(trial, self._previous_rule_idx)
                if parsed_action == prev_correct:
                    is_perseverative = True
                    self._perseverative_errors += 1
                    reward += REWARD_PERSEVERATIVE_ERROR
            
            self._feedback = "Incorrect. Please try again.\n"
            self._consecutive_correct = 0
        
        # Store last response for potential future use
        self._last_response = parsed_action
        
        # Add error type info to step_info
        step_info["is_perseverative_error"] = is_perseverative
        step_info["is_failure_to_maintain"] = is_failure_to_maintain
        
        self.history.append(step_info)
        
        # Check termination conditions
        done = False
        truncated = False
        
        if self._n_trials >= self.max_trials:
            done = True
            truncated = True
        elif self._rule_cycle >= 2:  # Completed 2 full cycles
            done = True
        
        self._done = done
        
        # Generate next trial if not done
        if not done:
            self._current_trial = self._generate_trial()
        
        return StepResult(
            observation=self._format_observation() if not done else "",
            reward=reward,
            done=done,
            info=step_info,
            truncated=truncated
        )
    
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get internal state for serialization."""
        return {
            "current_trial": {
                "given": self._current_trial.given if self._current_trial else None,
                "options": self._current_trial.options if self._current_trial else None,
                "correct_idx": self._current_trial.correct_idx if self._current_trial else None,
                "rule": self._current_trial.rule if self._current_trial else None,
                "is_ambiguous": self._current_trial.is_ambiguous if self._current_trial else None,
                "given_attrs": self._current_trial.given_attrs if self._current_trial else None,
            } if self._current_trial else None,
            "current_rule_idx": self._current_rule_idx,
            "consecutive_correct": self._consecutive_correct,
            "completed_categories": self._completed_categories,
            "n_trials": self._n_trials,
            "total_correct": self._total_correct,
            "force_ambig": self._force_ambig,
            "rule_cycle": self._rule_cycle,
            "feedback": self._feedback,
            "current_image_path": self._current_image_path,
            # WCST-specific error tracking
            "previous_rule_idx": self._previous_rule_idx,
            "last_response": self._last_response,
            "perseverative_errors": self._perseverative_errors,
            "failure_to_maintain_set": self._failure_to_maintain_set,
            "conceptual_responses": self._conceptual_responses,
            "first_after_rule_change": self._first_after_rule_change,
        }
    
    def _set_internal_state(self, state: Dict[str, Any]) -> None:
        """Restore internal state."""
        trial_data = state.get("current_trial")
        if trial_data:
            self._current_trial = WCSTTrial(
                given=trial_data["given"],
                options=trial_data["options"],
                correct_idx=trial_data["correct_idx"],
                rule=trial_data["rule"],
                is_ambiguous=trial_data.get("is_ambiguous"),
                given_attrs=trial_data.get("given_attrs"),
            )
        else:
            self._current_trial = None
        
        self._current_rule_idx = state.get("current_rule_idx", 0)
        self._consecutive_correct = state.get("consecutive_correct", 0)
        self._completed_categories = state.get("completed_categories", 0)
        self._n_trials = state.get("n_trials", 0)
        self._total_correct = state.get("total_correct", 0)
        self._force_ambig = state.get("force_ambig", False)
        self._rule_cycle = state.get("rule_cycle", 0)
        self._feedback = state.get("feedback", "")
        self._current_image_path = state.get("current_image_path")
        # WCST-specific error tracking
        self._previous_rule_idx = state.get("previous_rule_idx")
        self._last_response = state.get("last_response")
        self._perseverative_errors = state.get("perseverative_errors", 0)
        self._failure_to_maintain_set = state.get("failure_to_maintain_set", 0)
        self._conceptual_responses = state.get("conceptual_responses", 0)
        self._first_after_rule_change = state.get("first_after_rule_change", False)
    
    def compute_episode_reward(self) -> float:
        """Compute total episode reward."""
        total = 0.0
        for step in self.history:
            if step.get("status") == ActionStatus.INVALID_FORMAT.value:
                total += REWARD_INVALID_FORMAT
            elif step.get("status") == ActionStatus.INVALID_ACTION.value:
                total += REWARD_INVALID_ACTION
            elif step.get("correct"):
                total += REWARD_CORRECT
            else:
                # Base incorrect penalty
                total += REWARD_INCORRECT
                # Additional penalties for WCST-specific errors
                if step.get("is_perseverative_error"):
                    total += REWARD_PERSEVERATIVE_ERROR
                if step.get("is_failure_to_maintain"):
                    total += REWARD_FAILURE_TO_MAINTAIN
        return total
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics."""
        total_trials = len(self.history)
        valid_trials = sum(1 for s in self.history if s.get("status") == ActionStatus.VALID.value)
        correct_trials = sum(1 for s in self.history if s.get("correct"))
        total_errors = valid_trials - correct_trials
        
        # Calculate WCST-specific metrics from history
        perseverative_errors = sum(1 for s in self.history if s.get("is_perseverative_error"))
        failure_to_maintain_set = sum(1 for s in self.history if s.get("is_failure_to_maintain"))
        
        return {
            "total_trials": total_trials,
            "valid_trials": valid_trials,
            "correct_trials": correct_trials,
            "accuracy": correct_trials / valid_trials if valid_trials > 0 else 0.0,
            "completed_categories": self._completed_categories,
            "rule_cycles_completed": self._rule_cycle,
            # WCST-specific error metrics
            "total_errors": total_errors,
            "perseverative_errors": perseverative_errors,
            "failure_to_maintain_set": failure_to_maintain_set,
            "perseverative_error_rate": perseverative_errors / total_errors if total_errors > 0 else 0.0,
            "conceptual_responses": self._conceptual_responses,
            "episode_reward": self.compute_episode_reward(),
        }
