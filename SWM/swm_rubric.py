"""
SWM Rubric for multi-turn RL training.
Defines turn-level and outcome-level reward functions compatible with TRL.

Mirrors the reward design from swm_verifiers_env.py:
- 4 Turn-level rewards: format, box_validity, box_legality, token_found
- 2 Outcome rewards: total_tokens_found, error_ratio
"""

import os
import sys
import re
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from shared.rubrics import Rubric
    from shared.parsers import XMLParser
except ImportError:

    class Rubric:
        def __init__(self):
            self.turn_reward_funcs = []
            self.outcome_reward_funcs = []

    class XMLParser:
        def __init__(self, fields=None):
            self.fields = fields or []


# Reward constants (matching MT-GRPO paper reward design)
# Turn-level: +0.1 for correct format (encourage exploration with valid format)
REWARD_TOKEN_FOUND = 1.0
REWARD_VALID_FORMAT = 0.1  # Small positive for correct format (MT-GRPO Section 5.2)
REWARD_VALID_BOX = 0.1  # Small positive for valid box selection
REWARD_VALID_GUESS = 0.0
REWARD_REPEATED_BOX = -0.5
REWARD_ILLEGAL_BOX = -0.5
REWARD_NO_BOX = -1.0
REWARD_INVALID_FORMAT = -1.0
REWARD_INVALID_ACTION = -1.0

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class SWMRubric(Rubric):
    """
    Rubric for Spatial Working Memory (SWM).

    Supports both text and image modalities:
    - Text mode: box numbers (1 to n_boxes)
    - Image mode: coordinates (x, y) converted to box IDs

    Turn-level rewards (4 functions) - SUMMED across all turns:
    1. turn_format_reward: +0.1 per valid format, -1.0 per invalid
    2. turn_box_validity_reward: +0.1 per valid box, -1.0 per invalid
    3. turn_box_legality_reward: -0.5 per illegal/repeated, 0.0 per legal
    4. turn_token_found_reward: +1.0 per token found, 0.0 per empty

    Outcome-level rewards (2 functions):
    1. outcome_total_tokens_found: Tf / T (where T = n_boxes * n_tokens), [0.0-1.0]
    2. outcome_error_ratio: Nerr / Ntot (error turns / total turns), [0.0-1.0]
    """

    def __init__(self, n_boxes: int = 8, n_tokens: int = 1, mode: str = "text"):
        """
        Initialize with box/token counts and mode.

        Args:
            n_boxes: Number of boxes
            n_tokens: Number of token types
            mode: "text" (box numbers) or "image" (coordinates)
        """
        super().__init__()
        self.parser = XMLParser(fields=["thinking", "box"])
        self.n_boxes = n_boxes
        self.n_tokens = n_tokens
        self.mode = mode

    def _parse_text_answer(self, text: str, n_boxes: int) -> Tuple[Optional[int], str]:
        """
        Parse box number from <answer>X</answer> tags (text mode).

        Returns:
            (box_id, status) where status is one of:
            - "invalid_format": no <answer> tags or non-numeric
            - "invalid_action": numeric but out of range (1 to n_boxes)
            - "valid": valid box number
        """
        if not text:
            return None, "invalid_format"

        match = ANSWER_TAG_RE.search(text)
        if not match:
            return None, "invalid_format"

        answer_text = match.group(1).strip()
        nums = re.findall(r"\d+", answer_text)
        if not nums:
            return None, "invalid_format"

        try:
            box_num = int(nums[0])
            if 1 <= box_num <= n_boxes:
                return box_num, "valid"
            else:
                return None, "invalid_action"
        except ValueError:
            return None, "invalid_format"

    def _parse_image_answer(
        self, text: str, box_coords: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[Optional[int], str]:
        """
        Parse coordinates from <answer>X,Y</answer> tags (image mode).
        Converts coordinates to box ID by matching against box_coords.

        Returns:
            (box_id, status) where status is one of:
            - "invalid_format": no <answer> tags or malformed coordinates
            - "nobox": coordinates don't match any box
            - "valid": valid coordinates matching a box
        """
        if not text:
            return None, "invalid_format"

        match = ANSWER_TAG_RE.search(text)
        if not match:
            return None, "invalid_format"

        answer_text = match.group(1).strip()

        # Try to parse as "x,y" or "x y"
        nums = re.findall(r"\d+", answer_text)
        if len(nums) < 2:
            return None, "invalid_format"

        try:
            x = int(nums[0])
            y = int(nums[1])
            coord = (x, y)
        except ValueError:
            return None, "invalid_format"

        # Match coordinate to box ID
        if box_coords is None or len(box_coords) == 0:
            # No box coordinates available, can't match
            return None, "invalid_format"

        # Find which box this coordinate belongs to
        for box_id, box_coord in enumerate(box_coords, start=1):
            if box_coord == coord:
                return box_id, "valid"

        # Coordinate doesn't match any box
        return None, "nobox"

    def _parse_box_answer(
        self,
        text: str,
        n_boxes: int,
        mode: str = "text",
        box_coords: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[Optional[int], str]:
        """
        Parse box selection from <answer>...</answer> tags.
        Handles both text (box number) and image (coordinates) modes.

        Args:
            text: The answer text containing <answer>...</answer>
            n_boxes: Number of boxes (for text mode validation)
            mode: "text" or "image"
            box_coords: List of box coordinates (required for image mode)

        Returns:
            (box_id, status) where status depends on mode
        """
        if mode == "image":
            return self._parse_image_answer(text, box_coords)
        else:
            return self._parse_text_answer(text, n_boxes)

    # ========== TURN-LEVEL REWARDS (4 functions) ==========

    def turn_format_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Reward for answer format validity - summed across ALL turns.

        Per-turn rewards:
        - +0.1 for each turn with valid <answer> tags
        - -1.0 for each turn with invalid format

        Returns sum of per-turn rewards for each trajectory.
        """
        rewards = []
        n_boxes = kwargs.get("n_boxes", self.n_boxes)
        mode = kwargs.get("mode", self.mode)
        box_coords = kwargs.get("box_coords", None)

        for trajectory in completions:
            total_reward = 0.0
            has_assistant = False

            for msg in trajectory:
                if msg["role"] == "assistant":
                    has_assistant = True
                    _, status = self._parse_box_answer(
                        msg["content"], n_boxes, mode, box_coords
                    )
                    if status == "invalid_format":
                        total_reward += REWARD_INVALID_FORMAT  # -1.0
                    else:
                        total_reward += REWARD_VALID_FORMAT  # +0.1

            if not has_assistant:
                rewards.append(REWARD_INVALID_FORMAT)  # -1.0
            else:
                rewards.append(total_reward)

        return rewards

    def turn_box_validity_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Reward for box selection validity - summed across ALL turns.

        Per-turn rewards:
        - +0.1 for each turn with valid box selection
        - -1.0 for each turn with invalid box (out-of-range or nobox)
        - 0.0 for format errors (handled by turn_format_reward)

        Returns sum of per-turn rewards for each trajectory.
        """
        rewards = []
        n_boxes = kwargs.get("n_boxes", self.n_boxes)
        mode = kwargs.get("mode", self.mode)
        box_coords = kwargs.get("box_coords", None)

        for trajectory in completions:
            total_reward = 0.0

            for msg in trajectory:
                if msg["role"] == "assistant":
                    box_id, status = self._parse_box_answer(
                        msg["content"], n_boxes, mode, box_coords
                    )
                    if status == "valid":
                        total_reward += REWARD_VALID_BOX  # +0.1
                    elif status == "nobox" or status == "invalid_action":
                        total_reward += REWARD_INVALID_ACTION  # -1.0
                    # status == "invalid_format" -> 0.0 (handled by format reward)

            rewards.append(total_reward)

        return rewards

    def turn_box_legality_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Reward for box legality (not repeated, not already exhausted).

        Returns -0.5 if:
        - Box was already opened in current search (repeated)
        - Box is illegal (no token types can be in this box)
        Returns 0.0 if legal.

        Requires state information passed via kwargs.
        """
        rewards = []
        opened_boxes = kwargs.get("opened_boxes", set())
        legal_boxes = kwargs.get("legal_boxes", {})
        n_boxes = kwargs.get("n_boxes", self.n_boxes)
        mode = kwargs.get("mode", self.mode)
        box_coords = kwargs.get("box_coords", None)

        for trajectory in completions:
            # Get last assistant message
            last_assistant = None
            for msg in reversed(trajectory):
                if msg["role"] == "assistant":
                    last_assistant = msg["content"]
                    break

            if not last_assistant:
                rewards.append(REWARD_VALID_GUESS)  # 0.0
                continue

            # Parse box
            box_id, status = self._parse_box_answer(
                last_assistant, n_boxes, mode, box_coords
            )

            if status != "valid" or box_id is None:
                # Format or validity error handled by other rewards
                rewards.append(REWARD_VALID_GUESS)  # 0.0
                continue

            # Check if repeated
            if box_id in opened_boxes:
                rewards.append(REWARD_REPEATED_BOX)  # -0.5
                continue

            # Check if illegal (no more legal tokens can be in this box)
            is_illegal = True
            for token, legal in legal_boxes.items():
                if box_id in legal:
                    is_illegal = False
                    break

            if is_illegal:
                rewards.append(REWARD_ILLEGAL_BOX)  # -0.5
            else:
                rewards.append(REWARD_VALID_GUESS)  # 0.0

        return rewards

    def turn_token_found_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Reward for finding a token at the chosen box.

        Returns +1.0 if token was found.
        Returns 0.0 if box was empty.

        Requires state information passed via kwargs.
        """
        rewards = []
        token_box = kwargs.get("token_box", {})
        n_boxes = kwargs.get("n_boxes", self.n_boxes)
        mode = kwargs.get("mode", self.mode)
        box_coords = kwargs.get("box_coords", None)

        for trajectory in completions:
            # Get last assistant message
            last_assistant = None
            for msg in reversed(trajectory):
                if msg["role"] == "assistant":
                    last_assistant = msg["content"]
                    break

            if not last_assistant:
                rewards.append(REWARD_VALID_GUESS)  # 0.0
                continue

            # Parse box
            box_id, status = self._parse_box_answer(
                last_assistant, n_boxes, mode, box_coords
            )

            if status != "valid" or box_id is None:
                # Format or validity error handled by other rewards
                rewards.append(REWARD_VALID_GUESS)  # 0.0
                continue

            # Check if token found
            found_token = False
            for token, box in token_box.items():
                if box == box_id:
                    found_token = True
                    break

            if found_token:
                rewards.append(REWARD_TOKEN_FOUND)  # +1.0
            else:
                rewards.append(REWARD_VALID_GUESS)  # 0.0

        return rewards

    # ========== OUTCOME REWARDS (2 functions) ==========

    def outcome_total_tokens_found(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Outcome reward: total tokens found relative to required total.

        Formula: Tf / T
        where:
          Tf = number of tokens found (from final state)
          T = n_boxes * n_tokens (total required)

        Returns range [0.0, 1.0]

        Uses state-based computation via kwargs.
        """
        n_boxes = kwargs.get("n_boxes", self.n_boxes)
        n_tokens = kwargs.get("n_tokens", self.n_tokens)
        found_tokens = kwargs.get("found_tokens", [])  # From final state
        total_required = n_boxes * n_tokens

        # All completions share the same final state, so reward is same for all
        found_count = len(found_tokens)

        if total_required == 0:
            reward = 0.0
        else:
            reward = min(1.0, found_count / total_required)

        # Return same reward for each completion (batch size)
        return [reward] * len(completions)

    def outcome_error_ratio(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Outcome penalty: ratio of error turns to total turns.

        Formula: -Nerr/Ntot (as a penalty)
        where:
          Nerr = number of turns with errors (from final state)
          Ntot = total number of turns (from final state)

        Returns range [-1.0, 0.0]:
        - 0.0 if no errors (optimal)
        - -1.0 if all turns had errors (worst)

        Uses state-based computation via kwargs.
        """
        error_count = kwargs.get("error_count", 0)  # From final state
        n_guesses = kwargs.get("n_guesses", 0)  # From final state

        if n_guesses == 0:
            error_ratio = 0.0
        else:
            error_ratio = error_count / n_guesses

        # Return -error_ratio as a penalty (0.0 to -1.0)
        reward = -error_ratio

        # Return same reward for each completion (batch size)
        return [reward] * len(completions)

    @property
    def turn_reward_funcs(self) -> List:
        """
        Turn-level reward functions (called after each step).

        Order: format → validity → legality → token_found
        """
        return [
            self.turn_format_reward,
            self.turn_box_validity_reward,
            self.turn_box_legality_reward,
            self.turn_token_found_reward,
        ]

    @property
    def outcome_reward_funcs(self) -> List:
        """
        Outcome-level reward functions (called at episode end).

        Order: total_tokens_found → error_ratio
        """
        return [
            self.outcome_total_tokens_found,
            self.outcome_error_ratio,
        ]
