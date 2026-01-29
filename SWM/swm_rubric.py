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
    3. turn_box_legality_reward: Penalizes illegal box selections
       - -0.5 for repeated box (same box opened again in current search)
       - -0.5 for exhausted box (box has had ALL token types found in it)
    4. turn_token_found_reward: +1.0 per token found (detected from USER feedback)

    State reconstruction: Functions 3-4 parse USER feedback messages in the trajectory:
    - "Token X found in box Y!" → token found, start new search, update box_token_history
    - "No tokens found in box Y" → empty box, legal move
    - "already opened box" → repeated box in current search
    - Exhausted detection: box_token_history[box_id] contains all n_tokens token types

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
        Reward for box legality - summed across ALL turns.

        A box is ILLEGAL in two cases:
        1. Repeated: Box was already opened in the current search (within-search repeat)
        2. Exhausted: Box has already had ALL token types found in it across searches
           (i.e., it is IMPOSSIBLE for this box to contain any token)

        Per-turn rewards:
        - -0.5 for repeated box (detected from "already opened box" in USER feedback)
        - -0.5 for exhausted box (detected by tracing all "Token X found in box Y"
          messages and tracking which boxes have had all token types found)
        - 0.0 for legal moves

        State reconstruction:
        - Track `box_token_history[box_id] = set of tokens found there`
        - A box is exhausted when len(box_token_history[box_id]) == n_tokens
        - Reset `opened_boxes` set when any token is found (new search begins)

        Returns sum of per-turn rewards for each trajectory.
        """
        rewards = []
        n_boxes = kwargs.get("n_boxes", self.n_boxes)
        n_tokens = kwargs.get("n_tokens", self.n_tokens)
        mode = kwargs.get("mode", self.mode)
        box_coords = kwargs.get("box_coords", None)

        # Get token names (A, B, C, ... or color names for image mode)
        # Default to uppercase letters
        token_names = kwargs.get("tokens", None)
        if token_names is None:
            import string
            token_names = [string.ascii_uppercase[i] for i in range(n_tokens)]

        for trajectory in completions:
            total_reward = 0.0

            # Track which tokens have been found in each box across ALL searches
            # box_token_history[box_id] = set of token names found there
            box_token_history: dict = {}  # Dict[int, Set[str]]

            # Track boxes opened in the CURRENT search (resets when token found)
            opened_boxes_current_search: set = set()

            # Process trajectory turn by turn
            for i, msg in enumerate(trajectory):
                if msg["role"] != "assistant":
                    continue

                # Parse the box selection from this assistant message
                box_id, status = self._parse_box_answer(
                    msg["content"], n_boxes, mode, box_coords
                )

                if status != "valid" or box_id is None:
                    # Format/validity error handled by other rewards
                    continue

                # Look at the NEXT message (USER response) to determine outcome
                if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "user":
                    user_response = trajectory[i + 1]["content"]
                    user_response_lower = user_response.lower()

                    # Check for repeated box (within current search)
                    if "already opened box" in user_response_lower:
                        total_reward += REWARD_REPEATED_BOX  # -0.5
                        # Don't add to opened_boxes since it was already there
                        continue

                    # Check if this box is exhausted (all token types already found here)
                    if box_id in box_token_history and len(box_token_history[box_id]) >= n_tokens:
                        # Box is exhausted - cannot contain any more tokens
                        total_reward += REWARD_ILLEGAL_BOX  # -0.5
                        # Still add to opened_boxes for repeat tracking
                        opened_boxes_current_search.add(box_id)
                        continue

                    # Check if token was found
                    # Pattern: "Token X found in box Y!" or "Token X found in box (x, y)!"
                    token_found_match = re.search(
                        r"token\s+(\w+)\s+found", user_response_lower
                    )
                    if token_found_match:
                        token_name = token_found_match.group(1).upper()

                        # Record this token was found in this box
                        if box_id not in box_token_history:
                            box_token_history[box_id] = set()
                        box_token_history[box_id].add(token_name)

                        # New search begins - reset opened boxes
                        opened_boxes_current_search = set()
                    else:
                        # "No tokens found" or "Box is empty" - valid exploration
                        opened_boxes_current_search.add(box_id)

            rewards.append(total_reward)

        return rewards

    def turn_token_found_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Reward for finding tokens - summed across ALL turns.

        Per-turn rewards:
        - +1.0 for each turn where a token was found
        - 0.0 for turns where box was empty or repeated

        State is reconstructed by parsing USER feedback messages in the trajectory:
        - "Token X found in box Y!" indicates a token was found
        - "Box X is empty" or "already opened" indicates no token found

        Returns sum of per-turn rewards for each trajectory.
        """
        rewards = []

        for trajectory in completions:
            total_reward = 0.0

            # Process trajectory turn by turn
            # Each turn is: assistant message -> user response
            for i, msg in enumerate(trajectory):
                if msg["role"] != "assistant":
                    continue

                # Look at the NEXT message (USER response) to determine if token found
                if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "user":
                    user_response = trajectory[i + 1]["content"].lower()

                    # Check for token found pattern: "Token X found in box Y!"
                    if "token" in user_response and "found" in user_response:
                        total_reward += REWARD_TOKEN_FOUND  # +1.0
                    # else: no token found, 0.0

            rewards.append(total_reward)

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
