"""
Spatial Working Memory (SWM) Environment for RL training.
Supports both text and image-based versions.
"""

import random
import re
import string
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.base_env import CognitiveEnv, StepResult, ActionStatus

# Conditionally import SWMImage for image mode
try:
    from .image import SWMImage
except ImportError:
    SWMImage = None  # type: ignore


@dataclass
class SWMState:
    """Internal state for the SWM task."""
    tokens: List[str]
    legal_boxes: Dict[str, List[int]]  # Boxes that can still contain each token
    token_box: Dict[str, Optional[int]]  # Current location of each token (box ID)
    opened_boxes: Set[int]  # Boxes opened in current search
    found_tokens: List[str]  # Tokens found in current search
    searching: bool  # Whether we're in an active search
    # For image mode
    box_coords: Optional[List[Tuple[int, int]]] = None  # Grid coordinates of boxes


class SWMEnv(CognitiveEnv):
    """
    Spatial Working Memory environment for RL training.
    Supports both text-based and image-based versions.
    
    The model must find tokens hidden in boxes. Each token type must be found
    n_boxes times. Once found, the token regenerates in a box that has never
    contained that token type before.
    
    Modes:
    - text: Boxes identified by number (1 to n_boxes)
    - image: Boxes shown in a grid image, identified by coordinates (x, y)
    
    Reward structure:
    - +1.0 for finding a token
    - 0.0 for valid guess (not repeated, legal box)
    - -0.5 for repeated box in current search
    - -0.5 for illegal box (already had token of all types)
    - -1.0 for invalid coordinate (image mode)
    - -1.0 for invalid format
    - -1.0 for invalid action (text mode)
    
    Episode ends when:
    - All tokens found n_boxes times each
    - Maximum guesses reached (n_boxes^2)
    """
    
    def __init__(
        self,
        n_boxes: int = 8,
        n_tokens: int = 1,
        mode: str = "text",  # "text" or "image"
        cot: bool = False,
        think_budget: int = 64,
        note_assist: bool = False,
        image_path: Optional[str] = None,  # Directory to save images
        image_only: bool = False,  # Only include image in prompt (no text feedback)
        seed: Optional[int] = None,
        # Reward configuration
        reward_token_found: float = 1.0,
        reward_valid_guess: float = 0.0,
        reward_repeated_box: float = -0.5,
        reward_illegal_box: float = -0.5,
        reward_no_box: float = -1.0,
        reward_invalid_format: float = -1.0,
        reward_invalid_action: float = -1.0,
    ):
        """
        Initialize the SWM environment.
        
        Args:
            n_boxes: Number of boxes in the task
            n_tokens: Number of different token types
            mode: "text" for text-based, "image" for image-based
            cot: Whether to request chain-of-thought reasoning
            think_budget: Token budget for reasoning
            note_assist: Whether to provide note-taking assistance
            image_path: Directory to save generated images (required for image mode)
            image_only: If True, only return image path without text feedback
            seed: Random seed for reproducibility
            reward_token_found: Reward for finding a token (default: 1.0)
            reward_valid_guess: Reward for valid guess - not repeated, legal box (default: 0.0)
            reward_repeated_box: Penalty for opening same box again in current search (default: -0.5)
            reward_illegal_box: Penalty for box that can't contain any more tokens (default: -0.5)
            reward_no_box: Penalty for coordinate that doesn't match any box - image mode (default: -1.0)
            reward_invalid_format: Penalty for unparseable answer (default: -1.0)
            reward_invalid_action: Penalty for box number out of range (default: -1.0)
        """
        super().__init__(seed=seed)
        
        self.n_boxes = n_boxes
        self.n_tokens = n_tokens
        self.mode = mode
        self.cot = cot
        self.think_budget = think_budget
        self.note_assist = note_assist
        self.image_path = image_path
        self.image_only = image_only
        
        # Reward configuration
        self.reward_token_found = reward_token_found
        self.reward_valid_guess = reward_valid_guess
        self.reward_repeated_box = reward_repeated_box
        self.reward_illegal_box = reward_illegal_box
        self.reward_no_box = reward_no_box
        self.reward_invalid_format = reward_invalid_format
        self.reward_invalid_action = reward_invalid_action
        
        # Validate image mode requirements
        if mode == "image":
            if SWMImage is None:
                raise ImportError("PIL/Pillow is required for image mode. Install with: pip install Pillow")
            if self.image_path is None:
                self.image_path = os.path.join("SWM", "images")
            os.makedirs(self.image_path, exist_ok=True)
        
        # Image generator (initialized on reset for image mode)
        self._swm_image: Optional[Any] = None  # SWMImage instance
        
        # Episode state
        self._state: Optional[SWMState] = None
        self._n_guesses = 0
        self._max_guesses = n_boxes ** 2
        self._stats = {
            "illegal": 0,
            "valid": 0,
            "invalid": 0,
            "repeated": 0,
            "nobox": 0,  # For image mode: coordinate doesn't match a box
            "tokens_found": 0,
        }
        self._feedback = ""
        self._current_image_path: Optional[str] = None
        self._last_chosen_coord: Optional[Tuple[int, int]] = None  # For image mode feedback
        self._trial_idx = 0  # Counter for image filenames
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for SWM."""
        if self.mode == "image":
            prompt = f"""You will be performing the Spatial Working Memory task. 
You will be given an image containing {self.n_boxes} yellow boxes in a grid. 
There are {self.n_tokens} types of tokens, hidden in any one of {self.n_boxes} boxes.
Each token type is represented by a distinct color.
Your goal is to find the {self.n_tokens} types of tokens {self.n_boxes} times each, by repeatedly selecting a box to open.
A box can contain multiple types of tokens, but only one token of each type.
If the box contains multiple tokens, a token with mixed colors corresponding to the tokens will be shown.
Once the token is found, another will be generated in another box. 
The token will be generated in a box that has never contained a token of that type before in the trial. 
The token may be generated in a box that has been opened and found empty before, as long as it never contained that type of token previously. 
Your final answer should be a coordinate (x, y), the grid coordinate of the box you choose.
"""
        else:
            prompt = f"""You will be performing a text version of the Spatial Working Memory (SWM) test.
There are {self.n_tokens} types of tokens, hidden in any one of {self.n_boxes} boxes.
Your goal is to find the {self.n_tokens} types of tokens {self.n_boxes} times each, by repeatedly selecting a box to open.
If the box contains a token, you will be informed which token type it is.
If the box does not contain a token, you will be informed that it is empty.
Once the token is found, another token of the same type will be regenerated in another box.
The token will be generated in a box that has never contained a token of that type before in the trial.
The token may be generated in a box that has been opened and found empty before, as long as it never contained the token of that type previously.
Your final answer should be a number from 1-{self.n_boxes}, the index of the box you selected.
"""
        
        if self.cot:
            prompt += f"\nThink step-by-step, utilizing information from previous feedbacks, and state your reasoning in maximum {self.think_budget} tokens, wrapped with <think> and </think>. Then, provide a really short summary of your reasoning after the closing </think> tag.\n"
        else:
            prompt += "\nAnswer only with your final answer.\n"
        
        if self.mode == "image":
            prompt += f"""Which of the {self.n_boxes} boxes would you like to open?
Your final answer should be a grid coordinate (x, y), wrapped with <answer> and </answer>"""
        else:
            prompt += f"""Which of the {self.n_boxes} boxes would you like to open?
Your final answer should be a box number, wrapped with <answer> and </answer>"""
        
        return prompt
    
    def reset(self) -> str:
        """Reset the environment and return initial observation."""
        if self.seed is not None:
            random.seed(self.seed)
        
        # Reset counters
        self.step_count = 0
        self.history = []
        self._done = False
        self._n_guesses = 0
        self._stats = {
            "illegal": 0,
            "valid": 0,
            "invalid": 0,
            "repeated": 0,
            "nobox": 0,
            "tokens_found": 0,
        }
        self._feedback = ""
        
        # Initialize image generator for image mode
        box_coords: Optional[List[Tuple[int, int]]] = None
        if self.mode == "image":
            assert self.image_path is not None
            assert SWMImage is not None, "PIL/Pillow required for image mode"
            self._swm_image = SWMImage(self.image_path, self.n_boxes)
            box_coords = self._swm_image.box_coords.copy()
            self._current_image_path = os.path.join(self.image_path, "current.png")
            # For image mode, tokens are colors
            tokens = [self._swm_image.token_colors[i] for i in range(self.n_tokens)]
        else:
            # For text mode, tokens are letters
            tokens = [string.ascii_uppercase[x] for x in range(self.n_tokens)]
        
        # Initialize legal boxes for each token
        legal_boxes = {token: list(range(1, self.n_boxes + 1)) for token in tokens}
        
        # Place tokens initially
        token_box: Dict[str, Optional[int]] = {}
        for token in tokens:
            token_box[token] = random.choice(legal_boxes[token])
        
        self._state = SWMState(
            tokens=tokens,
            legal_boxes=legal_boxes,
            token_box=token_box,
            opened_boxes=set(),
            found_tokens=[],
            searching=True,
            box_coords=box_coords,
        )
        
        return self._format_observation()
    
    def _format_observation(self) -> str:
        """Format current state as observation."""
        state = self._state
        if state is None:
            return ""
        
        # For image_only mode, just return indication that image is available
        if self.mode == "image" and self.image_only:
            return f"[Image: {self._current_image_path}]"
        
        # Build status message
        msg = self._feedback
        
        for token in state.tokens:
            found_count = self.n_boxes - len(state.legal_boxes[token])
            msg += f"{token} tokens found: {found_count}\n"
        
        # Add notes if enabled
        if self.note_assist:
            for token, legal in state.legal_boxes.items():
                msg += f"Boxes that has contained token {token}: "
                for box in range(1, self.n_boxes + 1):
                    if box not in legal:
                        if self.mode == "image" and self._swm_image is not None:
                            box_coord = self._swm_image.get_box_coord(box)
                            msg += f"{box_coord}, "
                        else:
                            msg += f"{box}, "
                msg += "\n"
            
            msg += f"Opened boxes: "
            for box in state.opened_boxes:
                if self.mode == "image" and self._swm_image is not None:
                    box_coord = self._swm_image.get_box_coord(box)
                    msg += f"{box_coord}, "
                else:
                    msg += f"{box}, "
            msg += "\n"
        
        # Add question
        if self.mode == "image":
            msg += f"\nWhich of the {self.n_boxes} boxes would you like to open?\nYour final answer should be a grid coordinate (x, y), wrapped with <answer> and </answer>"
        else:
            msg += f"\nWhich of the {self.n_boxes} boxes would you like to open?\nYour final answer should be a box number, wrapped with <answer> and </answer>"
        
        return msg.strip()
    
    def get_current_image_path(self) -> Optional[str]:
        """Get the path to the current image (for image mode)."""
        return self._current_image_path
    
    def _regenerate_image(self) -> None:
        """
        Regenerate the SWM image after tokens are found and repositioned.
        
        Resets the image display to show all boxes in closed (base) state after
        token discovery, preparing for the next search phase with regenerated
        token positions.
        """
        if self._swm_image is None or self._state is None:
            return
        
        assert self.image_path is not None
        
        # Reset to base image (all boxes closed) by saving base_img as current
        self._swm_image.base_img.save(os.path.join(self.image_path, 'current.png'))
        self._current_image_path = os.path.join(self.image_path, "current.png")
    
    def parse_action(self, response: str) -> Tuple[Optional[int], ActionStatus]:
        """
        Parse the model's response to extract the chosen box ID.
        
        For text mode: expects a box number (1 to n_boxes)
        For image mode: expects coordinates (x, y) which are converted to box ID
        
        Special handling:
        - INVALID_FORMAT: Missing <answer> tags or unparseable content
        - INVALID_ACTION: Box number out of valid range (text mode)
        - NOBOX: Coordinate doesn't match any box in grid (image mode)
        
        Returns:
            Tuple of (box_id or None, ActionStatus)
            For image mode, also sets self._last_chosen_coord for feedback generation
        """
        match = re.search(r"<answer>(?s:.*?)</answer>", response)
        
        if match is None:
            return None, ActionStatus.INVALID_FORMAT
        
        answer_text = re.sub(r"<answer>|</answer>", "", match[0]).strip()
        
        if self.mode == "image":
            # Parse coordinate (x, y)
            try:
                coords = re.findall(r"[0-9]+", answer_text)
                if len(coords) < 2:
                    return None, ActionStatus.INVALID_FORMAT
                chosen_coord = (int(coords[0]), int(coords[1]))
                self._last_chosen_coord = chosen_coord  # Store for feedback
                
                # Convert coordinate to box ID
                if self._swm_image is None:
                    return None, ActionStatus.INVALID_ACTION
                try:
                    box_id = self._swm_image.get_box_id(chosen_coord)
                    return box_id, ActionStatus.VALID
                except ValueError:
                    # Coordinate doesn't match any box
                    return None, ActionStatus.NOBOX
            except (IndexError, ValueError):
                return None, ActionStatus.INVALID_FORMAT
        else:
            # Parse box number
            try:
                box = int(answer_text)
                if 1 <= box <= self.n_boxes:
                    return box, ActionStatus.VALID
                else:
                    return None, ActionStatus.INVALID_ACTION
            except ValueError:
                return None, ActionStatus.INVALID_FORMAT
    
    def step(self, action: str) -> StepResult:
        """
        Take a step in the environment by processing the model's box selection.
        
        Handles:
        - Parsing and validating the chosen box/coordinate
        - Detecting if a token was found in the selected box
        - Checking for legal/illegal and repeated box selections
        - Regenerating tokens in new locations after discovery
        - Updating visual feedback (image mode)
        
        Args:
            action: The model's response string containing <answer>choice</answer>
            
        Returns:
            StepResult with observation (formatted feedback or empty string), 
            accumulated reward, done flag, and detailed step info
        """
        if self._done:
            return StepResult(
                observation="",
                reward=0.0,
                done=True,
                info={"error": "Episode already finished"},
                truncated=False
            )
        
        if self._state is None:
            return StepResult(
                observation="",
                reward=0.0,
                done=True,
                info={"error": "No state available"},
                truncated=False
            )
        
        state = self._state  # Local reference for type checker
        
        self._n_guesses += 1
        self.step_count += 1
        
        # Parse action
        chosen_box, status = self.parse_action(action)
        
        # Build step info with coordinate for image mode
        step_info: Dict[str, Any] = {
            "guess_num": self._n_guesses,
            "token_boxes": {t: state.token_box[t] for t in state.tokens if state.token_box[t] is not None},
            "chosen_box": chosen_box,
            "raw_response": action,
            "status": status.value,
        }
        if self.mode == "image" and hasattr(self, '_last_chosen_coord'):
            step_info["chosen_coord"] = self._last_chosen_coord
        
        # Handle invalid format
        if status == ActionStatus.INVALID_FORMAT:
            if self.mode == "image":
                self._feedback = f"Please answer with a valid grid coordinate (x, y).\n"
            else:
                self._feedback = f"Please answer with a box number (1-{self.n_boxes}).\n"
            self._stats["invalid"] += 1
            
            step_info["found"] = False
            self.history.append(step_info)
            
            return StepResult(
                observation=self._format_observation(),
                reward=self.reward_invalid_format,
                done=False,
                info=step_info
            )
        
        # Handle invalid action (number out of range for text mode)
        if status == ActionStatus.INVALID_ACTION:
            self._feedback = f"Please answer with a valid box number (1-{self.n_boxes}).\n"
            self._stats["invalid"] += 1
            
            step_info["found"] = False
            self.history.append(step_info)
            
            return StepResult(
                observation=self._format_observation(),
                reward=self.reward_invalid_action,
                done=False,
                info=step_info
            )
        
        # Handle nobox (image mode: coordinate doesn't match any box)
        if status == ActionStatus.NOBOX:
            coord_str = str(self._last_chosen_coord) if self._last_chosen_coord else "given"
            self._feedback = f"No box in grid coordinate {coord_str}.\n"
            self._stats["nobox"] += 1
            
            step_info["found"] = False
            self.history.append(step_info)
            
            return StepResult(
                observation=self._format_observation(),
                reward=self.reward_no_box,
                done=False,
                info=step_info
            )
        
        # At this point chosen_box is valid (not None)
        assert chosen_box is not None
        
        # Check if box is legal (can still contain at least one token type)
        is_legal = any(chosen_box in legal for legal in state.legal_boxes.values())
        is_repeated = chosen_box in state.opened_boxes
        
        if not is_legal:
            self._stats["illegal"] += 1
            reward = self.reward_illegal_box
        elif is_repeated:
            self._stats["repeated"] += 1
            reward = self.reward_repeated_box
        else:
            self._stats["valid"] += 1
            reward = self.reward_valid_guess
        
        # Add box to opened set
        state.opened_boxes.add(chosen_box)
        
        # Check for token finds
        found_tokens = []
        for token in state.tokens:
            if state.token_box[token] is not None and chosen_box == state.token_box[token]:
                found_tokens.append(token)
                state.legal_boxes[token].remove(chosen_box)
                self._stats["tokens_found"] += 1
        
        # Format box description for feedback (coordinates for image mode, ID for text)
        if self.mode == "image" and self._last_chosen_coord is not None:
            box_desc = f"({self._last_chosen_coord[0]}, {self._last_chosen_coord[1]})"
        else:
            box_desc = str(chosen_box)
        
        # Build feedback
        if found_tokens:
            self._feedback = ""
            for token in found_tokens:
                self._feedback = f"Token {token} found in box {box_desc}.\n" + self._feedback
            reward += self.reward_token_found  # Bonus for finding token
            
            # Regenerate tokens in new locations
            for token in found_tokens:
                if len(state.legal_boxes[token]) > 0:
                    state.token_box[token] = random.choice(state.legal_boxes[token])
                else:
                    state.token_box[token] = None
            
            # Reset opened boxes for new search
            state.opened_boxes = set()
            
            # Update image for image mode (new token positions)
            if self.mode == "image" and self._swm_image is not None:
                self._regenerate_image()
        else:
            self._feedback = f"No tokens found in box {box_desc}.\n"
            
            # Update image for image mode (open the box)
            if self.mode == "image" and self._swm_image is not None and self._last_chosen_coord is not None:
                # Open the box in the image - pass coord tuple and token_box dict
                image = self._swm_image.open_box(self._last_chosen_coord, state.token_box)
                if self.image_path:
                    self._current_image_path = os.path.join(
                        self.image_path,
                        f"swm_trial_{self._trial_idx}_step_{self._n_guesses}.png"
                    )
                    image.save(self._current_image_path)
        
        step_info["found"] = len(found_tokens) > 0
        step_info["found_tokens"] = found_tokens
        self.history.append(step_info)
        
        # Check termination
        done = False
        truncated = False
        
        # All tokens found n_boxes times
        if all(len(legal) == 0 for legal in state.legal_boxes.values()):
            done = True
        
        # Max guesses reached
        if self._n_guesses >= self._max_guesses:
            done = True
            truncated = True
        
        self._done = done
        
        return StepResult(
            observation=self._format_observation() if not done else "",
            reward=reward,
            done=done,
            info=step_info,
            truncated=truncated
        )
    
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get internal state for serialization."""
        state = self._state
        return {
            "tokens": state.tokens if state else [],
            "legal_boxes": {k: list(v) for k, v in state.legal_boxes.items()} if state else {},
            "token_box": dict(state.token_box) if state else {},
            "opened_boxes": list(state.opened_boxes) if state else [],
            "found_tokens": list(state.found_tokens) if state else [],
            "searching": state.searching if state else False,
            "n_guesses": self._n_guesses,
            "stats": dict(self._stats),
            "feedback": self._feedback,
        }
    
    def _set_internal_state(self, state: Dict[str, Any]) -> None:
        """Restore internal state."""
        self._state = SWMState(
            tokens=state.get("tokens", []),
            legal_boxes={k: list(v) for k, v in state.get("legal_boxes", {}).items()},
            token_box=dict(state.get("token_box", {})),
            opened_boxes=set(state.get("opened_boxes", [])),
            found_tokens=list(state.get("found_tokens", [])),
            searching=state.get("searching", True),
        )
        self._n_guesses = state.get("n_guesses", 0)
        self._stats = dict(state.get("stats", {}))
        self._feedback = state.get("feedback", "")
    
    def compute_episode_reward(self) -> float:
        """Compute total episode reward by summing rewards from all steps.
        
        Accounts for:
        - Token discovery rewards (1.0 per token found)
        - Valid guess rewards (0.0 per valid attempt)
        - Penalties for invalid format, invalid action, and no-box errors
        
        Returns:
            Total accumulated reward for the episode.
        """
        total = 0.0
        for step in self.history:
            if step.get("status") == ActionStatus.INVALID_FORMAT.value:
                total += self.reward_invalid_format
            elif step.get("status") == ActionStatus.INVALID_ACTION.value:
                total += self.reward_invalid_action
            elif step.get("status") == ActionStatus.NOBOX.value:
                total += self.reward_no_box
            elif step.get("found"):
                total += self.reward_token_found + self.reward_valid_guess
            else:
                # Valid guess but no token found (could be legal, illegal, or repeated)
                total += self.reward_valid_guess
        return total
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics for the episode.
        
        Returns:
            Dict containing:
            - total_guesses: Total number of box selections made
            - tokens_found: Number of tokens discovered
            - tokens_to_find: Total tokens needed (n_boxes * n_tokens)
            - completion_rate: Tokens found / tokens to find
            - error_rate: (illegal + repeated) / valid guesses
            - illegal_count: Number of selections of boxes with all token types
            - repeated_count: Number of repeated box selections in current search
            - invalid_count: Number of invalid format/action responses
            - valid_count: Number of valid box selections
            - efficiency: Tokens found / total guesses
            - score: Traditional SWM score (1 - error_rate)
        """
        total_guesses = self._n_guesses
        tokens_to_find = self.n_boxes * self.n_tokens
        tokens_found = self._stats.get("tokens_found", 0)
        
        valid_guesses = total_guesses - self._stats.get("invalid", 0)
        error_rate = 0.0
        if valid_guesses > 0:
            error_rate = (self._stats.get("illegal", 0) + self._stats.get("repeated", 0)) / valid_guesses
        
        return {
            "total_guesses": total_guesses,
            "tokens_found": tokens_found,
            "tokens_to_find": tokens_to_find,
            "completion_rate": tokens_found / tokens_to_find if tokens_to_find > 0 else 0.0,
            "error_rate": error_rate,
            "illegal_count": self._stats.get("illegal", 0),
            "repeated_count": self._stats.get("repeated", 0),
            "invalid_count": self._stats.get("invalid", 0),
            "valid_count": self._stats.get("valid", 0),
            "efficiency": tokens_found / total_guesses if total_guesses > 0 else 0.0,
            "score": 1 - error_rate,  # Traditional SWM score
        }
