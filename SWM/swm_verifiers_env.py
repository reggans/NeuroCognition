"""
Verifiers environment for SWM (Spatial Working Memory).
Mirrors the RL reward design from swm_env.py without modifying it.

Supports:
- Text mode: box numbers 1 to n_boxes
- Image mode: coordinate-based box selection (placeholder)
- Rewards: +1.0 token found; 0.0 valid guess; -0.5 repeated/illegal; -1.0 invalid
"""

import os
import random
import re
import string
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import Dataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import verifiers as vf
except ImportError:
    vf = None  # type: ignore

from SWM.swm_rubric import SWMRubric

# Reward constants (matching swm_env.py)
REWARD_TOKEN_FOUND = 1.0
REWARD_VALID_GUESS = 0.0
REWARD_REPEATED_BOX = -0.5
REWARD_ILLEGAL_BOX = -0.5
REWARD_NO_BOX = -1.0
REWARD_INVALID_FORMAT = -1.0
REWARD_INVALID_ACTION = -1.0

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _generate_swm_example(
    n_boxes: int, n_tokens: int, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate a single SWM episode (procedural generation)."""
    if seed is not None:
        random.seed(seed)

    # Text mode: tokens are letters
    tokens = [string.ascii_uppercase[i] for i in range(n_tokens)]

    # Initialize legal boxes for each token
    legal_boxes = {token: list(range(1, n_boxes + 1)) for token in tokens}

    # Place tokens initially
    token_box: Dict[str, Optional[int]] = {}
    for token in tokens:
        token_box[token] = random.choice(legal_boxes[token])

    return {
        "tokens": tokens,
        "legal_boxes": legal_boxes,
        "token_box": token_box,
        "n_boxes": n_boxes,
        "n_tokens": n_tokens,
        "max_trials": n_boxes**2,  # Default max trials
    }


def _create_swm_dataset(
    n_boxes: int = 8,
    n_tokens: int = 1,
    max_trials: int = 64,
    num_episodes: int = 10,
    seed: Optional[int] = None,
) -> Dataset:
    """Create dataset with procedurally generated SWM episodes."""
    rows: List[Dict[str, Any]] = []
    for i in range(num_episodes):
        episode_seed = (seed + i) if seed is not None else None
        episode_data = _generate_swm_example(n_boxes, n_tokens, episode_seed)
        # Override max_trials if specified
        episode_data["max_trials"] = max_trials

        rows.append(
            {
                "example_id": f"swm_{i}",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Find {n_tokens} token(s) {n_boxes} times each by opening boxes.",
                    }
                ],
                "info": episode_data,
            }
        )
    return Dataset.from_list(rows)


def _parse_box_answer(
    text: str, n_boxes: int, mode: str = "text"
) -> Tuple[Optional[int], str]:
    """Parse box selection from answer tags. Returns (box_id, status)."""
    match = ANSWER_TAG_RE.search(text or "")
    if not match:
        return None, "invalid_format"

    answer_text = match.group(1).strip()

    if mode == "text":
        # Parse numeric box (1 to n_boxes)
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
    else:
        # Image mode: parse coordinates (x, y) - placeholder
        # For now, treat as text mode
        return _parse_box_answer(text, n_boxes, "text")


def load_environment(
    n_boxes: int = 8,
    n_tokens: int = 1,
    mode: str = "text",
    image_only: bool = False,
    max_trials: int = 64,
    num_episodes: int = 10,
    seed: Optional[int] = None,
    **kwargs,
):
    """
    Load a verifiers MultiTurnEnv for SWM.

    Args:
        n_boxes: Number of boxes
        n_tokens: Number of token types
        mode: "text" or "image"
        image_only: If True (image mode), only show image feedback without text
        max_trials: Maximum number of trials per episode (default: n_boxes^2)
        num_episodes: Number of episodes in dataset
        seed: Random seed for episode generation
        **kwargs: Additional args passed to MultiTurnEnv

    Returns:
        vf.Environment instance
    """
    if vf is None:
        raise ImportError("verifiers is not installed; use in a verifiers workspace")

    dataset = _create_swm_dataset(n_boxes, n_tokens, max_trials, num_episodes, seed)

    # Store mode info in dataset for rubric
    feedback_desc = "text feedback"
    if mode == "image":
        if image_only:
            feedback_desc = "image-only feedback (no text)"
        else:
            feedback_desc = "image + text feedback"

    system_prompt = f"""You will be performing a {'text' if mode == 'text' else 'image-based'} version of the Spatial Working Memory (SWM) test.
There are {n_tokens} types of tokens, hidden in any one of {n_boxes} boxes.
Your goal is to find the {n_tokens} types of tokens {n_boxes} times each, by repeatedly selecting a box to open.
{'You will receive ' + feedback_desc + ' after each guess.' if mode == 'image' else 'If the box contains a token, you will be informed which token type it is. If the box does not contain a token, you will be informed that it is empty.'}
Once the token is found, another token of the same type will be regenerated in another box.
The token will be generated in a box that has never contained a token of that type before in the trial.
The token may be generated in a box that has been opened and found empty before, as long as it never contained the token of that type previously.
Your final answer should be a number from 1-{n_boxes}, the index of the box you selected.

Which of the {n_boxes} boxes would you like to open?
Your final answer should be a box number, wrapped with <answer> and </answer>"""

    parser = vf.Parser()

    # Rubric: compute reward based on state updates
    async def compute_reward(parser, completion, answer, state, **_):
        """
        Compute reward matching CognitiveEval SWM design:
        - +1.0 token found
        - 0.0 valid guess (not repeated, legal box)
        - -0.5 repeated box in current search
        - -0.5 illegal box (already had all token types)
        - -1.0 invalid format/action
        """
        info = state.get("info", {})
        n_boxes = info.get("n_boxes", 8)

        # Get current episode state
        opened_boxes = state.get("opened_boxes", set())
        legal_boxes = state.get("legal_boxes", {})
        token_box = state.get("token_box", {})

        # Parse box selection
        last_completion = completion or state.get("completion") or ""
        box_id, status = _parse_box_answer(str(last_completion), n_boxes, "text")

        if status != "valid" or box_id is None:
            if status == "invalid_format":
                return REWARD_INVALID_FORMAT
            elif status == "invalid_action":
                return REWARD_INVALID_ACTION
            return REWARD_INVALID_FORMAT

        # Check if repeated
        if box_id in opened_boxes:
            return REWARD_REPEATED_BOX

        # Check if illegal (no more legal tokens can be in this box)
        is_illegal = True
        for token, legal in legal_boxes.items():
            if box_id in legal:
                is_illegal = False
                break

        if is_illegal:
            return REWARD_ILLEGAL_BOX

        # Check if token found
        found_token = False
        for token, box in token_box.items():
            if box == box_id:
                found_token = True
                break

        if found_token:
            return REWARD_TOKEN_FOUND
        else:
            return REWARD_VALID_GUESS

    rubric = vf.Rubric(funcs=[compute_reward], parser=parser)

    class SWMVerifiersEnv(vf.MultiTurnEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        async def setup_state(self, state: vf.State) -> vf.State:
            """Initialize episode state from info."""
            info = state.get("info", {})
            state["opened_boxes"] = set()
            state["found_tokens"] = []
            state["legal_boxes"] = {
                k: v.copy() for k, v in info.get("legal_boxes", {}).items()
            }
            state["token_box"] = info.get("token_box", {}).copy()
            state["tokens"] = info.get("tokens", [])
            state["n_boxes"] = info.get("n_boxes", 8)
            state["n_tokens"] = info.get("n_tokens", 1)
            state["n_guesses"] = 0
            state["error_count"] = 0  # Track errors for outcome reward
            state["max_guesses"] = info.get("max_trials", state["n_boxes"] ** 2)
            return state

        @vf.stop
        async def all_tokens_found(self, state: vf.State) -> bool:
            """Stop when all tokens found n_boxes times."""
            n_boxes = state.get("n_boxes", 8)
            legal_boxes = state.get("legal_boxes", {})
            # All tokens found when no legal boxes remain
            for token, legal in legal_boxes.items():
                if len(legal) > 0:
                    return False
            return True

        @vf.stop
        async def max_guesses_reached(self, state: vf.State) -> bool:
            """Stop when max guesses reached."""
            max_guesses = state.get("max_guesses", 64)
            return state.get("n_guesses", 0) >= max_guesses

        async def env_response(
            self, messages: vf.Messages, state: vf.State, **kwargs
        ) -> vf.Messages:
            """Provide feedback after each box selection."""
            info = state.get("info", {})
            n_boxes = state.get("n_boxes", 8)
            opened_boxes = state.get("opened_boxes", set())
            legal_boxes = state.get("legal_boxes", {})
            token_box = state.get("token_box", {})
            tokens = state.get("tokens", [])

            last = state.get("completion") or ""
            box_id, status = _parse_box_answer(str(last), n_boxes, "text")

            state["n_guesses"] = state.get("n_guesses", 0) + 1

            if status != "valid" or box_id is None:
                state["error_count"] = state.get("error_count", 0) + 1
                if status == "invalid_format":
                    feedback = f"Please answer with a box number between 1 and {n_boxes} inside <answer> tags."
                else:
                    feedback = f"Invalid box number. Please choose a box between 1 and {n_boxes}."
                return [{"role": "user", "content": feedback}]

            # Check if repeated
            if box_id in opened_boxes:
                state["error_count"] = state.get("error_count", 0) + 1
                feedback = f"You already opened box {box_id} in this search. Try a different box."
                return [{"role": "user", "content": feedback}]

            # Mark as opened
            opened_boxes.add(box_id)
            state["opened_boxes"] = opened_boxes

            # Check if illegal
            is_illegal = True
            for token, legal in legal_boxes.items():
                if box_id in legal:
                    is_illegal = False
                    break

            if is_illegal:
                state["error_count"] = state.get("error_count", 0) + 1
                feedback = (
                    f"Box {box_id} cannot contain any more tokens. Try a different box."
                )
                return [{"role": "user", "content": feedback}]

            # Check if token found
            found_token = None
            for token, box in token_box.items():
                if box == box_id:
                    found_token = token
                    break

            if found_token:
                # Update state: remove from legal boxes, regenerate token
                legal_boxes[found_token].remove(box_id)
                state["legal_boxes"] = legal_boxes
                state["found_tokens"] = state.get("found_tokens", []) + [found_token]

                # Regenerate token if legal boxes remain
                if len(legal_boxes[found_token]) > 0:
                    new_box = random.choice(legal_boxes[found_token])
                    token_box[found_token] = new_box
                    state["token_box"] = token_box
                    feedback = (
                        f"Found token {found_token}! It will regenerate in another box."
                    )
                else:
                    token_box[found_token] = None
                    state["token_box"] = token_box
                    feedback = f"Found token {found_token}! All {found_token} tokens have been found."

                # Reset opened boxes for new search
                state["opened_boxes"] = set()
            else:
                feedback = f"Box {box_id} is empty."

            # Add status update
            status_msg = "\n\n"
            for token in tokens:
                found_count = n_boxes - len(legal_boxes[token])
                status_msg += f"{token} tokens found: {found_count}/{n_boxes}\n"

            feedback += status_msg
            feedback += f"\nWhich of the {n_boxes} boxes would you like to open next?"

            return [{"role": "user", "content": feedback}]

        def get_rubric(self):
            """Return the rubric instance for this environment."""
            return self._swm_rubric

        def compute_reward_with_state(
            self, completions: List[List[dict]], state: vf.State
        ) -> List[float]:
            """
            Compute turn-level and outcome-level rewards using current state information.
            This method passes state to the rubric for accurate reward calculation.
            """
            rubric = self.get_rubric()

            # Prepare kwargs with state information (for both turn and outcome rewards)
            reward_kwargs = {
                "opened_boxes": state.get("opened_boxes", set()),
                "legal_boxes": state.get("legal_boxes", {}),
                "token_box": state.get("token_box", {}),
                "n_boxes": state.get("n_boxes", 8),
                "n_tokens": state.get("n_tokens", 1),
                "mode": "text",
                "box_coords": None,
                # Final state for outcome rewards
                "found_tokens": state.get("found_tokens", []),
                "n_guesses": state.get("n_guesses", 0),
                "error_count": state.get("error_count", 0),
            }

            # Compute turn-level rewards with state
            turn_rewards = []
            for func in rubric.turn_reward_funcs:
                rewards_list = func(completions, [""], **reward_kwargs)
                turn_rewards.append(rewards_list)

            # Sum turn-level rewards for each completion
            total_turn_rewards = [
                sum(r[i] for r in turn_rewards) for i in range(len(completions))
            ]

            # Compute outcome-level rewards with state (only at episode end)
            total_outcome_rewards = [0.0] * len(completions)
            for func in rubric.outcome_reward_funcs:
                outcome_rewards_list = func(completions, [""], **reward_kwargs)
                for i in range(len(completions)):
                    total_outcome_rewards[i] += outcome_rewards_list[i]

            # Combine turn and outcome rewards
            total_rewards = [
                total_turn_rewards[i] + total_outcome_rewards[i]
                for i in range(len(completions))
            ]
            return total_rewards

    # Create rubric instance
    swm_rubric = SWMRubric(
        n_boxes=kwargs.get("n_boxes", 8), n_tokens=kwargs.get("n_tokens", 1), mode=mode
    )

    env = SWMVerifiersEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
    env._swm_rubric = swm_rubric
    return env
