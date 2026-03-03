"""
Multi-Turn Environment for SWM using Multi-Turn-RL-Agent's verifiers.

Easy setup: 8 boxes, 1 token
Hard setup: 12 boxes, 2 tokens

Game logic (from original swm.py):
- There are n_tokens types of tokens, each must be found n_boxes times
- legal_boxes[token] starts as ALL boxes [1..n_boxes] for each token
- When token is found in a box, that box is REMOVED from legal_boxes[token]
  (token can never appear in that box again)
- opened_boxes tracks boxes opened in CURRENT search, resets when ANY token found
- A box can contain MULTIPLE tokens (one of each type)
- Model searches for ALL tokens at once, not one at a time
"""

import os
import random
import re
import string
import sys
import threading
from typing import Any, Dict, List, Optional

from datasets import Dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.envs import MultiTurnEnv

# Reward constants
REWARD_TOKEN_FOUND = 1.0
REWARD_VALID_GUESS = 0.0
REWARD_REPEATED_BOX = -0.5
REWARD_ILLEGAL_BOX = -0.5
REWARD_INVALID_FORMAT = -1.0

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# MODULE-LEVEL STATE STORAGE
# These persist across environment instances (important for multi-worker training)
_GLOBAL_EPISODE_STATES: Dict[str, Dict[str, Any]] = {}
_GLOBAL_PROMPT_TO_INFO: Dict[str, Dict[str, Any]] = {}
_GLOBAL_STATE_LOCK = threading.Lock()  # Thread-safety for concurrent access


def _generate_swm_example(
    n_boxes: int, n_tokens: int, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate a single SWM episode.

    Args:
        n_boxes: Number of boxes (8 for easy, 12 for hard)
        n_tokens: Number of tokens to find (1 for easy, 2 for hard)
        seed: Random seed
    """
    if seed is not None:
        random.seed(seed)

    tokens = [string.ascii_uppercase[i] for i in range(n_tokens)]
    # legal_boxes tracks which boxes can still contain each token
    # Starts with ALL boxes - shrinks as tokens are found
    legal_boxes = {token: list(range(1, n_boxes + 1)) for token in tokens}

    # Place each token in a random legal box
    token_box = {}
    for token in tokens:
        token_box[token] = random.choice(legal_boxes[token])

    return {
        "tokens": tokens,
        "legal_boxes": legal_boxes,
        "token_box": token_box,
        "n_boxes": n_boxes,
        "n_tokens": n_tokens,
        "max_trials": n_boxes**2,
    }


def _parse_box_answer(text: str, n_boxes: int) -> tuple:
    """Parse box number from answer tags."""
    match = ANSWER_TAG_RE.search(text or "")
    if not match:
        return None, "invalid_format"

    answer_text = match.group(1).strip()
    nums = re.findall(r"\d+", answer_text)
    if not nums:
        return None, "invalid_format"

    try:
        box = int(nums[0])
        if 1 <= box <= n_boxes:
            return box, "valid"
        else:
            return None, "invalid_choice"
    except ValueError:
        return None, "invalid_format"


def _extract_config_from_prompt(prompt_content: str) -> Dict[str, Any]:
    """Extract n_boxes and tokens from the initial prompt text.

    This allows state recovery when info is not passed to env_response.
    """
    config = {}

    # Extract n_boxes: look for "There are N boxes" or "1 to N"
    n_boxes_match = re.search(r"There are (\d+) boxes", prompt_content)
    if n_boxes_match:
        config["n_boxes"] = int(n_boxes_match.group(1))
    else:
        # Fallback: look for "1-N" pattern
        range_match = re.search(r"1-(\d+)", prompt_content)
        if range_match:
            config["n_boxes"] = int(range_match.group(1))

    # Extract tokens: look for "types of tokens (A, B)" or similar
    tokens_match = re.search(r"types of tokens \(([A-Z, ]+)\)", prompt_content)
    if tokens_match:
        tokens_str = tokens_match.group(1)
        config["tokens"] = [t.strip() for t in tokens_str.split(",")]
    else:
        # Fallback: check for multi-token status lines
        if "A tokens found" in prompt_content and "B tokens found" in prompt_content:
            config["tokens"] = ["A", "B"]
        elif "A tokens found" in prompt_content:
            config["tokens"] = ["A"]

    return config


def create_swm_dataset(
    n_boxes: int = 8,
    n_tokens: int = 1,
    max_trials: int = 64,
    num_episodes: int = 10,
    seed: Optional[int] = None,
    feedback_mode: str = "text",  # "text", "image+text", "image-only"
    enable_thinking: bool = False,  # Whether to include thinking instructions
) -> Dataset:
    """Create dataset with procedurally generated SWM episodes.

    Args:
        n_boxes: Number of boxes (e.g., 8 for easy, 12 for hard)
        n_tokens: Number of tokens to find (e.g., 1 for easy, 2 for hard)
        max_trials: Maximum number of trials per episode
        num_episodes: Number of episodes to generate
        seed: Random seed for reproducibility
        feedback_mode: Feedback type - "text", "image+text", or "image-only"
        enable_thinking: Whether to include thinking instructions in prompts
    """
    rows = []

    for i in range(num_episodes):
        episode_seed = (seed + i) if seed is not None else None
        episode_data = _generate_swm_example(n_boxes, n_tokens, episode_seed)
        episode_data["max_trials"] = max_trials
        episode_data["feedback_mode"] = feedback_mode

        tokens = episode_data["tokens"]

        # Build status for ALL tokens (searched simultaneously)
        token_status = "\n".join([f"{t} tokens found: 0/{n_boxes}" for t in tokens])

        initial_prompt = f"""You are playing the Spatial Working Memory (SWM) game.

There are {n_boxes} boxes numbered 1 to {n_boxes}.
There are {len(tokens)} types of tokens ({', '.join(tokens)}), each hidden in one box.
Your goal is to find each token type {n_boxes} times total.
A box can contain multiple token types at once.
Once a token is found, it regenerates in a box that has never contained that token type before.
Remember which boxes you've opened - opening the same box twice in a search is penalized.
When ANY token is found, the search resets (opened boxes cleared).

{token_status}
Boxes opened this search: none

Which box do you want to open? Answer with a number 1-{n_boxes} in <answer> tags."""

        # Add thinking instruction or /no_think suffix
        if enable_thinking:
            initial_prompt += "\n\nThink step-by-step about which boxes you've tried in maximum 1000 tokens, wrapped with <think> and </think>. Then provide your final answer."
        else:
            initial_prompt += " /no_think"

        example_id = f"swm_{feedback_mode}_{n_boxes}box_{n_tokens}tok_{i}"
        episode_data["example_id"] = example_id

        rows.append(
            {
                "example_id": example_id,
                "prompt": [{"role": "user", "content": initial_prompt}],
                "info": episode_data,
            }
        )

    return Dataset.from_list(rows)


class SWMMultiTurnEnv(MultiTurnEnv):
    """SWM environment using Multi-Turn-RL-Agent's simpler synchronous API.

    Easy setup: 8 boxes, 1 token
    Hard setup: 12 boxes, 2 tokens

    Game logic:
    - legal_boxes[token]: boxes that can still contain this token (starts full, shrinks)
    - opened_boxes: boxes opened in current search (resets when ANY token found)
    - Model searches for ALL tokens simultaneously

    State is stored in MODULE-LEVEL dicts to persist across environment instances.
    """

    def __init__(self, dataset: Dataset, enable_thinking: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._dataset = dataset
        self._enable_thinking = enable_thinking

        # PRE-INDEX: Build a mapping from prompt hash to info (use global storage)
        # This allows env_response to access confidential info (token_box, etc.)
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
        return [self.compute_reward]

    def _get_prompt_key(self, messages: List[Dict[str, str]]) -> str:
        """Get key based on prompt only (for info lookup)."""
        if messages:
            return f"swm_prompt_{hash(messages[0].get('content', ''))}"
        return "swm_prompt_default"

    def _get_state_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate a unique key for this specific trajectory's state.

        Uses prompt hash + trajectory signature to ensure each parallel
        generation has its own isolated state.

        The key includes ALL messages after the initial prompt (both assistant
        and user feedback), so different generations diverge as soon as they
        get different feedback.

        For the first turn (only 1 assistant message, no user feedback yet),
        we use id(messages) to ensure uniqueness even if multiple generations
        produce the same first answer.
        """
        if not messages:
            return "swm_default"

        # Base key from prompt
        prompt_key = hash(messages[0].get("content", ""))

        # Get all messages after the initial prompt
        subsequent_msgs = messages[1:]  # Skip initial user prompt

        # Count assistant and user messages (excluding initial prompt)
        assistant_count = sum(
            1 for m in subsequent_msgs if m.get("role") == "assistant"
        )
        user_feedback_count = sum(1 for m in subsequent_msgs if m.get("role") == "user")

        if assistant_count == 1 and user_feedback_count == 0:
            # First turn: only 1 assistant response, no user feedback yet
            # Use id(messages) to ensure each parallel generation gets unique state
            trajectory_sig = f"id_{id(messages)}"
        else:
            # After first feedback: use hash of ALL subsequent content
            # This includes assistant responses AND user feedback
            all_contents = tuple(m.get("content", "") for m in subsequent_msgs)
            trajectory_sig = f"content_{hash(all_contents)}"

        return f"swm_{prompt_key}_{trajectory_sig}"

    def _get_info_for_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Look up the original info for this episode from global pre-indexed data."""
        if not messages:
            return {}
        # Use prompt-only key for info lookup (shared across generations)
        key = self._get_prompt_key(messages)
        return _GLOBAL_PROMPT_TO_INFO.get(key, {})

    def _get_or_create_state(
        self, messages: List[Dict[str, str]], info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get existing state or create new one if this is the first call.

        Uses MODULE-LEVEL storage so state persists across environment instances.
        Each parallel generation gets its own isolated state via trajectory-specific keys.

        State migration: When a generation moves from turn 1 (id-based key) to turn 2+
        (content-based key), we look for the parent state and copy it.
        """
        if not messages:
            return {}

        key = self._get_state_key(messages)

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
            prompt_config = _extract_config_from_prompt(first_msg)
            info = prompt_config

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
                        if stored_key.startswith(f"swm_{prompt_key}_id_"):
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
                        prev_key = f"swm_{prompt_key}_content_{hash(all_contents)}"
                        parent_state = _GLOBAL_EPISODE_STATES.get(prev_key)

        if parent_state:
            # Copy parent state to new key
            import copy

            _GLOBAL_EPISODE_STATES[key] = copy.deepcopy(parent_state)
            return _GLOBAL_EPISODE_STATES[key]

        # No parent state - create fresh initial state
        tokens = info.get("tokens", ["A"])
        n_boxes = info.get("n_boxes", 8)
        max_trials = info.get("max_trials", n_boxes**2)
        legal_boxes_from_info = info.get("legal_boxes", {})
        token_box_from_info = info.get("token_box", {})

        # legal_boxes: which boxes can still contain each token (starts full)
        if legal_boxes_from_info:
            # Convert lists to sets for efficient operations
            legal_boxes = {
                t: set(legal_boxes_from_info.get(t, range(1, n_boxes + 1)))
                for t in tokens
            }
        else:
            legal_boxes = {t: set(range(1, n_boxes + 1)) for t in tokens}

        # token_box: current location of each token
        if token_box_from_info:
            token_box = dict(token_box_from_info)
        else:
            # Generate random initial positions if not provided
            token_box = {}
            for token in tokens:
                legal_for_token = list(legal_boxes.get(token, range(1, n_boxes + 1)))
                token_box[token] = (
                    random.choice(legal_for_token) if legal_for_token else 1
                )

        _GLOBAL_EPISODE_STATES[key] = {
            "tokens": tokens,
            "n_boxes": n_boxes,
            "max_trials": max_trials,
            "tokens_found": {t: 0 for t in tokens},  # times each token has been found
            "legal_boxes": legal_boxes,  # boxes that can still contain each token
            "token_box": token_box,  # current location of each token
            "opened_boxes": set(),  # boxes opened in current search
            "trial_idx": 0,
            # Store first assistant response for parent lookup in turn 2
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
        if not messages:
            return False

        last_msg = messages[-1].get("content", "")
        # Check for explicit completion markers from env_response
        if "Game complete" in last_msg:
            return True
        if "All tokens found" in last_msg:
            return True
        if "Max trials" in last_msg:
            return True

        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        return len(assistant_msgs) >= self.max_steps

    def env_response(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        """Provide environment response after agent's choice."""
        if not messages:
            return {"role": "user", "content": "Please choose a box."}

        # Get last assistant message
        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        if not last_assistant:
            return {
                "role": "user",
                "content": "Please choose a box number in <answer> tags.",
            }

        # Get info from kwargs (may be empty in training framework calls)
        info = kwargs.get("info", {})

        # Get or create state - extracts config from prompt if info not available
        state = self._get_or_create_state(messages, info)

        # Extract values from state
        n_boxes = state["n_boxes"]
        tokens = state["tokens"]
        max_trials = state["max_trials"]

        no_think_suffix = "" if self._enable_thinking else " /no_think"

        # Parse answer
        box, status = _parse_box_answer(last_assistant, n_boxes)

        if status != "valid" or box is None:
            return {
                "role": "user",
                "content": f"Please answer with a number 1-{n_boxes} in <answer> tags.{no_think_suffix}",
            }

        state["trial_idx"] += 1

        # Check if max trials exceeded
        if state["trial_idx"] > max_trials:
            return {
                "role": "user",
                "content": f"Game complete! Max trials ({max_trials}) reached.{no_think_suffix}",
            }

        # Check if box was already opened in current search
        if box in state["opened_boxes"]:
            feedback = (
                f"You already opened box {box} in this search. Try a different box."
            )
            state["opened_boxes"].add(box)  # Still track it
        else:
            state["opened_boxes"].add(box)

            # Check which tokens (if any) are in this box
            found_tokens = []
            for token in tokens:
                if state["token_box"].get(token) == box:
                    found_tokens.append(token)

            if found_tokens:
                # Found one or more tokens!
                feedback_parts = []
                for token in found_tokens:
                    feedback_parts.append(f"Token {token} found in box {box}!")
                    state["tokens_found"][token] = (
                        state["tokens_found"].get(token, 0) + 1
                    )

                    # Remove this box from legal_boxes for this token
                    state["legal_boxes"][token].discard(box)

                    # Regenerate token in a new legal box (if any remain)
                    legal_for_token = state["legal_boxes"].get(token, set())
                    if legal_for_token:
                        new_box = random.choice(list(legal_for_token))
                        state["token_box"][token] = new_box
                    else:
                        state["token_box"][token] = None

                feedback = " ".join(feedback_parts)

                # Reset opened_boxes for new search
                state["opened_boxes"] = set()
            else:
                # Empty box
                feedback = f"Box {box} is empty."

        # Check if game is complete (all tokens found n_boxes times)
        all_done = all(state["tokens_found"].get(t, 0) >= n_boxes for t in tokens)
        if all_done:
            return {
                "role": "user",
                "content": f"{feedback} Game complete! All tokens found {n_boxes} times each.{no_think_suffix}",
            }

        # Build status message
        token_status = "\n".join(
            [
                f"{t} tokens found: {state['tokens_found'].get(t, 0)}/{n_boxes}"
                for t in tokens
            ]
        )

        opened_list = sorted(state["opened_boxes"])
        opened_str = str(opened_list) if opened_list else "none"

        response = f"""{feedback}

{token_status}
Boxes opened this search: {opened_str}

Which box do you want to open?{no_think_suffix}"""

        return {"role": "user", "content": response}

    def compute_reward(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        rewards = []

        for prompt, completion in zip(prompts, completions):
            full_msgs = prompt + completion

            # Get state for this episode
            state = self._get_or_create_state(full_msgs)
            n_boxes = state.get("n_boxes", 8)

            # Count outcomes from completion messages
            total_reward = 0.0
            for msg in completion:
                content = msg.get("content", "")
                if "Token" in content and "found" in content:
                    total_reward += REWARD_TOKEN_FOUND
                elif "already opened" in content.lower():
                    total_reward += REWARD_REPEATED_BOX
                elif "empty" in content.lower():
                    total_reward += REWARD_VALID_GUESS
                elif "Please answer" in content:
                    total_reward += REWARD_INVALID_FORMAT

            rewards.append(total_reward)

        return rewards


def load_environment(
    mode: str = "text",
    n_boxes: int = 8,
    n_tokens: int = 1,
    eval_data: Optional[str] = None,
    num_episodes: int = 20,
    max_trials: int = 50,
    seed: int = 42,
    enable_thinking: bool = False,
    **kwargs,
) -> SWMMultiTurnEnv:
    """Load SWM environment with specified configuration.

    Args:
        mode: "text", "image+text", or "image-only"
        n_boxes: Number of boxes (8 for easy, 12 for hard)
        n_tokens: Number of tokens (1 for easy, 2 for hard)
        eval_data: Path to evaluation data (not used, generates procedurally)
        num_episodes: Number of episodes to generate
        max_trials: Maximum trials per episode
        seed: Random seed
        enable_thinking: Whether to include thinking instructions
    """
    dataset = create_swm_dataset(
        n_boxes=n_boxes,
        n_tokens=n_tokens,
        max_trials=max_trials,
        num_episodes=num_episodes,
        seed=seed,
        feedback_mode=mode,
        enable_thinking=enable_thinking,
    )

    return SWMMultiTurnEnv(dataset, enable_thinking=enable_thinking, **kwargs)
