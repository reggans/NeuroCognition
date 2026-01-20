"""
Multi-Turn Environment for RAPM using Multi-Turn-RL-Agent's verifiers.
"""

import json
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional

from datasets import Dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.envs import MultiTurnEnv

# Reward constants
REWARD_CORRECT = 1.0
REWARD_WRONG = -0.1
REWARD_INVALID_FORMAT = -1.0

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# MODULE-LEVEL STATE STORAGE
# These persist across environment instances (important for multi-worker training)
_GLOBAL_EPISODE_STATES: Dict[str, Dict[str, Any]] = {}
_GLOBAL_PROMPT_TO_INFO: Dict[str, Dict[str, Any]] = {}
_GLOBAL_STATE_LOCK = threading.Lock()  # Thread-safety for concurrent access


def _extract_config_from_prompt(prompt_content: str) -> Dict[str, Any]:
    """Extract answer_mode, max_options, and correct_answer from the initial prompt text.

    This allows state recovery when info is not passed to env_response.
    """
    config = {}

    # Determine answer_mode: "gen" if generative, "mc" if multiple choice
    if "generate the missing cell's value" in prompt_content.lower():
        config["answer_mode"] = "gen"
    else:
        config["answer_mode"] = "mc"

    # Extract max_options from "1-N" pattern or "Options:" section
    range_match = re.search(r"Select the correct answer \(1-(\d+)\)", prompt_content)
    if range_match:
        config["max_options"] = int(range_match.group(1))
    else:
        # Count options if present
        options_match = re.findall(r"^\d+\.", prompt_content, re.MULTILINE)
        if options_match:
            config["max_options"] = len(options_match)

    # Default max_turns based on max_options
    config["max_turns"] = config.get("max_options", 8)

    return config


def _parse_answer(text: str, max_options: int = 8, answer_mode: str = "mc") -> tuple:
    """Parse answer from answer tags.

    Args:
        text: The response text
        max_options: Maximum number of options for mc mode
        answer_mode: "mc" for multiple choice (expects number), "gen" for generative (any string)
    """
    match = ANSWER_TAG_RE.search(text or "")
    if not match:
        return None, "invalid_format"

    answer_text = match.group(1).strip()

    if answer_mode == "gen":
        # Generative mode: accept any non-empty string
        if answer_text:
            return answer_text, "valid"
        else:
            return None, "invalid_format"
    else:
        # Multiple choice mode: expect a number 1-N
        nums = re.findall(r"\d+", answer_text)
        if not nums:
            return None, "invalid_format"

        try:
            choice = int(nums[0])
            if 1 <= choice <= max_options:
                return choice, "valid"
            else:
                return None, "invalid_choice"
        except ValueError:
            return None, "invalid_format"


def _load_rapm_image_dataset(
    path: str,
    image_base_path: Optional[str] = None,
    limit: Optional[int] = None,
    answer_mode: str = "mc",  # "mc" = multiple choice (image only supports mc)
    enable_thinking: bool = False,  # Whether to include thinking instructions
) -> Dataset:
    """Load RAPM image dataset.

    Args:
        path: Path to JSON file with image questions
        image_base_path: Base path for images
        limit: Max number of examples
        answer_mode: "mc" only for image mode
        enable_thinking: Whether to include thinking instructions
    """
    with open(path, "r") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    if limit is not None:
        questions = questions[:limit]

    rows = []
    for i, q in enumerate(questions):
        correct_idx = q.get("correct_answer", -1)
        answer = (
            correct_idx + 1 if isinstance(correct_idx, int) and correct_idx >= 0 else -1
        )
        img_path = q.get("full_image") or q.get("image_path")
        if image_base_path and img_path:
            img_path = os.path.join(image_base_path, img_path)

        initial_prompt = """You are taking the Raven's Progressive Matrices (RPM) test.
You will see a 3x3 matrix with the bottom-right image missing.
Select the correct answer from options 1-8 that completes the pattern.

State your final answer as: <answer>number</answer>"""

        # Add thinking instruction or /no_think suffix
        if enable_thinking:
            initial_prompt += "\n\nThink step-by-step about the visual patterns in maximum 1000 tokens, wrapped with <think> and </think>. Then provide your final answer."
        else:
            initial_prompt += " /no_think"

        example_id = f"rapm_image_mc_{i}"
        rows.append(
            {
                "example_id": example_id,
                "prompt": [{"role": "user", "content": initial_prompt}],
                "info": {
                    "example_id": example_id,  # Add to info for state tracking
                    "question_id": q.get("id", f"img_{i}"),
                    "image_path": img_path,
                    "max_turns": 8,
                    "mode": "image",
                    "answer_mode": "mc",  # Image mode always uses mc
                    "correct_answer": str(answer),  # Store as string for consistency
                },
            }
        )
    return Dataset.from_list(rows)


def _load_rapm_text_dataset(
    path: str,
    limit: Optional[int] = None,
    answer_mode: str = "mc",  # "mc" = multiple choice, "gen" = generative
    enable_thinking: bool = False,  # Whether to include thinking instructions
) -> Dataset:
    """Load RAPM text dataset.

    Args:
        path: Path to JSONL file with text questions
        limit: Max number of examples
        answer_mode: "mc" for multiple choice, "gen" for generative
        enable_thinking: Whether to include thinking instructions
    """
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            item = json.loads(line.strip())

            # Handle both 'matrix' and 'question_grid' field names
            matrix = item.get("matrix") or item.get("question_grid", [])
            options = item.get("options", [])
            answer = item.get("answer", "")
            correct_idx = item.get("correct_index")
            constraints = item.get("constraints") or item.get("cell_constraints", {})

            # Build matrix string representation
            if matrix:
                matrix_rows = []
                for row in matrix:
                    # Handle None values (missing cells) with '?'
                    row_strs = [cell if cell is not None else "?" for cell in row]
                    matrix_rows.append(" | ".join(row_strs))
                matrix_str = "\n".join(matrix_rows)
            else:
                matrix_str = "(Matrix not available)"

            if answer_mode == "gen":
                # Generative mode: ask for the actual answer string
                initial_prompt = f"""You are solving a TEXT-BASED 3x3 pattern matrix.
Each cell contains a string; the bottom-right cell is missing ('?').

Matrix:
{matrix_str}

Determine the pattern and generate the missing cell's value.
State your final answer as: <answer>your_answer</answer>"""
            else:
                # Multiple choice mode
                options_str = "\n".join(
                    [f"{j+1}. {opt}" for j, opt in enumerate(options)]
                )
                initial_prompt = f"""You are solving a TEXT-BASED 3x3 pattern matrix.
Each cell contains a string; the bottom-right cell is missing ('?').

Matrix:
{matrix_str}

Options:
{options_str}

Select the correct answer (1-{len(options)}) that completes the pattern.
State your final answer as: <answer>number</answer>"""

            # Add thinking instruction or /no_think suffix
            if enable_thinking:
                initial_prompt += "\n\nThink step-by-step about the pattern in maximum 1000 tokens, wrapped with <think> and </think>. Then provide your final answer."
            else:
                initial_prompt += " /no_think"

            example_id = f"rapm_text_{answer_mode}_{i}"
            rows.append(
                {
                    "example_id": example_id,
                    "prompt": [{"role": "user", "content": initial_prompt}],
                    "info": {
                        "example_id": example_id,  # Add to info for state tracking
                        "question_id": item.get("id", f"text_{i}"),
                        "matrix": matrix,
                        "options": options,
                        "constraints": constraints,
                        "max_turns": (
                            len(options) if options else 8
                        ),  # Use number of options as max turns
                        "mode": "text",
                        "answer_mode": answer_mode,
                        "correct_answer": (
                            answer
                            if answer
                            else (correct_idx + 1 if correct_idx is not None else None)
                        ),
                    },
                }
            )
    return Dataset.from_list(rows)


class RAPMMultiTurnEnv(MultiTurnEnv):
    """RAPM environment using Multi-Turn-RL-Agent's simpler synchronous API.

    Supports two answer modes:
    - mc (multiple choice): Answer must be 1-N (option number)
    - gen (generative): Answer can be any string, validated against correct answer

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
        return [self.compute_reward]

    def _get_prompt_key(self, messages: List[Dict[str, str]]) -> str:
        """Get key based on prompt only (for info lookup)."""
        if messages:
            return f"rapm_prompt_{hash(messages[0].get('content', ''))}"
        return "rapm_prompt_default"

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
            return "rapm_default"

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

        return f"rapm_{prompt_key}_{trajectory_sig}"

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
        """Get existing state or create new one for this trajectory.

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
                        if stored_key.startswith(f"rapm_{prompt_key}_id_"):
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
                        prev_key = f"rapm_{prompt_key}_content_{hash(all_contents)}"
                        parent_state = _GLOBAL_EPISODE_STATES.get(prev_key)

        if parent_state:
            # Copy parent state to new key
            import copy

            _GLOBAL_EPISODE_STATES[key] = copy.deepcopy(parent_state)
            return _GLOBAL_EPISODE_STATES[key]

        # No parent state - create fresh initial state
        answer_mode = info.get("answer_mode", "mc")
        options = info.get("options", [])
        max_options = len(options) if options else info.get("max_options", 8)
        correct_answer = info.get("correct_answer")
        max_turns = info.get("max_turns", 8)

        _GLOBAL_EPISODE_STATES[key] = {
            "turn_idx": 0,
            "attempts": [],
            "answer_mode": answer_mode,
            "max_options": max_options,
            "correct_answer": correct_answer,
            "max_turns": max_turns,
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
        if not messages:
            return False

        last_msg = messages[-1].get("content", "")
        # Check for explicit completion markers
        if "RAPM completed" in last_msg:
            return True
        if "Correct!" in last_msg and "is right" in last_msg:
            return True
        if "Maximum attempts reached" in last_msg:
            return True

        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        return len(assistant_msgs) >= self.max_steps

    def env_response(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        """Provide environment response after agent's answer.

        For text-mc: Answer must be 1-N, correct if matches correct option index
        For text-gen: Answer can be any string, correct if matches correct answer string
        """
        if not messages:
            return {"role": "user", "content": "Please select an answer."}

        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        if not last_assistant:
            return {
                "role": "user",
                "content": "Please provide your answer in <answer> tags.",
            }

        info = kwargs.get("info", {})

        # Get state first - it may have stored values from initial call with info
        state = self._get_state_for_messages(messages, info)

        # Use state values (set during first call with info), fallback to info, then defaults
        answer_mode = state.get("answer_mode") or info.get("answer_mode", "mc")
        max_options = state.get("max_options") or len(info.get("options") or []) or 8
        correct = state.get("correct_answer") or info.get("correct_answer")
        max_turns = state.get("max_turns") or info.get("max_turns", 8)

        no_think_suffix = "" if self._enable_thinking else " /no_think"

        # Parse answer based on mode
        choice, status = _parse_answer(last_assistant, max_options, answer_mode)

        # Always count as an attempt, even for invalid format
        state["turn_idx"] = state.get("turn_idx", 0) + 1

        # Check if max attempts reached BEFORE processing answer
        if state["turn_idx"] > max_turns:
            return {
                "role": "user",
                "content": f"Maximum attempts reached. The correct answer was '{correct}'. RAPM completed.{no_think_suffix}",
            }

        if status != "valid" or choice is None:
            if answer_mode == "gen":
                return {
                    "role": "user",
                    "content": f"Invalid format. Please provide your answer inside <answer> tags. ({max_turns - state['turn_idx']} attempts remaining){no_think_suffix}",
                }
            else:
                return {
                    "role": "user",
                    "content": f"Invalid format. Please answer with a number 1-{max_options} in <answer> tags. ({max_turns - state['turn_idx']} attempts remaining){no_think_suffix}",
                }

        # Update state - turn already incremented above
        state["attempts"].append(choice)

        # Check correctness
        if answer_mode == "gen":
            # Generative mode: compare strings (case-insensitive, strip whitespace)
            is_correct = str(choice).strip().lower() == str(correct).strip().lower()
        else:
            # MC mode: compare numbers
            is_correct = choice == correct or str(choice) == str(correct)

        if is_correct:
            return {
                "role": "user",
                "content": f"Correct! Your answer '{choice}' is right. RAPM completed.{no_think_suffix}",
            }

        if state["turn_idx"] >= max_turns:
            return {
                "role": "user",
                "content": f"Maximum attempts reached. The correct answer was '{correct}'. RAPM completed.{no_think_suffix}",
            }

        return {
            "role": "user",
            "content": f"Incorrect. Try again. ({max_turns - state['turn_idx']} attempts remaining){no_think_suffix}",
        }

    def compute_reward(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        rewards = []
        infos = kwargs.get("info", [{}] * len(prompts))

        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            total_reward = 0.0
            info = infos[idx] if idx < len(infos) else {}
            correct_answer = info.get("correct_answer")
            answer_mode = info.get("answer_mode", "mc")
            max_options = len(info.get("options", [])) or 8

            for msg in completion:
                if msg.get("role") != "assistant":
                    continue

                content = msg.get("content", "")
                choice, status = _parse_answer(content, max_options, answer_mode)

                if status != "valid":
                    total_reward += REWARD_INVALID_FORMAT
                elif correct_answer:
                    if answer_mode == "gen":
                        # Generative: compare strings
                        if (
                            str(choice).strip().lower()
                            == str(correct_answer).strip().lower()
                        ):
                            total_reward += REWARD_CORRECT
                        else:
                            total_reward += REWARD_WRONG
                    else:
                        # MC: compare numbers
                        if str(choice) == str(correct_answer):
                            total_reward += REWARD_CORRECT
                        else:
                            total_reward += REWARD_WRONG
                else:
                    total_reward += REWARD_WRONG

            rewards.append(total_reward)

        return rewards


def load_environment(
    mode: str = "text",
    answer_mode: str = "mc",  # "mc" = multiple choice, "gen" = generative
    eval_data: Optional[str] = None,
    image_base_path: Optional[str] = None,
    num_episodes: int = 10,
    limit: Optional[int] = None,
    max_steps: int = 20,  # Increased: allow up to 20 turns per episode (RAPM has fewer turns)
    max_episode_tokens: int = 32768,  # Total tokens for entire episode
    enable_thinking: bool = False,  # Whether to include thinking instructions
    **kwargs,
) -> RAPMMultiTurnEnv:
    """Load RAPM MultiTurn environment.

    Args:
        mode: "text" or "image"
        answer_mode: "mc" for multiple choice, "gen" for generative (text-mode only)
        eval_data: Path to evaluation data file
        image_base_path: Base path for images
        num_episodes: Number of episodes
        limit: Limit on number of examples
        max_steps: Maximum conversation turns (default 20)
        max_episode_tokens: Total token budget for entire episode (default 32768)
        enable_thinking: Whether to include thinking instructions in prompts
    """
    # Set default image base path if not provided
    if image_base_path is None:
        image_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "rapm_data", "images"
        )

    if mode == "image":
        if not eval_data:
            raise ValueError("eval_data path required for image mode")
        # Image mode always uses mc (answer_mode is ignored for images)
        dataset = _load_rapm_image_dataset(
            eval_data,
            image_base_path,
            limit or num_episodes,
            enable_thinking=enable_thinking,
        )
    else:
        if not eval_data:
            # Use default text data path from rapm_data
            eval_data = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "rapm_data",
                "text_rapm_train.jsonl",
            )
        dataset = _load_rapm_text_dataset(
            eval_data,
            limit or num_episodes,
            answer_mode=answer_mode,
            enable_thinking=enable_thinking,
        )

    return RAPMMultiTurnEnv(
        dataset=dataset,
        enable_thinking=enable_thinking,
        max_steps=max_steps,
        max_episode_tokens=max_episode_tokens,
        **kwargs,
    )
