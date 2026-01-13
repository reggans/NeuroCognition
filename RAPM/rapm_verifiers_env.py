"""
Verifiers environment for RAPM (Raven's Progressive Matrices).
Mirrors the RL reward design from rapm_env.py without modifying it.

Supports:
- Image-MC: max 8 turns, -0.1 per wrong, -1.0 final penalty if max turns reached
- Text-MC: max turns = # constraints, -0.1 × violations per wrong turn, -1.0 final penalty
- Text-Gen: same as text-MC but generates strings instead of selecting from options
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from datasets import Dataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import verifiers as vf
except ImportError:
    vf = None  # type: ignore

# Import existing validators from the RAPM module
try:
    from RAPM.text_rapm.per_cell_constraints import CellConstraint
    from RAPM.text_rapm.validator import cell_satisfies, constraint_violations
except ImportError:
    CellConstraint = None
    cell_satisfies = None
    constraint_violations = None

from RAPM.rapm_rubric import RAPMRubric

# Reward constants (matching rapm_env.py)
REWARD_CORRECT = 1.0
REWARD_WRONG_MULTITURN = -0.1
REWARD_CONSTRAINT_VIOLATION = -0.1
REWARD_MAX_TURNS_FAILED = -1.0
REWARD_INVALID_FORMAT = -0.5

# System prompts
IMAGE_BASE_PROMPT = (
    "You are taking the Raven's Progressive Matrices (RPM) test, a non-verbal intelligence test that measures abstract reasoning ability.\n\n"
    "You will see a 3x3 matrix of images with the bottom-right image missing (shown as a question mark), followed by 8 answer choices numbered 1-8.\n\n"
    "Your task is to: \n1. Analyze rows and columns\n2. Infer the governing logical rule(s)\n3. Select the answer choice (1-8) that correctly completes the matrix.\n\n"
    "State your final answer as: <answer>number</answer>\n"
)

IMAGE_PATTERN_INFO = (
    "The patterns can involve: \n"
    "- Shape transformations (rotation, reflection, scaling)\n"
    "- Position changes (movement, arrangement)\n"
    "- Attribute changes (color, size, number of elements)\n"
    "- Logical operations (addition, subtraction, intersection)\n"
    "- Sequence progressions (systematic changes across rows/columns)\n\n"
)

TEXT_BASE_PROMPT = (
    "You are solving a TEXT-BASED 3x3 pattern matrix (Raven-style). Each cell contains a string; the bottom-right cell is missing ('?').\n\n"
    "Goal: Infer the rule(s) acting across rows and columns.\n\n"
)

TEXT_PATTERN_INFO = (
    "Possible dimensions (one or more):\n"
    "- Character set restriction (digits / letters / symbols)\n"
    "- Quantitative constant (exact length / count / unique)\n"
    "- Quantitative progression (arithmetic step across row/column)\n"
    "- Parity / multiple rules (all even / all odd / multiples of N)\n"
    "- Positional constraints (first/last/even/odd positions restricted)\n"
    "- Ordering (ascending / descending / mixed)\n"
    "- Layered combinations (e.g. constant + parity, progression + positional)\n\n"
)

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _load_rapm_image_dataset(
    path: str, image_base_path: Optional[str] = None, limit: Optional[int] = None
) -> Dataset:
    """Load RAPM image dataset into HF Dataset with prompt/answer/info fields."""
    with open(path, "r") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    if limit is not None:
        questions = questions[:limit]

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(questions):
        correct_idx = q.get("correct_answer", -1)
        # correct_answer in source is 0-indexed; we store as 1-indexed for answer
        answer = (
            correct_idx + 1 if isinstance(correct_idx, int) and correct_idx >= 0 else -1
        )
        img_path = q.get("full_image") or q.get("image_path")
        if image_base_path and img_path:
            img_path = os.path.join(image_base_path, img_path)

        rows.append(
            {
                "example_id": f"rapm_image_{i}",
                "prompt": [
                    {
                        "role": "user",
                        "content": "Analyze the image matrix and select the correct answer (1-8).",
                    }
                ],
                "answer": str(answer),
                "info": {
                    "question_id": q.get("id", f"img_{i}"),
                    "image_path": img_path,
                    "max_turns": 8,
                    "mode": "image",
                    "answer_mode": "mc",
                },
            }
        )
    return Dataset.from_list(rows)


def _load_rapm_text_dataset(
    path: str, answer_mode: str = "mc", limit: Optional[int] = None
) -> Dataset:
    """Load RAPM text dataset (MC or generative)."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        questions = questions[:limit]

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(questions):
        grid = q.get("question_grid") or q.get("full_grid") or []
        options = q.get("options", [])
        correct_idx = q.get("correct_index", -1)

        # Build grid text
        grid_lines = []
        for r in range(3):
            row_cells = []
            for c in range(3):
                v = grid[r][c] if r < len(grid) and c < len(grid[r]) else None
                row_cells.append("?" if v is None else str(v))
            grid_lines.append(" | ".join(row_cells))
        grid_text = "\n".join(grid_lines)

        # Extract cell constraint for cell 2,2
        cell_constraint_data = None
        if CellConstraint is not None:
            cell_constraints = q.get("cell_constraints", {})
            constraint_22 = cell_constraints.get("2,2")
            if constraint_22:
                try:
                    cell_constraint_data = {
                        "fixed_length": constraint_22.get("fixed_length"),
                        "target_counts": constraint_22.get("target_counts", {}),
                        "parity_rules": constraint_22.get("parity_rules", {}),
                        "multiple_rules": constraint_22.get("multiple_rules", {}),
                        "unique_exact": constraint_22.get("unique_exact"),
                        "ordering": constraint_22.get("ordering"),
                        "positional_type": constraint_22.get("positional_type"),
                        "positional_index_rule": constraint_22.get(
                            "positional_index_rule"
                        ),
                    }
                except Exception:
                    pass

        # Count total constraints
        total_constraints = 1
        if cell_constraint_data:
            count = 0
            if cell_constraint_data.get("fixed_length") is not None:
                count += 1
            if cell_constraint_data.get("target_counts"):
                count += len(cell_constraint_data["target_counts"])
            if cell_constraint_data.get("parity_rules"):
                count += len(cell_constraint_data["parity_rules"])
            if cell_constraint_data.get("multiple_rules"):
                count += len(cell_constraint_data["multiple_rules"])
            if cell_constraint_data.get("unique_exact") is not None:
                count += 1
            if cell_constraint_data.get("ordering") is not None:
                count += 1
            if cell_constraint_data.get("positional_type") is not None:
                count += 1
            total_constraints = max(count, 1)

        if answer_mode == "mc":
            opt_lines = [f"{j+1}. {o}" for j, o in enumerate(options)]
            prompt_text = (
                f"Matrix:\n{grid_text}\n\nOptions:\n"
                + "\n".join(opt_lines)
                + "\n\nAnswer with <answer>N</answer>."
            )
            # Answer is 1-indexed choice
            answer_str = str(correct_idx + 1) if correct_idx >= 0 else "-1"
        else:
            # Generative mode
            prompt_text = f"Matrix:\n{grid_text}\n\nGenerate the missing cell. Answer with <answer>STRING</answer>."
            # Answer is the actual string
            answer_str = q.get("answer", "")

        rows.append(
            {
                "example_id": f"rapm_text_{i}",
                "prompt": [{"role": "user", "content": prompt_text}],
                "answer": answer_str,
                "info": {
                    "question_id": q.get("id", f"text_{i}"),
                    "max_turns": total_constraints,
                    "mode": "text",
                    "answer_mode": answer_mode,
                    "options": options,
                    "cell_constraint": cell_constraint_data,
                    "correct_index": correct_idx,
                },
            }
        )
    return Dataset.from_list(rows)


def _parse_numeric_answer(text: str) -> Optional[int]:
    """Parse numeric answer from <answer> tags (1-8 for image-mc or text-mc)."""
    match = ANSWER_TAG_RE.search(text or "")
    if not match:
        return None
    nums = re.findall(r"\d+", match.group(1).strip())
    if not nums:
        return None
    try:
        n = int(nums[0])
        return n if 1 <= n <= 8 else None
    except ValueError:
        return None


def _parse_string_answer(text: str) -> Optional[str]:
    """Parse string answer from <answer> tags (for text-gen)."""
    match = ANSWER_TAG_RE.search(text or "")
    if not match:
        return None
    return match.group(1).strip().strip('"')


def _count_constraint_violations(
    answer_str: str, cell_constraint_data: Optional[Dict]
) -> int:
    """Count constraint violations using existing validator."""
    if (
        cell_constraint_data is None
        or constraint_violations is None
        or CellConstraint is None
    ):
        return 1
    try:
        cc = CellConstraint(
            fixed_length=cell_constraint_data.get("fixed_length"),
            target_counts=cell_constraint_data.get("target_counts", {}),
            parity_rules=cell_constraint_data.get("parity_rules", {}),
            multiple_rules=cell_constraint_data.get("multiple_rules", {}),
            unique_exact=cell_constraint_data.get("unique_exact"),
            ordering=cell_constraint_data.get("ordering"),
            positional_type=cell_constraint_data.get("positional_type"),
            positional_index_rule=cell_constraint_data.get("positional_index_rule"),
        )
        violations = constraint_violations(answer_str, cc)
        return len(violations)
    except Exception:
        return 1


def _check_text_gen_correct(
    answer_str: str, cell_constraint_data: Optional[Dict]
) -> bool:
    """Check if text-gen answer satisfies all constraints."""
    if cell_constraint_data is None or cell_satisfies is None or CellConstraint is None:
        return False
    try:
        cc = CellConstraint(
            fixed_length=cell_constraint_data.get("fixed_length"),
            target_counts=cell_constraint_data.get("target_counts", {}),
            parity_rules=cell_constraint_data.get("parity_rules", {}),
            multiple_rules=cell_constraint_data.get("multiple_rules", {}),
            unique_exact=cell_constraint_data.get("unique_exact"),
            ordering=cell_constraint_data.get("ordering"),
            positional_type=cell_constraint_data.get("positional_type"),
            positional_index_rule=cell_constraint_data.get("positional_index_rule"),
        )
        return cell_satisfies(answer_str, cc)
    except Exception:
        return False


def load_environment(
    mode: str = "image",
    answer_mode: str = "mc",
    image_dataset_path: Optional[str] = None,
    text_dataset_path: Optional[str] = None,
    image_base_path: Optional[str] = None,
    patterns: bool = False,
    max_examples: int = -1,
    **kwargs,
):
    """
    Load a verifiers MultiTurnEnv for RAPM.

    Args:
        mode: "image" or "text"
        answer_mode: "mc" (multiple choice) or "gen" (generative, text only)
        image_dataset_path: Path to raven_subset.json
        text_dataset_path: Path to text_rapm JSONL
        image_base_path: Base path for resolving image paths
        patterns: Include pattern hints in system prompt
        max_examples: Limit dataset size (-1 for all)
        **kwargs: Additional args passed to MultiTurnEnv

    Returns:
        vf.Environment instance
    """
    if vf is None:
        raise ImportError("verifiers is not installed; use in a verifiers workspace")

    # Load dataset
    if mode == "image":
        if image_dataset_path is None:
            raise ValueError("image_dataset_path required for image mode")
        dataset = _load_rapm_image_dataset(
            image_dataset_path,
            image_base_path=image_base_path,
            limit=(None if max_examples < 0 else max_examples),
        )
        system_prompt = IMAGE_BASE_PROMPT
        if patterns:
            system_prompt += IMAGE_PATTERN_INFO
    else:
        if text_dataset_path is None:
            raise ValueError("text_dataset_path required for text mode")
        dataset = _load_rapm_text_dataset(
            text_dataset_path,
            answer_mode=answer_mode,
            limit=(None if max_examples < 0 else max_examples),
        )
        system_prompt = TEXT_BASE_PROMPT
        if patterns:
            system_prompt += TEXT_PATTERN_INFO
        if answer_mode == "mc":
            system_prompt += "You will be given 8 answer options (1-8). Select the single option that correctly fills the missing cell.\n"
        else:
            system_prompt += "You must GENERATE the exact missing cell string that satisfies ALL inferred constraints.\n"
        system_prompt += "State your final answer as: <answer>...</answer>\n"

    parser = vf.Parser()

    # Rubric: compute reward based on trajectory
    async def compute_reward(parser, completion, answer, state, **_):
        """
        Compute reward matching CognitiveEval RAPM design:
        - Image-MC: +1.0 correct; -0.1 per wrong; -1.0 if max turns reached
        - Text-MC/Gen: +1.0 correct; -0.1 × violations per wrong; -1.0 if max turns reached
        - Invalid format: -0.5
        """
        info = state.get("info", {})
        mode = info.get("mode", "image")
        answer_mode = info.get("answer_mode", "mc")
        max_turns = info.get("max_turns", 8)

        trajectory = state.get("trajectory", [])
        attempts = len(trajectory)

        # Parse last completion
        last_completion = completion or state.get("completion") or ""

        # Determine correctness and violations
        if mode == "image" or answer_mode == "mc":
            # Numeric answer
            pred = _parse_numeric_answer(str(last_completion))
            if pred is None:
                return REWARD_INVALID_FORMAT
            try:
                gold = int(state.get("answer"))
            except Exception:
                gold = None
            is_correct = pred is not None and gold is not None and pred == gold
            violations = 1  # Not used for image-mc
        else:
            # Text-gen: string answer
            pred = _parse_string_answer(str(last_completion))
            if pred is None:
                return REWARD_INVALID_FORMAT
            cell_constraint = info.get("cell_constraint")
            is_correct = _check_text_gen_correct(pred, cell_constraint)
            violations = (
                _count_constraint_violations(pred, cell_constraint)
                if not is_correct
                else 0
            )

        if is_correct:
            return REWARD_CORRECT

        # Wrong answer
        if mode == "text":
            # Text modes: constraint-based penalty
            if attempts >= max_turns:
                return REWARD_MAX_TURNS_FAILED
            return REWARD_CONSTRAINT_VIOLATION * violations
        else:
            # Image-MC: fixed penalty per wrong turn
            if attempts >= max_turns:
                return REWARD_MAX_TURNS_FAILED
            return REWARD_WRONG_MULTITURN

    rubric = vf.Rubric(funcs=[compute_reward], parser=parser)

    class RAPMVerifiersEnv(vf.MultiTurnEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @vf.stop
        async def correct_answered(self, state: vf.State) -> bool:
            """Stop when correct answer is given."""
            info = state.get("info", {})
            mode = info.get("mode", "image")
            answer_mode = info.get("answer_mode", "mc")
            last = state.get("completion") or ""

            if mode == "image" or answer_mode == "mc":
                pred = _parse_numeric_answer(str(last))
                if pred is None:
                    return False
                try:
                    gold = int(state.get("answer"))
                except Exception:
                    return False
                return pred == gold
            else:
                pred = _parse_string_answer(str(last))
                if pred is None:
                    return False
                cell_constraint = info.get("cell_constraint")
                return _check_text_gen_correct(pred, cell_constraint)

        @vf.stop
        async def max_turns_reached(self, state: vf.State) -> bool:
            """Stop when max turns reached."""
            max_turns = state.get("info", {}).get("max_turns", 8)
            return len(state.get("trajectory", [])) >= max_turns

        async def env_response(
            self, messages: vf.Messages, state: vf.State, **kwargs
        ) -> vf.Messages:
            """Provide feedback after each turn."""
            info = state.get("info", {})
            mode = info.get("mode", "image")
            answer_mode = info.get("answer_mode", "mc")
            max_turns = info.get("max_turns", 8)
            attempts = len(state.get("trajectory", []))
            last = state.get("completion") or ""

            # Parse answer
            if mode == "image" or answer_mode == "mc":
                pred = _parse_numeric_answer(str(last))
            else:
                pred = _parse_string_answer(str(last))

            # Check correctness
            if pred is None:
                feedback = "Please answer with the correct format: <answer>your answer</answer>"
            elif mode == "image" or answer_mode == "mc":
                try:
                    gold = int(state.get("answer"))
                except Exception:
                    gold = None
                is_correct = pred == gold
                if is_correct:
                    feedback = ""
                elif attempts >= max_turns:
                    feedback = ""
                else:
                    feedback = "Incorrect. Try again."
            else:
                # Text-gen
                cell_constraint = info.get("cell_constraint")
                is_correct = _check_text_gen_correct(pred, cell_constraint)
                if is_correct:
                    feedback = ""
                elif attempts >= max_turns:
                    feedback = ""
                else:
                    violations = _count_constraint_violations(pred, cell_constraint)
                    feedback = (
                        f"Incorrect. {violations} constraint(s) violated. Try again."
                    )

            if not feedback:
                return []
            return [{"role": "user", "content": feedback}]

        def get_rubric(self):
            """Return the rubric instance for this environment."""
            return self._rapm_rubric

        def compute_reward_with_state(
            self, completions: List[List[dict]], state: vf.State
        ) -> List[float]:
            """Compute rewards using rubric with current state context.

            Passes mode, answer_mode, max_turns, cell_constraint, gold_answer,
            and attempts to the rubric for proper reward computation.
            """
            rubric = self.get_rubric()

            info = state.get("info", {})
            mode = info.get("mode", "image")
            answer_mode = info.get("answer_mode", "mc")
            max_turns = info.get("max_turns", 8)
            cell_constraint = info.get("cell_constraint")
            gold_answer = state.get("answer")
            attempts = len(state.get("trajectory", []))

            # Prepare kwargs for rubric functions
            reward_kwargs = {
                "mode": mode,
                "answer_mode": answer_mode,
                "max_turns": max_turns,
                "cell_constraint": cell_constraint,
                "attempts": attempts,
            }

            # Compute turn-level rewards
            turn_rewards = []
            for func in rubric.turn_reward_funcs:
                turn_rewards.append(
                    func(completions, [gold_answer] * len(completions), **reward_kwargs)
                )

            total_turn = [
                sum(r[i] for r in turn_rewards) for i in range(len(completions))
            ]

            # Compute outcome-level rewards
            outcome_rewards = [0.0] * len(completions)
            for func in rubric.outcome_reward_funcs:
                o = func(completions, [gold_answer] * len(completions), **reward_kwargs)
                for i in range(len(completions)):
                    outcome_rewards[i] += o[i]

            # Return total rewards
            return [total_turn[i] + outcome_rewards[i] for i in range(len(completions))]

    # Create rubric instance with correct mode/answer_mode from parameters
    rapm_rubric = RAPMRubric(mode=mode, answer_mode=answer_mode)

    env = RAPMVerifiersEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
    env._rapm_rubric = rapm_rubric
    return env
