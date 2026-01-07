"""
Raven's Progressive Matrices (RAPM) Environment for RL training.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.base_env import CognitiveEnv, StepResult, ActionStatus


@dataclass
class RAPMQuestion:
    """Represents a single RAPM question."""
    id: str
    question_type: str  # "image" or "text"
    correct_answer: int  # 0-indexed for image, varies for text
    dataset_type: Optional[str] = None
    image_path: Optional[str] = None  # For image-based
    question_grid: Optional[List[List[str]]] = None  # For text-based
    options: Optional[List[str]] = None  # For text-based MC
    categories: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)


class RAPMEnv(CognitiveEnv):
    """
    Raven's Progressive Matrices environment for RL training.
    
    Unlike WCST/SWM, RAPM is typically evaluated as single-shot questions.
    This environment supports:
    - Single question mode: One question per episode
    - Dataset mode: Iterate through a dataset of questions
    
    Reward structure:
    - +1.0 for correct answer
    - 0.0 for incorrect answer
    - -0.5 for invalid format
    
    Episode ends after one valid attempt per question.
    """
    
    # System prompts
    IMAGE_BASE_PROMPT = (
        "You are taking the Raven's Progressive Matrices (RPM) test, a non-verbal intelligence test that measures abstract reasoning ability.\n\n"
        "You will see a 3x3 matrix of images with the bottom-right image missing (shown as a question mark), followed by 8 answer choices numbered 1-8.\n\n"
        "Your task is to: \n1. Analyze rows and columns\n2. Infer the governing logical rule(s)\n3. Select the answer choice (1-8) that correctly completes the matrix.\n\n"
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
    
    def __init__(
        self,
        mode: str = "image",  # "image" or "text"
        answer_mode: str = "mc",  # "mc" (multiple choice) or "gen" (generative)
        cot: bool = False,
        think_budget: int = 256,
        patterns: bool = False,  # Include pattern hints
        max_retries: int = 3,  # Max retries for invalid format
        image_base_path: Optional[str] = None,  # Base path for resolving image paths
        image_only: bool = False,  # If True, observation only indicates image path
        seed: Optional[int] = None,
    ):
        """
        Initialize the RAPM environment.
        
        Args:
            mode: Question type ("image" or "text")
            answer_mode: Answer format ("mc" or "gen")
            cot: Whether to request chain-of-thought reasoning
            think_budget: Token budget for reasoning
            patterns: Whether to include pattern hints in prompt
            max_retries: Maximum retries for invalid format responses
            image_base_path: Base directory for resolving relative image paths
            image_only: If True, observation only indicates image path (for multimodal models)
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        
        self.mode = mode
        self.answer_mode = answer_mode
        self.cot = cot
        self.think_budget = think_budget
        self.patterns = patterns
        self.max_retries = max_retries
        self.image_base_path = image_base_path
        self.image_only = image_only
        
        # Current question state
        self._current_question: Optional[RAPMQuestion] = None
        self._attempts = 0
        self._answered = False
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for RAPM."""
        if self.mode == "image":
            prompt = self.IMAGE_BASE_PROMPT
            if self.patterns:
                prompt += self.IMAGE_PATTERN_INFO
            prompt += "Your final answer should be a number between 1-8 corresponding to the correct choice.\n"
        else:
            prompt = self.TEXT_BASE_PROMPT
            if self.patterns:
                prompt += self.TEXT_PATTERN_INFO
            if self.answer_mode == "mc":
                prompt += (
                    "You will be given 8 answer options (1-8). Select the single option that correctly fills the missing cell.\n"
                )
            else:
                prompt += (
                    "You must GENERATE the exact missing cell string that satisfies ALL inferred constraints.\n"
                )
        
        if self.cot:
            prompt += f"\nExplain your thought process (max {self.think_budget} tokens) inside <think>...</think> then give final answer.\n"
        else:
            prompt += "\nAnswer only with your final answer.\n"
        
        if self.mode == "image" or self.answer_mode == "mc":
            prompt += "State your final answer as: <answer>number</answer>\n"
        else:
            prompt += "State your final answer as: <answer>string</answer>\n"
        
        return prompt
    
    def set_question(self, question: RAPMQuestion) -> str:
        """
        Set a specific question and return the observation.
        
        Args:
            question: The RAPM question to present
            
        Returns:
            str: The observation/prompt for this question
        """
        self._current_question = question
        self._attempts = 0
        self._answered = False
        self._done = False
        self.step_count = 0
        self.history = []
        
        return self._format_observation()
    
    def set_question_from_dict(self, data: Dict[str, Any], question_type: str = "image") -> str:
        """
        Set a question from a dictionary (loaded from JSON).
        
        Args:
            data: Dictionary with question data
            question_type: "image" or "text"
            
        Returns:
            str: The observation/prompt for this question
        """
        if question_type == "image":
            question = RAPMQuestion(
                id=data.get("id", "unknown"),
                question_type="image",
                correct_answer=data.get("correct_answer", -1),
                dataset_type=data.get("dataset_type"),
                image_path=data.get("full_image"),
                raw_data=data,
            )
        else:
            question = RAPMQuestion(
                id=data.get("id", "unknown"),
                question_type="text",
                correct_answer=data.get("correct_index", -1),
                question_grid=data.get("question_grid") or data.get("full_grid"),
                options=data.get("options", []),
                categories=data.get("credited_categories") or data.get("assigned_categories") or [],
                raw_data=data,
            )
        
        return self.set_question(question)
    
    def reset(self) -> str:
        """
        Reset the environment. Returns empty string if no question is set.
        Use set_question() or set_question_from_dict() to load a question.
        """
        self.step_count = 0
        self.history = []
        self._done = False
        self._attempts = 0
        self._answered = False
        
        if self._current_question is not None:
            return self._format_observation()
        
        return ""
    
    def _format_observation(self) -> str:
        """Format current question as observation."""
        q = self._current_question
        if q is None:
            return ""
        
        if q.question_type == "image":
            # For image mode
            image_path = self.get_current_image_path()
            if self.image_only:
                return f"[Image: {image_path}]\nSelect the correct answer (1-8)."
            else:
                return "Analyze the image and select the correct answer (1-8)."
        
        else:
            # Text mode: format the grid
            grid = q.question_grid
            if grid is None:
                return ""
            
            rows = []
            for r in range(3):
                row_cells = []
                for c in range(3):
                    v = grid[r][c] if r < len(grid) and c < len(grid[r]) else None
                    row_cells.append("?" if v is None else str(v))
                rows.append(" | ".join(row_cells))
            grid_text = "\n".join(rows)
            
            if self.answer_mode == "mc" and q.options:
                opt_lines = [f"{i+1}. {o}" for i, o in enumerate(q.options)]
                return f"Matrix:\n{grid_text}\n\nOptions:\n" + "\n".join(opt_lines) + "\n\nAnswer with <answer>N</answer>."
            else:
                return f"Matrix:\n{grid_text}\n\nGenerate the missing cell. Answer with <answer>STRING</answer>."
    
    def get_image_path(self) -> Optional[str]:
        """Get the raw image path for the current question (image mode only).
        This returns the path as stored in the question data."""
        if self._current_question and self._current_question.question_type == "image":
            return self._current_question.image_path
        return None
    
    def get_current_image_path(self) -> Optional[str]:
        """Get the full resolved image path for the current question (image mode only).
        If image_base_path is set, this returns the full path."""
        if self._current_question and self._current_question.question_type == "image":
            img_path = self._current_question.image_path
            if img_path and self.image_base_path:
                return os.path.join(self.image_base_path, img_path)
            return img_path
        return None
    
    def parse_action(self, response: str) -> Tuple[Optional[Any], ActionStatus]:
        """Parse the model's response to extract the answer."""
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        
        if match is None:
            return None, ActionStatus.INVALID_FORMAT
        
        answer_text = match.group(1).strip()
        
        if self.mode == "image" or self.answer_mode == "mc":
            # Parse numeric answer (1-8)
            nums = re.findall(r"\d+", answer_text)
            if not nums:
                return None, ActionStatus.INVALID_FORMAT
            try:
                n = int(nums[0])
                if 1 <= n <= 8:
                    return n, ActionStatus.VALID
                else:
                    return None, ActionStatus.INVALID_ACTION
            except ValueError:
                return None, ActionStatus.INVALID_FORMAT
        else:
            # Generative mode: return the string as-is
            if answer_text:
                return answer_text.strip().strip('"'), ActionStatus.VALID
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
        
        if self._current_question is None:
            return StepResult(
                observation="",
                reward=0.0,
                done=True,
                info={"error": "No question set"},
                truncated=False
            )
        
        self._attempts += 1
        self.step_count += 1
        
        # Parse action
        parsed_answer, status = self.parse_action(action)
        
        q = self._current_question
        step_info = {
            "question_id": q.id,
            "attempt": self._attempts,
            "predicted_answer": parsed_answer,
            "correct_answer": q.correct_answer,
            "raw_response": action,
            "status": status.value,
        }
        
        # Handle invalid format
        if status == ActionStatus.INVALID_FORMAT:
            step_info["is_correct"] = False
            self.history.append(step_info)
            
            if self._attempts >= self.max_retries:
                self._done = True
                return StepResult(
                    observation="",
                    reward=-0.5,
                    done=True,
                    info=step_info,
                    truncated=True
                )
            
            return StepResult(
                observation="Please answer with the correct format: <answer>your answer</answer>",
                reward=-0.5,
                done=False,
                info=step_info
            )
        
        # Handle invalid action
        if status == ActionStatus.INVALID_ACTION:
            step_info["is_correct"] = False
            self.history.append(step_info)
            
            if self._attempts >= self.max_retries:
                self._done = True
                return StepResult(
                    observation="",
                    reward=-0.5,
                    done=True,
                    info=step_info,
                    truncated=True
                )
            
            return StepResult(
                observation="Please answer with a number between 1 and 8.",
                reward=-0.5,
                done=False,
                info=step_info
            )
        
        # Check correctness
        is_correct = self._check_answer(parsed_answer)
        step_info["is_correct"] = is_correct
        self.history.append(step_info)
        
        reward = 1.0 if is_correct else 0.0
        self._done = True
        self._answered = True
        
        return StepResult(
            observation="",
            reward=reward,
            done=True,
            info=step_info
        )
    
    def _check_answer(self, predicted: Any) -> bool:
        """Check if the predicted answer is correct."""
        q = self._current_question
        if q is None:
            return False
        
        if self.mode == "image" or self.answer_mode == "mc":
            # For MC, check if predicted (1-indexed) matches correct (0-indexed)
            return predicted is not None and (predicted - 1) == q.correct_answer
        else:
            # For generative mode, need constraint checking
            # This would require the constraint validator from rapm_utils
            # For now, exact match with gold answer
            gold = q.raw_data.get("answer")
            return predicted == gold
    
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get internal state for serialization."""
        q = self._current_question
        return {
            "current_question": {
                "id": q.id if q else None,
                "question_type": q.question_type if q else None,
                "correct_answer": q.correct_answer if q else None,
                "dataset_type": q.dataset_type if q else None,
                "image_path": q.image_path if q else None,
                "question_grid": q.question_grid if q else None,
                "options": q.options if q else None,
                "categories": q.categories if q else [],
            } if q else None,
            "attempts": self._attempts,
            "answered": self._answered,
        }
    
    def _set_internal_state(self, state: Dict[str, Any]) -> None:
        """Restore internal state."""
        q_data = state.get("current_question")
        if q_data and q_data.get("id"):
            self._current_question = RAPMQuestion(
                id=q_data["id"],
                question_type=q_data.get("question_type", "image"),
                correct_answer=q_data.get("correct_answer", -1),
                dataset_type=q_data.get("dataset_type"),
                image_path=q_data.get("image_path"),
                question_grid=q_data.get("question_grid"),
                options=q_data.get("options"),
                categories=q_data.get("categories", []),
            )
        else:
            self._current_question = None
        
        self._attempts = state.get("attempts", 0)
        self._answered = state.get("answered", False)
    
    def compute_episode_reward(self) -> float:
        """Compute total episode reward."""
        total = 0.0
        for step in self.history:
            if step.get("status") in [ActionStatus.INVALID_FORMAT.value, ActionStatus.INVALID_ACTION.value]:
                total -= 0.5
            elif step.get("is_correct"):
                total += 1.0
        return total
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics."""
        if not self.history:
            return {"answered": False}
        
        final_step = self.history[-1]
        return {
            "question_id": self._current_question.id if self._current_question else None,
            "answered": self._answered,
            "is_correct": final_step.get("is_correct", False),
            "attempts": self._attempts,
            "dataset_type": self._current_question.dataset_type if self._current_question else None,
            "categories": self._current_question.categories if self._current_question else [],
        }


class RAPMDatasetEnv:
    """
    Wrapper for iterating through a dataset of RAPM questions.
    
    This provides a convenient interface for evaluating on a full dataset
    while using the single-question RAPMEnv internally.
    """
    
    def __init__(
        self,
        data_path: str,
        mode: str = "image",
        answer_mode: str = "mc",
        cot: bool = False,
        think_budget: int = 256,
        patterns: bool = False,
        limit_per_type: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the dataset environment.
        
        Args:
            data_path: Path to the evaluation data JSON/JSONL
            mode: Question type ("image" or "text")
            answer_mode: Answer format ("mc" or "gen")
            cot: Whether to request chain-of-thought reasoning
            think_budget: Token budget for reasoning
            patterns: Whether to include pattern hints
            limit_per_type: Limit questions per dataset type
            seed: Random seed
        """
        self.data_path = data_path
        self.mode = mode
        self.limit_per_type = limit_per_type
        
        # Load data
        self.questions = self._load_data()
        self._current_idx = 0
        
        # Create internal env
        self.env = RAPMEnv(
            mode=mode,
            answer_mode=answer_mode,
            cot=cot,
            think_budget=think_budget,
            patterns=patterns,
            seed=seed,
        )
        
        # Results tracking
        self.results: List[Dict[str, Any]] = []
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load evaluation data from file."""
        if self.mode == "image":
            with open(self.data_path, "r") as f:
                data = json.load(f)
            questions = data.get("questions", [])
        else:
            # JSONL format for text
            questions = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        questions.append(obj)
                    except json.JSONDecodeError:
                        continue
        
        # Apply limit if specified
        if self.limit_per_type is not None:
            from collections import defaultdict
            by_type = defaultdict(list)
            for q in questions:
                dt = q.get("dataset_type", "unknown")
                by_type[dt].append(q)
            
            limited = []
            for dt, lst in by_type.items():
                limited.extend(lst[:self.limit_per_type])
            questions = limited
        
        return questions
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __iter__(self):
        self._current_idx = 0
        return self
    
    def __next__(self) -> Tuple[str, RAPMEnv]:
        if self._current_idx >= len(self.questions):
            raise StopIteration
        
        q_data = self.questions[self._current_idx]
        self._current_idx += 1
        
        observation = self.env.set_question_from_dict(q_data, self.mode)
        return observation, self.env
    
    def get_question(self, idx: int) -> Tuple[str, RAPMEnv]:
        """Get a specific question by index."""
        q_data = self.questions[idx]
        observation = self.env.set_question_from_dict(q_data, self.mode)
        return observation, self.env
    
    def record_result(self, result: Dict[str, Any]) -> None:
        """Record a result for the current question."""
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all recorded results."""
        if not self.results:
            return {}
        
        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("is_correct"))
        
        from collections import defaultdict
        by_type = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in self.results:
            dt = r.get("dataset_type", "unknown")
            by_type[dt]["total"] += 1
            if r.get("is_correct"):
                by_type[dt]["correct"] += 1
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "by_type": {
                dt: {
                    **vals,
                    "accuracy": vals["correct"] / vals["total"] if vals["total"] > 0 else 0.0
                }
                for dt, vals in by_type.items()
            }
        }
