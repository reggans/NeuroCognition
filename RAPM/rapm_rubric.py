"""
RAPM Rubric for multi-turn RL training.
Defines turn-level and outcome-level reward functions compatible with TRL.

Supports:
- Image-MC: max 8 turns, -0.1 per wrong, -1.0 final penalty if max turns reached
- Text-MC: max turns = # constraints, -0.1 × violations per wrong turn, -1.0 final penalty
- Text-Gen: same as text-MC but generates strings instead of selecting from options
"""

import re
from typing import List

try:
    from verifiers.rubrics import Rubric
    from verifiers.parsers import XMLParser
except ImportError:
    raise ImportError(
        "verifiers not installed. Install Multi-Turn-RL-Agent:\n"
        "  cd /root/Multi-Turn-RL-Agent && pip install -e ."
    )

# Import existing validators from the RAPM module
try:
    from RAPM.text_rapm.per_cell_constraints import CellConstraint
    from RAPM.text_rapm.validator import cell_satisfies, constraint_violations
except ImportError:
    CellConstraint = None
    cell_satisfies = None
    constraint_violations = None

# Reward constants (matching rapm_env.py)
REWARD_CORRECT = 1.0
REWARD_WRONG_MULTITURN = -0.1
REWARD_CONSTRAINT_VIOLATION = -0.1
REWARD_MAX_TURNS_FAILED = -1.0
REWARD_INVALID_FORMAT = -0.5


class RAPMRubric(Rubric):
    """
    Rubric for Raven's Progressive Matrices (RAPM).
    
    Reward structure matching rapm_env.py:
    - Turn-level: -0.1 per wrong (image) or -0.1 × violations (text); -0.5 invalid format
    - Outcome-level: +1.0 correct; -1.0 additional if max_turns reached without success
    """
    
    def __init__(self, mode: str = "image", answer_mode: str = "mc"):
        """Initialize with mode configuration."""
        super().__init__()
        self.parser = XMLParser(fields=["reasoning", "answer"])
        self.mode = mode
        self.answer_mode = answer_mode
    
    def _parse_numeric_answer(self, text: str) -> int or None:
        """Parse numeric answer (1-8) from <answer> tags."""
        if not text:
            return None
        
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
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
    
    def _parse_string_answer(self, text: str) -> str or None:
        """Parse string answer from <answer> tags (for text-gen)."""
        if not text:
            return None
        
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            return None
        
        return match.group(1).strip().strip('"')
    
    def _count_constraint_violations(
        self, answer_str: str, cell_constraint_data: dict or None
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
        self, answer_str: str, cell_constraint_data: dict or None
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
    
    # ========== TURN-LEVEL REWARDS ==========
    
    def turn_wrong_answer_penalty(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Turn-level penalty for wrong answers.
        
        - Image-MC: -0.1 per wrong answer
        - Text mode: -0.1 × constraint_violations per wrong answer
        
        Args:
            completions: List[List[Dict]]  # Trajectories
            answer: List[str]  # Expected answers
            **kwargs: mode, answer_mode, cell_constraint
        
        Returns:
            List[float]  # Penalty per example
        """
        rewards = []
        mode = kwargs.get('mode', self.mode)
        answer_mode = kwargs.get('answer_mode', self.answer_mode)
        
        for trajectory, expected in zip(completions, answer):
            # Get last assistant message
            last_assistant = None
            for msg in reversed(trajectory):
                if msg['role'] == 'assistant':
                    last_assistant = msg['content']
                    break
            
            if not last_assistant:
                rewards.append(0.0)
                continue
            
            # Parse answer based on mode
            if mode == "image" or answer_mode == "mc":
                pred = self._parse_numeric_answer(last_assistant)
                if pred is None:
                    # Invalid format handled separately
                    rewards.append(0.0)
                    continue
                
                try:
                    gold = int(expected)
                    is_correct = (pred == gold)
                except (ValueError, TypeError):
                    is_correct = False
                
                if is_correct:
                    rewards.append(0.0)  # No penalty for correct
                else:
                    rewards.append(REWARD_WRONG_MULTITURN)  # -0.1
            else:
                # Text-gen mode
                pred = self._parse_string_answer(last_assistant)
                if pred is None:
                    rewards.append(0.0)
                    continue
                
                cell_constraint = kwargs.get('cell_constraint')
                is_correct = self._check_text_gen_correct(pred, cell_constraint)
                
                if is_correct:
                    rewards.append(0.0)
                else:
                    violations = self._count_constraint_violations(pred, cell_constraint)
                    rewards.append(REWARD_CONSTRAINT_VIOLATION * violations)  # -0.1 × violations
        
        return rewards
    
    def turn_invalid_format_penalty(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Turn-level penalty for invalid answer format.
        
        -0.5 for responses without proper <answer> tags or unparseable answers.
        
        Args:
            completions: List[List[Dict]]  # Trajectories
            answer: List[str]  # Not used
        
        Returns:
            List[float]  # Penalty per example
        """
        rewards = []
        mode = kwargs.get('mode', self.mode)
        answer_mode = kwargs.get('answer_mode', self.answer_mode)
        
        for trajectory in completions:
            # Get last assistant message
            last_assistant = None
            for msg in reversed(trajectory):
                if msg['role'] == 'assistant':
                    last_assistant = msg['content']
                    break
            
            if not last_assistant:
                rewards.append(REWARD_INVALID_FORMAT)
                continue
            
            # Check if answer is parseable
            if mode == "image" or answer_mode == "mc":
                pred = self._parse_numeric_answer(last_assistant)
            else:
                pred = self._parse_string_answer(last_assistant)
            
            if pred is None:
                rewards.append(REWARD_INVALID_FORMAT)  # -0.5
            else:
                rewards.append(0.0)  # Valid format
        
        return rewards
    
    # ========== OUTCOME REWARDS ==========
    
    def outcome_correctness_reward(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Outcome reward: +1.0 if final answer is correct, 0 otherwise.
        
        Called at episode end.
        
        Args:
            completions: List[List[Dict]]  # Complete trajectories
            answer: List[str]  # Expected answers
        
        Returns:
            List[float]  # Correctness reward per example
        """
        rewards = []
        mode = kwargs.get('mode', self.mode)
        answer_mode = kwargs.get('answer_mode', self.answer_mode)
        
        for trajectory, expected in zip(completions, answer):
            # Get last assistant message
            last_assistant = None
            for msg in reversed(trajectory):
                if msg['role'] == 'assistant':
                    last_assistant = msg['content']
                    break
            
            if not last_assistant:
                rewards.append(0.0)
                continue
            
            # Determine correctness based on mode
            if mode == "image" or answer_mode == "mc":
                pred = self._parse_numeric_answer(last_assistant)
                if pred is None:
                    rewards.append(0.0)
                else:
                    try:
                        gold = int(expected)
                        if pred == gold:
                            rewards.append(REWARD_CORRECT)  # +1.0
                        else:
                            rewards.append(0.0)
                    except (ValueError, TypeError):
                        rewards.append(0.0)
            else:
                # Text-gen: string answer
                pred = self._parse_string_answer(last_assistant)
                if pred is None:
                    rewards.append(0.0)
                else:
                    cell_constraint = kwargs.get('cell_constraint')
                    is_correct = self._check_text_gen_correct(pred, cell_constraint)
                    if is_correct:
                        rewards.append(REWARD_CORRECT)  # +1.0
                    else:
                        rewards.append(0.0)
        
        return rewards
    
    def outcome_max_turns_failure(
        self, completions: List[List[dict]], answer: List[str], **kwargs
    ) -> List[float]:
        """
        Outcome penalty: -1.0 if max turns reached without correct answer.
        
        This is an additional penalty applied at episode end when the model
        failed to find the correct answer within the turn limit.
        
        Args:
            completions: List[List[Dict]]  # Complete trajectories
            answer: List[str]  # Expected answers
            **kwargs: max_turns, attempts
        
        Returns:
            List[float]  # Penalty per example
        """
        rewards = []
        mode = kwargs.get('mode', self.mode)
        answer_mode = kwargs.get('answer_mode', self.answer_mode)
        max_turns = kwargs.get('max_turns', 8)
        attempts = kwargs.get('attempts', 0)
        
        for trajectory, expected in zip(completions, answer):
            # Count turns (assistant messages)
            turns_taken = sum(1 for msg in trajectory if msg['role'] == 'assistant')
            
            # Get last assistant message
            last_assistant = None
            for msg in reversed(trajectory):
                if msg['role'] == 'assistant':
                    last_assistant = msg['content']
                    break
            
            if not last_assistant:
                rewards.append(0.0)
                continue
            
            # Check if correct
            is_correct = False
            if mode == "image" or answer_mode == "mc":
                pred = self._parse_numeric_answer(last_assistant)
                if pred is not None:
                    try:
                        gold = int(expected)
                        is_correct = (pred == gold)
                    except (ValueError, TypeError):
                        pass
            else:
                pred = self._parse_string_answer(last_assistant)
                if pred is not None:
                    cell_constraint = kwargs.get('cell_constraint')
                    is_correct = self._check_text_gen_correct(pred, cell_constraint)
            
            # Apply penalty if max turns reached without success
            if turns_taken >= max_turns and not is_correct:
                rewards.append(REWARD_MAX_TURNS_FAILED)  # -1.0
            else:
                rewards.append(0.0)
        
        return rewards
    
    @property
    def turn_reward_funcs(self) -> List:
        """
        Turn-level reward functions.
        Called after each step to provide immediate feedback.
        """
        return [
            self.turn_wrong_answer_penalty,
            self.turn_invalid_format_penalty,
        ]
    
    @property
    def outcome_reward_funcs(self) -> List:
        """
        Outcome-level reward functions.
        Called at episode end to evaluate overall performance.
        """
        return [
            self.outcome_correctness_reward,
            self.outcome_max_turns_failure,
        ]
