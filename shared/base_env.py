"""
Base environment class for cognitive evaluation tasks.
Designed for RL post-training with methods like GRPO.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class ActionStatus(Enum):
    """Status of an action taken in the environment."""
    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    INVALID_ACTION = "invalid_action"
    REPEATED = "repeated"
    ILLEGAL = "illegal"
    NOBOX = "nobox"  # For image mode: coordinate doesn't match any box


@dataclass
class StepResult:
    """Result from taking a step in the environment."""
    observation: str  # The next observation/prompt to show the model
    reward: float  # Immediate reward
    done: bool  # Whether the episode is finished
    info: Dict[str, Any] = field(default_factory=dict)  # Additional info
    truncated: bool = False  # Whether episode was truncated (e.g., max steps)


@dataclass
class EnvState:
    """Serializable environment state for checkpointing."""
    step_count: int
    history: List[Dict[str, Any]]
    internal_state: Dict[str, Any]


class CognitiveEnv(ABC):
    """
    Abstract base class for cognitive evaluation environments.
    
    Designed for RL training where:
    - reset() initializes a new episode and returns the initial observation
    - step(action) takes an action string and returns (observation, reward, done, info)
    - The environment maintains internal state that can be serialized
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the environment.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng = None
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []
        self._done = False
        
    @abstractmethod
    def reset(self) -> str:
        """
        Reset the environment and return the initial observation.
        
        Returns:
            str: Initial observation/prompt for the model
        """
        pass
    
    @abstractmethod
    def step(self, action: str) -> StepResult:
        """
        Take an action in the environment.
        
        Args:
            action: The raw response from the model
            
        Returns:
            StepResult: Contains observation, reward, done flag, and info dict
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this task.
        
        Returns:
            str: System prompt describing the task
        """
        pass
    
    @abstractmethod
    def parse_action(self, response: str) -> Tuple[Optional[Any], ActionStatus]:
        """
        Parse the model's response to extract the action.
        
        Args:
            response: Raw model response
            
        Returns:
            Tuple of (parsed_action, status)
        """
        pass
    
    def get_state(self) -> EnvState:
        """
        Get the current environment state for serialization.
        
        Returns:
            EnvState: Current state
        """
        return EnvState(
            step_count=self.step_count,
            history=self.history.copy(),
            internal_state=self._get_internal_state()
        )
    
    def set_state(self, state: EnvState) -> None:
        """
        Restore environment state from a checkpoint.
        
        Args:
            state: State to restore
        """
        self.step_count = state.step_count
        self.history = state.history.copy()
        self._set_internal_state(state.internal_state)
    
    @abstractmethod
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get task-specific internal state."""
        pass
    
    @abstractmethod
    def _set_internal_state(self, state: Dict[str, Any]) -> None:
        """Set task-specific internal state."""
        pass
    
    def is_done(self) -> bool:
        """Check if episode is finished."""
        return self._done
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the history of all steps in this episode."""
        return self.history.copy()
    
    @abstractmethod
    def compute_episode_reward(self) -> float:
        """
        Compute the total reward for the episode.
        Can be used for sparse reward settings.
        
        Returns:
            float: Total episode reward
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get evaluation metrics for the episode.
        
        Returns:
            Dict containing task-specific metrics
        """
        pass
