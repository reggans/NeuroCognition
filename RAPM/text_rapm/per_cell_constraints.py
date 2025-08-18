from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class CellConstraint:
    # Fixed or target metrics
    fixed_length: Optional[int] = None
    target_counts: Dict[str, int] = field(
        default_factory=dict
    )  # exact counts for types
    parity_rules: Dict[str, str] = field(default_factory=dict)  # even/odd for a type
    multiple_rules: Dict[str, int] = field(
        default_factory=dict
    )  # multiple_of for a type
    unique_exact: Optional[int] = None

    ordering: Optional[str] = None  # ascending/descending/mixed

    positional_type: Optional[str] = None  # char type for positional rule
    positional_index_rule: Optional[str] = None  # first/last/even/odd

    # Feasible character pool intersection (list of allowed chars overall)
    allowed_chars: Optional[List[str]] = None

    def describe(self) -> Dict:
        return {
            "fixed_length": self.fixed_length,
            "target_counts": self.target_counts,
            "parity_rules": self.parity_rules,
            "multiple_rules": self.multiple_rules,
            "unique_exact": self.unique_exact,
            "ordering": self.ordering,
            "positional_type": self.positional_type,
            "positional_index_rule": self.positional_index_rule,
            "allowed_chars_size": (
                len(self.allowed_chars) if self.allowed_chars else None
            ),
        }
