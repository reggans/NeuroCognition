from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

AttributeType = Literal[
    "character_set_type",
    "type_count",
    "quant_constant",
    "quant_progression",
    "sorted",
    "positional",
]


@dataclass
class CharacterSetTypeSpec:
    attribute: Literal["character_set_type"] = "character_set_type"
    character_set_type: Literal["letters", "digits", "symbols"] = "letters"


@dataclass
class TypeCountSpec:
    attribute: Literal["type_count"] = "type_count"
    character_type: str = "vowels"  # any allowed type
    rule: Literal["even", "odd", "multiple"] = "even"
    multiple_of: Optional[int] = None  # 2,3,4 when rule == multiple


@dataclass
class QuantConstantSpec:
    attribute: Literal["quant_constant"] = "quant_constant"
    metric: str = "length"  # or any character type or unique
    value: int = 1  # 1-10


@dataclass
class QuantProgressionSpec:
    attribute: Literal["quant_progression"] = "quant_progression"
    metric: str = "length"  # one metric only
    start: int = 1  # 1-3
    step: int = 1  # 1-5


@dataclass
class SortedSpec:
    attribute: Literal["sorted"] = "sorted"
    order: Literal["ascending", "descending", "mixed"] = "ascending"


@dataclass
class PositionalSpec:
    attribute: Literal["positional"] = "positional"
    character_type: str = "vowels"  # not unique
    index: Literal["first", "last", "even", "odd"] = "first"


AxisSpec = (
    CharacterSetTypeSpec
    | TypeCountSpec
    | QuantConstantSpec
    | QuantProgressionSpec
    | SortedSpec
    | PositionalSpec
)


def spec_to_dict(spec: AxisSpec) -> Dict[str, Any]:
    return {k: getattr(spec, k) for k in spec.__dataclass_fields__}
