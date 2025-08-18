from __future__ import annotations
from typing import List, cast
from .constraint_specs import (
    CharacterSetTypeSpec,
    TypeCountSpec,
    QuantConstantSpec,
    QuantProgressionSpec,
    SortedSpec,
    PositionalSpec,
    AxisSpec,
)
from .character_sets import CHARACTER_POOLS, POSITIONAL_ALLOWED
from .util_rng import RNG

AXIS_ATTRIBUTE_TYPES = [
    "character_set_type",
    "type_count",
    "quant_constant",
    "quant_progression",
    "sorted",
    "positional",
]

MULTIPLE_CHOICES = [2, 3, 4]

SORT_ORDERS = ["ascending", "descending", "mixed"]

POSITIONAL_INDEXES = ["first", "last", "even", "odd"]

CHAR_SET_TYPE_ALLOWED = ["letters", "digits", "symbols"]

METRIC_TYPES = list(CHARACTER_POOLS.keys()) + ["length", "unique"]

COUNT_TYPES = list(CHARACTER_POOLS.keys()) + ["unique"]


def sample_axis_spec(rng: RNG) -> AxisSpec:
    atype = rng.choice(AXIS_ATTRIBUTE_TYPES)
    if atype == "character_set_type":
        cst = cast(str, rng.choice(CHAR_SET_TYPE_ALLOWED))
        return CharacterSetTypeSpec(character_set_type=cst)  # type: ignore[arg-type]
    if atype == "type_count":
        character_type = rng.choice(COUNT_TYPES)
        rule_type = rng.choice(["even", "odd", "multiple"])  # type: ignore[assignment]
        multiple_of = None
        if rule_type == "multiple":
            multiple_of = rng.choice(MULTIPLE_CHOICES)
        return TypeCountSpec(character_type=character_type, rule=rule_type, multiple_of=multiple_of)  # type: ignore[arg-type]
    if atype == "quant_constant":
        metric = rng.choice(METRIC_TYPES)
        value = rng.randint(2, 5)  # restricted to 2-5
        return QuantConstantSpec(metric=metric, value=value)
    if atype == "quant_progression":
        metric = rng.choice(METRIC_TYPES)
        start = rng.randint(1, 3)
        step = rng.randint(1, 3)  # restricted to 1-3
        return QuantProgressionSpec(metric=metric, start=start, step=step)
    if atype == "sorted":
        order = rng.choice(SORT_ORDERS)
        return SortedSpec(order=order)  # type: ignore[arg-type]
    if atype == "positional":
        chartype = rng.choice([ct for ct in POSITIONAL_ALLOWED if ct != "unique"])
        index = rng.choice(POSITIONAL_INDEXES)
        return PositionalSpec(character_type=chartype, index=index)  # type: ignore[arg-type]
    raise ValueError("Unknown attribute type")
