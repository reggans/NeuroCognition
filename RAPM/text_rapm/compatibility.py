from __future__ import annotations
from typing import List, Tuple, Dict, Optional, cast
from .constraint_specs import (
    AxisSpec,
    CharacterSetTypeSpec,
    TypeCountSpec,
    QuantConstantSpec,
    QuantProgressionSpec,
    SortedSpec,
    PositionalSpec,
)
from .character_sets import CHARACTER_POOLS
from .per_cell_constraints import CellConstraint

MAX_LENGTH = 20

# Reason codes constants
EMPTY_POOL = "empty_pool"
DISJOINT_TYPES = "disjoint_types"
COUNT_EXCEEDS_LENGTH = "count_exceeds_length"
UNIQUE_EXCEEDS_POOL = "unique_exceeds_pool"
SORTED_INCOMPATIBLE = "sorted_incompatible"
POSITIONAL_INCOMPATIBLE = "positional_incompatible"
CONFLICT_SORTED_POSITIONAL = "conflict_sorted_positional"
PROGRESSION_OUT_OF_RANGE = "progression_out_of_range"
UNSAT_CELL = "unsatisfiable_cell"
PARITY_CONFLICT = "parity_conflict"
MULTIPLE_CONFLICT = "multiple_conflict"
PARITY_INFEASIBLE = "parity_infeasible"
TARGET_INFEASIBLE = "target_infeasible"
PARITY_NONZERO_INFEASIBLE = "parity_nonzero_infeasible"
MULTIPLE_NONZERO_INFEASIBLE = "multiple_nonzero_infeasible"
MULTIPLE_INFEASIBLE = "multiple_infeasible"  # new: no chars available for multiple rule


def axis_progression_values(spec: AxisSpec) -> List[int] | None:
    if isinstance(spec, QuantProgressionSpec):
        v1 = spec.start
        v2 = spec.start + spec.step
        v3 = spec.start + 2 * spec.step
        return [v1, v2, v3]
    return None


def _has_chars_for_type(allowed: Optional[List[str]], t: str) -> bool:
    if t == "length" or t == "unique":
        return True
    pool = CHARACTER_POOLS.get(t)
    if pool is None:
        return False
    if allowed is None:
        return True
    aset = set(allowed)
    return any(ch in aset for ch in pool)


def build_cell_constraint(
    row_spec: AxisSpec, col_spec: AxisSpec, r: int, c: int
) -> Tuple[CellConstraint, List[str]]:
    reasons: List[str] = []
    cc = CellConstraint()

    allowed_chars: Optional[List[str]] = None

    def restrict_to_char_set(t: str):
        nonlocal allowed_chars
        pool = CHARACTER_POOLS.get(t)
        if pool is None:
            return
        if allowed_chars is None:
            allowed_chars = list(pool)
        else:
            current = cast(List[str], allowed_chars)
            allowed_chars = [ch for ch in current if ch in pool]

    for spec in (row_spec, col_spec):
        if isinstance(spec, CharacterSetTypeSpec):
            restrict_to_char_set(spec.character_set_type)

    positional_specs = [
        s for s in (row_spec, col_spec) if isinstance(s, PositionalSpec)
    ]
    sorted_specs = [s for s in (row_spec, col_spec) if isinstance(s, SortedSpec)]

    for spec in positional_specs:
        if spec.character_type not in CHARACTER_POOLS:
            reasons.append(POSITIONAL_INCOMPATIBLE)
        else:
            pos_pool = CHARACTER_POOLS[spec.character_type]
            if allowed_chars is not None and not any(
                ch in pos_pool for ch in allowed_chars
            ):
                reasons.append(POSITIONAL_INCOMPATIBLE)
        cc.positional_type = spec.character_type
        cc.positional_index_rule = spec.index

    for spec in sorted_specs:
        cc.ordering = spec.order

    if cc.ordering and positional_specs:
        reasons.append(CONFLICT_SORTED_POSITIONAL)

    if cc.ordering in ("ascending", "descending"):
        if not any(isinstance(s, CharacterSetTypeSpec) for s in (row_spec, col_spec)):
            reasons.append(SORTED_INCOMPATIBLE)

    for spec in (row_spec, col_spec):
        if isinstance(spec, QuantConstantSpec):
            if spec.metric == "length":
                if cc.fixed_length is not None and cc.fixed_length != spec.value:
                    reasons.append(UNSAT_CELL)
                cc.fixed_length = spec.value
            elif spec.metric == "unique":
                if cc.unique_exact is not None and cc.unique_exact != spec.value:
                    reasons.append(UNSAT_CELL)
                cc.unique_exact = spec.value
            else:
                existing = cc.target_counts.get(spec.metric)
                if existing is not None and existing != spec.value:
                    reasons.append(UNSAT_CELL)
                cc.target_counts[spec.metric] = spec.value

    for spec in (row_spec, col_spec):
        if isinstance(spec, TypeCountSpec):
            if spec.rule in ("even", "odd"):
                if spec.character_type in cc.target_counts:
                    if (
                        cc.target_counts[spec.character_type] % 2 == 0
                        and spec.rule == "odd"
                    ) or (
                        cc.target_counts[spec.character_type] % 2 == 1
                        and spec.rule == "even"
                    ):
                        reasons.append(PARITY_CONFLICT)
                cc.parity_rules[spec.character_type] = spec.rule
            elif spec.rule == "multiple" and spec.multiple_of:
                if (
                    spec.character_type in cc.target_counts
                    and cc.target_counts[spec.character_type] % spec.multiple_of != 0
                ):
                    reasons.append(MULTIPLE_CONFLICT)
                cc.multiple_rules[spec.character_type] = spec.multiple_of

    row_prog = axis_progression_values(row_spec)
    col_prog = axis_progression_values(col_spec)
    if row_prog is not None:
        val = row_prog[r]
        if isinstance(row_spec, QuantProgressionSpec):
            if row_spec.metric == "length":
                if cc.fixed_length is not None and cc.fixed_length != val:
                    reasons.append(UNSAT_CELL)
                cc.fixed_length = val
            elif row_spec.metric == "unique":
                if cc.unique_exact is not None and cc.unique_exact != val:
                    reasons.append(UNSAT_CELL)
                cc.unique_exact = val
            else:
                existing = cc.target_counts.get(row_spec.metric)
                if existing is not None and existing != val:
                    reasons.append(UNSAT_CELL)
                cc.target_counts[row_spec.metric] = val
            if val < 1 or val > MAX_LENGTH:
                reasons.append(PROGRESSION_OUT_OF_RANGE)
    if col_prog is not None:
        val = col_prog[c]
        if isinstance(col_spec, QuantProgressionSpec):
            if col_spec.metric == "length":
                if cc.fixed_length is not None and cc.fixed_length != val:
                    reasons.append(UNSAT_CELL)
                cc.fixed_length = val
            elif col_spec.metric == "unique":
                if cc.unique_exact is not None and cc.unique_exact != val:
                    reasons.append(UNSAT_CELL)
                cc.unique_exact = val
            else:
                existing = cc.target_counts.get(col_spec.metric)
                if existing is not None and existing != val:
                    reasons.append(UNSAT_CELL)
                cc.target_counts[col_spec.metric] = val
            if val < 1 or val > MAX_LENGTH:
                reasons.append(PROGRESSION_OUT_OF_RANGE)

    if cc.fixed_length is not None:
        total_exact = sum(cc.target_counts.values())
        if total_exact > cc.fixed_length:
            reasons.append(COUNT_EXCEEDS_LENGTH)
        remaining_after_exact = cc.fixed_length - total_exact
        for t, rule in cc.parity_rules.items():
            if t in cc.target_counts:
                continue
            if rule == "odd":
                if remaining_after_exact <= 0:
                    reasons.append(PARITY_INFEASIBLE)
            if rule == "even":
                if remaining_after_exact <= 1:
                    reasons.append(PARITY_INFEASIBLE)
        if cc.unique_exact is not None and cc.unique_exact > cc.fixed_length:
            reasons.append(UNIQUE_EXCEEDS_POOL)
    if cc.unique_exact is not None and allowed_chars is not None:
        if cc.unique_exact > len(set(allowed_chars)):
            reasons.append(UNIQUE_EXCEEDS_POOL)

    # Disjoint / infeasible type checks
    for t, v in cc.target_counts.items():
        if v > 0 and not _has_chars_for_type(allowed_chars, t):
            reasons.append(TARGET_INFEASIBLE)
    # parity infeasible now for any rule if chars absent
    for t, rule in cc.parity_rules.items():
        if not _has_chars_for_type(allowed_chars, t):
            reasons.append(PARITY_INFEASIBLE)
    # multiple infeasible if chars absent
    for t in cc.multiple_rules:
        if not _has_chars_for_type(allowed_chars, t):
            reasons.append(MULTIPLE_INFEASIBLE)

    if allowed_chars is not None and len(allowed_chars) == 0:
        reasons.append(EMPTY_POOL)
    cc.allowed_chars = allowed_chars

    # after progression and basic feasibility, enforce new non-zero feasibility logic
    total_exact = sum(cc.target_counts.values())
    if cc.fixed_length is not None:
        remaining = cc.fixed_length - total_exact
        needed = 0
        for t, rule in cc.parity_rules.items():
            if t in cc.target_counts:
                continue
            needed += 1 if rule == "odd" else 2
        for t, k in cc.multiple_rules.items():
            if t in cc.target_counts:
                continue
            needed += k
        if needed > remaining:
            if any(t not in cc.target_counts for t in cc.parity_rules):
                reasons.append(PARITY_NONZERO_INFEASIBLE)
            if any(t not in cc.target_counts for t in cc.multiple_rules):
                reasons.append(MULTIPLE_NONZERO_INFEASIBLE)

    return cc, reasons


def combine_and_validate(
    row_spec: AxisSpec, col_spec: AxisSpec
) -> Tuple[List[List[CellConstraint]], List[str]]:
    grid: List[List[CellConstraint]] = []
    all_reasons: List[str] = []
    for r in range(3):
        row_list: List[CellConstraint] = []
        for c in range(3):
            cc, reasons = build_cell_constraint(row_spec, col_spec, r, c)
            row_list.append(cc)
            if reasons:
                all_reasons.extend([f"cell({r},{c}):{reason}" for reason in reasons])
        grid.append(row_list)
    return grid, all_reasons
