from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from .per_cell_constraints import CellConstraint
from .character_sets import CHARACTER_POOLS
from .validator import count_type as v_count_type, check_ordering, check_positional

# Metrics available for leak detection (excluding 'unique' where special handling needed)
ALL_METRICS = list(CHARACTER_POOLS.keys()) + ["length", "unique"]
CHAR_SET_TYPES = ["letters", "digits", "symbols"]
PARITY_RULES = ["even", "odd"]
MULTIPLE_CHOICES = [2, 3, 4]
SORT_ORDERS = ["ascending", "descending", "mixed"]
POSITION_INDEXES = ["first", "last", "even", "odd"]


def _all_chars_in_set(strings: List[str], set_name: str) -> bool:
    pool = set(CHARACTER_POOLS.get(set_name, []))
    return all(all(ch in pool for ch in s) and len(s) > 0 for s in strings)


def _detect_character_set_leak(
    strings: List[str], chosen_attribute: Dict[str, Any]
) -> List[Dict[str, Any]]:
    leaks: List[Dict[str, Any]] = []
    if chosen_attribute.get("attribute") == "character_set_type":
        return leaks
    for cst in CHAR_SET_TYPES:
        if _all_chars_in_set(strings, cst):
            leaks.append({"attribute": "character_set_type", "character_set_type": cst})
    return leaks


def _detect_type_count_leaks(
    strings: List[str],
    constraints: List[CellConstraint],
    chosen_attribute: Dict[str, Any],
) -> List[Dict[str, Any]]:
    # Evaluate parity and multiple conditions for every metric
    leaks: List[Dict[str, Any]] = []
    if chosen_attribute.get("attribute") == "type_count":
        # Still allow additional type_count leaks for different character types
        pass
    for metric in CHARACTER_POOLS.keys():
        counts = [sum(1 for ch in s if ch in CHARACTER_POOLS[metric]) for s in strings]
        if any(c == 0 for c in counts):
            continue  # non-zero enforcement
        # Parity
        if all(c % 2 == 0 for c in counts) and not _already_has_parity(
            constraints, metric, "even"
        ):
            leaks.append(
                {
                    "attribute": "type_count",
                    "character_type": metric,
                    "rule": "even",
                    "multiple_of": None,
                }
            )
        if all(c % 2 == 1 for c in counts) and not _already_has_parity(
            constraints, metric, "odd"
        ):
            leaks.append(
                {
                    "attribute": "type_count",
                    "character_type": metric,
                    "rule": "odd",
                    "multiple_of": None,
                }
            )
        # Multiple-of
        for m in MULTIPLE_CHOICES:
            if all(c % m == 0 for c in counts):
                # ensure at least one group (non-zero) already guaranteed
                if not _already_has_multiple(constraints, metric, m):
                    leaks.append(
                        {
                            "attribute": "type_count",
                            "character_type": metric,
                            "rule": "multiple",
                            "multiple_of": m,
                        }
                    )
    return leaks


def _already_has_parity(
    constraints: List[CellConstraint], metric: str, rule: str
) -> bool:
    return all(
        cc.parity_rules.get(metric) == rule or metric not in cc.parity_rules
        for cc in constraints
    )


def _already_has_multiple(
    constraints: List[CellConstraint], metric: str, k: int
) -> bool:
    return all(
        cc.multiple_rules.get(metric) == k or metric not in cc.multiple_rules
        for cc in constraints
    )


def _detect_quant_constant_leaks(
    strings: List[str], chosen_attribute: Dict[str, Any]
) -> List[Dict[str, Any]]:
    leaks: List[Dict[str, Any]] = []
    for metric in ALL_METRICS:
        if metric == "length":
            vals = [len(s) for s in strings]
        elif metric == "unique":
            vals = [len(set(s)) for s in strings]
        else:
            pool = set(CHARACTER_POOLS.get(metric, []))
            vals = [sum(1 for ch in s if ch in pool) for s in strings]
        if all(v == vals[0] for v in vals):
            v = vals[0]
            if 2 <= v <= 5 or metric not in ("length", "unique"):
                leaks.append(
                    {"attribute": "quant_constant", "metric": metric, "value": v}
                )
    return leaks


def _detect_quant_progression_leaks(
    strings: List[str], chosen_attribute: Dict[str, Any]
) -> List[Dict[str, Any]]:
    leaks: List[Dict[str, Any]] = []
    for metric in ALL_METRICS:
        if metric == "length":
            seq = [len(s) for s in strings]
        elif metric == "unique":
            seq = [len(set(s)) for s in strings]
        else:
            pool = set(CHARACTER_POOLS.get(metric, []))
            seq = [sum(1 for ch in s if ch in pool) for s in strings]
        a, b, c = seq
        # constant handled by quant_constant
        if a == b == c:
            continue
        # arithmetic progression detection
        if (b - a) == (c - b):
            step = b - a
            if step > 0 and 1 <= step <= 3 and 1 <= a <= 3:
                leaks.append(
                    {
                        "attribute": "quant_progression",
                        "metric": metric,
                        "start": a,
                        "step": step,
                    }
                )
    return leaks


def _is_sorted_order(s: str) -> str | None:
    asc = "".join(sorted(s))
    desc = "".join(sorted(s, reverse=True))
    if s == asc:
        return "ascending"
    if s == desc:
        return "descending"
    # mixed: neither asc nor desc
    if s != asc and s != desc:
        return "mixed"
    return None


def _detect_sorted_leak(
    strings: List[str],
    constraints: List[CellConstraint],
    chosen_attribute: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if chosen_attribute.get("attribute") == "sorted":
        return []
    orders = [_is_sorted_order(s) for s in strings]
    if any(o is None for o in orders):
        return []
    if all(o == orders[0] for o in orders):
        order = orders[0]
        # cannot add if any positional constraint present (incompatibility)
        if not any(cc.positional_type for cc in constraints):
            return [{"attribute": "sorted", "order": order}]
    return []


def _check_positional_rule(s: str, chartype: str, index: str) -> bool:
    pool = set(CHARACTER_POOLS.get(chartype, []))
    if index == "first":
        return len(s) > 0 and s[0] in pool
    if index == "last":
        return len(s) > 0 and s[-1] in pool
    if index == "even":
        return all(s[i] in pool for i in range(0, len(s), 2)) and len(s) > 0
    if index == "odd":
        return all(s[i] in pool for i in range(1, len(s), 2)) and len(s) > 0
    return False


def _detect_positional_leaks(
    strings: List[str],
    constraints: List[CellConstraint],
    chosen_attribute: Dict[str, Any],
) -> List[Dict[str, Any]]:
    leaks: List[Dict[str, Any]] = []
    if chosen_attribute.get("attribute") == "positional":
        return leaks
    # Skip if any existing ordering constraint (incompatible) or existing positional (already constrained)
    if any(cc.ordering for cc in constraints):
        return leaks
    for chartype in CHARACTER_POOLS.keys():
        if chartype == "unique":
            continue
        for idx in POSITION_INDEXES:
            if all(_check_positional_rule(s, chartype, idx) for s in strings):
                leaks.append(
                    {
                        "attribute": "positional",
                        "character_type": chartype,
                        "index": idx,
                    }
                )
    return leaks


def detect_axis_leaks(
    strings: List[str],
    constraints: List[CellConstraint],
    chosen_attribute: Dict[str, Any],
) -> List[Dict[str, Any]]:
    leaks: List[Dict[str, Any]] = []
    # Order of detection does not imply priority; we collect all
    leaks.extend(_detect_character_set_leak(strings, chosen_attribute))
    leaks.extend(_detect_type_count_leaks(strings, constraints, chosen_attribute))
    leaks.extend(_detect_quant_constant_leaks(strings, chosen_attribute))
    leaks.extend(_detect_quant_progression_leaks(strings, chosen_attribute))
    leaks.extend(_detect_sorted_leak(strings, constraints, chosen_attribute))
    leaks.extend(_detect_positional_leaks(strings, constraints, chosen_attribute))
    # Deduplicate identical dicts
    unique: List[Dict[str, Any]] = []
    seen = set()
    for spec in leaks:
        key = tuple(sorted(spec.items()))
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def apply_leaks_to_constraints(
    leaks: List[Dict[str, Any]], constraints: List[CellConstraint], strings: List[str]
) -> None:
    # Mutate constraints in place to include leak specifications
    for spec in leaks:
        attr = spec.get("attribute")
        if attr == "character_set_type":
            cst = spec["character_set_type"]
            pool = set(CHARACTER_POOLS.get(cst, []))
            for cc, s in zip(constraints, strings):
                if all(ch in pool for ch in s):
                    # Narrow allowed_chars if present else set
                    if cc.allowed_chars is None:
                        cc.allowed_chars = list(pool)
                    else:
                        cc.allowed_chars = [ch for ch in cc.allowed_chars if ch in pool]
        elif attr == "type_count":
            t = spec["character_type"]
            rule = spec["rule"]
            if rule in ("even", "odd"):
                for cc in constraints:
                    if t not in cc.target_counts and t not in cc.parity_rules:
                        cc.parity_rules[t] = rule
            elif rule == "multiple":
                k = spec.get("multiple_of")
                if k:
                    for cc in constraints:
                        if t not in cc.target_counts and t not in cc.multiple_rules:
                            cc.multiple_rules[t] = k
        elif attr == "quant_constant":
            metric = spec["metric"]
            val = spec["value"]
            for cc, s in zip(constraints, strings):
                if metric == "length":
                    if cc.fixed_length is None:
                        cc.fixed_length = len(s)
                elif metric == "unique":
                    if cc.unique_exact is None:
                        cc.unique_exact = len(set(s))
                else:
                    if metric not in cc.target_counts:
                        cc.target_counts[metric] = val
        elif attr == "quant_progression":
            metric = spec["metric"]
            # Determine per-string values
            for cc, s in zip(constraints, strings):
                if metric == "length":
                    if cc.fixed_length is None:
                        cc.fixed_length = len(s)
                elif metric == "unique":
                    if cc.unique_exact is None:
                        cc.unique_exact = len(set(s))
                else:
                    val = sum(
                        1 for ch in s if ch in set(CHARACTER_POOLS.get(metric, []))
                    )
                    if metric not in cc.target_counts:
                        cc.target_counts[metric] = val
        elif attr == "sorted":
            order = spec["order"]
            for cc in constraints:
                if cc.ordering is None and not cc.positional_type:
                    cc.ordering = order
        elif attr == "positional":
            chartype = spec["character_type"]
            index = spec["index"]
            for cc in constraints:
                if cc.positional_type is None and cc.ordering is None:
                    cc.positional_type = chartype
                    cc.positional_index_rule = index


def _simplify_character_type_overlaps(
    leaks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Phase 1: consolidate specs differing only by character_type
    grouped: Dict[Tuple[Tuple[str, Any], ...], List[Dict[str, Any]]] = {}
    for spec in leaks:
        if "character_type" not in spec:
            key = tuple(sorted(spec.items()))
            grouped.setdefault(key, []).append(spec)
            continue
        key_items = tuple(
            sorted((k, v) for k, v in spec.items() if k != "character_type")
        )
        grouped.setdefault(key_items, []).append(spec)
    interim: List[Dict[str, Any]] = []
    for key, specs in grouped.items():
        if len(specs) == 1:
            interim.append(specs[0])
            continue

        def pool_size(sp: Dict[str, Any]) -> int:
            ct = sp.get("character_type")
            return len(CHARACTER_POOLS.get(ct, [])) if isinstance(ct, str) else -1

        interim.append(max(specs, key=pool_size))
    # Phase 2: consolidate specs differing only by metric (excluding length/unique)
    grouped_metric: Dict[Tuple[Tuple[str, Any], ...], List[Dict[str, Any]]] = {}
    for spec in interim:
        metric = spec.get("metric")
        if metric not in CHARACTER_POOLS or metric in ("length", "unique"):
            key = tuple(sorted(spec.items()))
            grouped_metric.setdefault(key, []).append(spec)
            continue
        key_items = tuple(sorted((k, v) for k, v in spec.items() if k != "metric"))
        grouped_metric.setdefault(key_items, []).append(spec)
    final_specs: List[Dict[str, Any]] = []
    for key, specs in grouped_metric.items():
        if len(specs) == 1:
            final_specs.append(specs[0])
            continue

        def pool_size_metric(sp: Dict[str, Any]) -> int:
            m = sp.get("metric")
            return len(CHARACTER_POOLS.get(m, [])) if isinstance(m, str) else -1

        final_specs.append(max(specs, key=pool_size_metric))
    return final_specs


def audit_and_apply_leaks(
    full_grid: List[List[str]],
    raw_constraints: List[List[CellConstraint]],
    row_attribute: Dict[str, Any],
    col_attribute: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Report only leaks that are consistent across ALL rows (row_leaks) / ALL columns (col_leaks)
    # Each leak appears ONCE (no per-axis duplication, no axis field).

    def key_for(spec: Dict[str, Any]) -> Tuple:
        return tuple(sorted(spec.items()))

    # Rows
    per_row_leaks: List[List[Dict[str, Any]]] = []
    for r in range(3):
        leaks = detect_axis_leaks(full_grid[r], raw_constraints[r], row_attribute)
        per_row_leaks.append(leaks)
    row_leaks_out: List[Dict[str, Any]] = []
    if per_row_leaks:
        intersect_keys = set(key_for(spec) for spec in per_row_leaks[0])
        for leaks in per_row_leaks[1:]:
            intersect_keys &= set(key_for(spec) for spec in leaks)
        if intersect_keys:
            # representative specs from first row
            common_specs = [
                spec for spec in per_row_leaks[0] if key_for(spec) in intersect_keys
            ]
            # simplify overlapping positional leaks
            common_specs = _simplify_character_type_overlaps(common_specs)
            for r in range(3):
                apply_leaks_to_constraints(
                    common_specs, raw_constraints[r], full_grid[r]
                )
            row_leaks_out = common_specs  # no axis annotations

    # Columns
    per_col_leaks: List[List[Dict[str, Any]]] = []
    for c in range(3):
        strings = [full_grid[r][c] for r in range(3)]
        constraints = [raw_constraints[r][c] for r in range(3)]
        leaks = detect_axis_leaks(strings, constraints, col_attribute)
        per_col_leaks.append(leaks)
    col_leaks_out: List[Dict[str, Any]] = []
    if per_col_leaks:
        intersect_keys = set(key_for(spec) for spec in per_col_leaks[0])
        for leaks in per_col_leaks[1:]:
            intersect_keys &= set(key_for(spec) for spec in leaks)
        if intersect_keys:
            common_specs = [
                spec for spec in per_col_leaks[0] if key_for(spec) in intersect_keys
            ]
            common_specs = _simplify_character_type_overlaps(common_specs)
            for c in range(3):
                strings = [full_grid[r][c] for r in range(3)]
                constraints = [raw_constraints[r][c] for r in range(3)]
                apply_leaks_to_constraints(common_specs, constraints, strings)
            col_leaks_out = common_specs

    return row_attribute, col_attribute, row_leaks_out, col_leaks_out
