from __future__ import annotations
from typing import Dict, List
from .per_cell_constraints import CellConstraint
from .character_sets import CHARACTER_POOLS, case_key

# Build reverse lookup for fast membership
TYPE_SETS = {t: set(v) for t, v in CHARACTER_POOLS.items()}


def count_type(s: str, t: str) -> int:
    if t == "length":
        return len(s)
    if t == "unique":
        return len(set(s))
    pool = TYPE_SETS.get(t)
    if pool is None:
        return 0
    return sum(1 for ch in s if ch in pool)


def check_ordering(s: str, ordering: str) -> bool:
    if ordering == "mixed":
        asc = "".join(sorted(s, key=case_key))
        desc = "".join(sorted(s, key=case_key, reverse=True))
        return s != asc and s != desc
    if ordering == "ascending":
        return list(s) == sorted(s, key=case_key)
    if ordering == "descending":
        return list(s) == sorted(s, key=case_key, reverse=True)
    return True


def check_positional(s: str, cc: CellConstraint) -> bool:
    if not cc.positional_type:
        return True
    target_type = cc.positional_type
    pool = TYPE_SETS.get(target_type, set())
    if cc.positional_index_rule == "first":
        return len(s) > 0 and s[0] in pool
    if cc.positional_index_rule == "last":
        return len(s) > 0 and s[-1] in pool
    if cc.positional_index_rule == "even":
        return all(s[i] in pool for i in range(0, len(s), 2))
    if cc.positional_index_rule == "odd":
        return all(s[i] in pool for i in range(1, len(s), 2))
    return True


def cell_satisfies(s: str, cc: CellConstraint) -> bool:
    # length
    if cc.fixed_length is not None and len(s) != cc.fixed_length:
        return False
    # target counts
    for t, v in cc.target_counts.items():
        if count_type(s, t) != v:
            return False
    # parity
    for t, rule in cc.parity_rules.items():
        val = count_type(s, t)
        if val == 0:
            return False
        if rule == "even" and (val % 2 != 0):
            return False
        if rule == "odd" and (val % 2 != 1):
            return False
    # multiple rules
    for t, k in cc.multiple_rules.items():
        val = count_type(s, t)
        if val == 0 or val % k != 0:
            return False
    # unique
    if cc.unique_exact is not None and len(set(s)) != cc.unique_exact:
        return False
    # ordering
    if cc.ordering and not check_ordering(s, cc.ordering):
        return False
    # positional
    if not check_positional(s, cc):
        return False
    return True


def constraint_violations(s: str, cc: CellConstraint) -> List[str]:
    violations: List[str] = []
    if cc.fixed_length is not None and len(s) != cc.fixed_length:
        violations.append("length")
    for t, v in cc.target_counts.items():
        if count_type(s, t) != v:
            violations.append(f"target:{t}")
    for t, rule in cc.parity_rules.items():
        val = count_type(s, t)
        if val == 0:
            violations.append(f"parity_zero:{t}")
        elif (rule == "even" and val % 2 != 0) or (rule == "odd" and val % 2 != 1):
            violations.append(f"parity:{t}")
    for t, k in cc.multiple_rules.items():
        val = count_type(s, t)
        if val == 0 or val % k != 0:
            violations.append(f"multiple:{t}")
    if cc.unique_exact is not None and len(set(s)) != cc.unique_exact:
        violations.append("unique")
    if cc.ordering and not check_ordering(s, cc.ordering):
        violations.append("ordering")
    if cc.positional_type and not check_positional(s, cc):
        violations.append("positional")
    return violations


def validate_grid(
    grid: List[List[str]], constraints: List[List[CellConstraint]]
) -> List[str]:
    reasons: List[str] = []
    for r in range(3):
        for c in range(3):
            if not cell_satisfies(grid[r][c], constraints[r][c]):
                reasons.append(f"cell({r},{c}) fails constraints")
    return reasons


__all__ = ["cell_satisfies", "validate_grid", "constraint_violations"]
