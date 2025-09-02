from __future__ import annotations
from typing import List, Set, Tuple
from .util_rng import RNG
from .per_cell_constraints import CellConstraint
from .validator import cell_satisfies, constraint_violations
from .character_sets import CHARACTER_POOLS

# Basic character universe for mutations
ALPHABET = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()"
)


def mutate_char(s: str, rng: RNG) -> str:
    if not s:
        return s
    idx = rng.randrange(len(s))
    choices = [ch for ch in ALPHABET if ch != s[idx]]
    if not choices:
        return s
    return s[:idx] + rng.choice(choices) + s[idx + 1 :]


def break_ordering(s: str, rng: RNG) -> str:
    if len(s) < 2:
        return mutate_char(s, rng)
    i = rng.randrange(len(s) - 1)
    lst = list(s)
    lst[i], lst[i + 1] = lst[i + 1], lst[i]
    return "".join(lst)


def break_positional(s: str, cc: CellConstraint, rng: RNG) -> str:
    if not cc.positional_type or not s:
        return mutate_char(s, rng)
    pool_all = [
        ch for ch in ALPHABET if ch not in CHARACTER_POOLS.get(cc.positional_type, [])
    ]
    if not pool_all:
        pool_all = ALPHABET
    lst = list(s)
    if cc.positional_index_rule == "first":
        lst[0] = rng.choice(pool_all)
    elif cc.positional_index_rule == "last":
        lst[-1] = rng.choice(pool_all)
    elif cc.positional_index_rule == "even":
        for i in range(0, len(lst), 2):
            lst[i] = rng.choice(pool_all)
    elif cc.positional_index_rule == "odd":
        for i in range(1, len(lst), 2):
            lst[i] = rng.choice(pool_all)
    return "".join(lst)


def adjust_count(s: str, cc: CellConstraint, rng: RNG) -> str:
    # Try to alter a target count or parity
    lst = list(s)
    if cc.target_counts:
        t, val = next(iter(cc.target_counts.items()))
        # remove a character of that type if present
        pool = set(CHARACTER_POOLS.get(t, []))
        idxs = [i for i, ch in enumerate(lst) if ch in pool]
        if idxs:
            lst[idxs[0]] = rng.choice([ch for ch in ALPHABET if ch not in pool])
            return "".join(lst)
    # fallback
    return mutate_char(s, rng)


STRATEGY_BUILDERS = [
    lambda s, cc, rng: break_positional(s, cc, rng),
    lambda s, cc, rng: break_ordering(s, rng),
    lambda s, cc, rng: adjust_count(s, cc, rng),
    lambda s, cc, rng: mutate_char(s, rng),
]


def _compute_min_len_for_constraint(cc: CellConstraint) -> int:
    sum_exact = sum(cc.target_counts.values())
    min_len = sum_exact
    if cc.unique_exact is not None:
        min_len = max(min_len, cc.unique_exact)
    for t, rule in cc.parity_rules.items():
        if t in cc.target_counts:
            continue
        if rule == "odd":
            min_len = max(min_len, sum_exact + 1)
        elif rule == "even":
            min_len = max(min_len, sum_exact + 2)
    for t, k in cc.multiple_rules.items():
        if t in cc.target_counts:
            continue
        min_len = max(min_len, sum_exact + k)
    if min_len <= 0:
        min_len = 1
    return min_len


def _apply_variable_length(candidate: str, cc: CellConstraint, rng: RNG) -> str:
    if cc.fixed_length is not None:
        return candidate
    # only adjust if no ordering or positional constraints (to stay consistent with answer logic)
    # if cc.ordering or cc.positional_type:
    #     return candidate
    min_len = _compute_min_len_for_constraint(cc)
    upper = 10 if min_len < 10 else min_len + 2
    if upper > 20:
        upper = 20
    if min_len > upper:
        upper = min_len
    target_len = rng.randint(min_len, upper)
    if target_len == len(candidate):
        return candidate
    lst = list(candidate)
    if target_len > len(lst):
        # insert random chars
        for _ in range(target_len - len(lst)):
            pos = rng.randrange(len(lst) + 1)
            lst.insert(pos, rng.choice(ALPHABET))
    else:
        # remove random positions
        for _ in range(len(lst) - target_len):
            if not lst:
                break
            pos = rng.randrange(len(lst))
            del lst[pos]
    return "".join(lst)


def generate_distractors(
    correct: str,
    rng: RNG,
    existing: Set[str],
    answer_constraint: CellConstraint | None = None,
    count: int = 7,
    max_attempts: int = 150,
) -> List[str]:
    distractors: List[str] = []
    attempts = 0
    while len(distractors) < count and attempts < max_attempts:
        attempts += 1
        candidate = correct
        if answer_constraint:
            strat = rng.choice(STRATEGY_BUILDERS)
            candidate = strat(correct, answer_constraint, rng)
        else:
            candidate = mutate_char(correct, rng)
        # length variability only if no fixed_length, no ordering, no positional
        if answer_constraint and answer_constraint.fixed_length is None:
            candidate = _apply_variable_length(candidate, answer_constraint, rng)
        # shuffle if no ordering/positional constraints apply (mirrors answer post-processing)
        if (
            answer_constraint
            and not answer_constraint.ordering
            and not answer_constraint.positional_type
        ):
            lst = list(candidate)
            rng.shuffle(lst)
            candidate = "".join(lst)
        if candidate == correct:
            continue
        if candidate in existing or candidate in distractors:
            continue
        # Ensure candidate does NOT satisfy answer constraint (avoid second correct answer)
        if answer_constraint and cell_satisfies(candidate, answer_constraint):
            continue
        if answer_constraint:
            v = constraint_violations(candidate, answer_constraint)
            if not v:
                continue
        distractors.append(candidate)
    return distractors
