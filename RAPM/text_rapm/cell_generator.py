from __future__ import annotations
from typing import List, Set, Dict
from .per_cell_constraints import CellConstraint
from .character_sets import CHARACTER_POOLS, case_key
from .util_rng import RNG

ALL_BASE_POOL: List[str] = (
    CHARACTER_POOLS["letters"] + CHARACTER_POOLS["digits"] + CHARACTER_POOLS["symbols"]
)


def get_pool(t: str) -> List[str]:
    if t == "unique":
        return ALL_BASE_POOL
    return CHARACTER_POOLS.get(t, ALL_BASE_POOL)


def enforce_parity(val: int, rule: str) -> bool:
    return (rule == "even" and val % 2 == 0 and val > 0) or (
        rule == "odd" and val % 2 == 1 and val > 0
    )


def generate_cell_string(
    cc: CellConstraint, rng: RNG, used: Set[str], max_attempts: int = 60
) -> str:
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        fixed_len = cc.fixed_length
        sum_exact = sum(cc.target_counts.values())
        min_len = sum_exact
        if cc.unique_exact is not None:
            min_len = max(min_len, cc.unique_exact)
        # parity rules minimums (even now requires at least 2, odd at least 1) when not already fixed by target_counts
        for t, rule in cc.parity_rules.items():
            if t in cc.target_counts:
                continue
            if rule == "odd":
                min_len = max(min_len, sum_exact + 1)
            elif rule == "even":
                min_len = max(min_len, sum_exact + 2)
        # multiple rules minimum (must allocate at least k if not already exact)
        for t, k in cc.multiple_rules.items():
            if t in cc.target_counts:
                continue
            min_len = max(min_len, sum_exact + k)
        if fixed_len is not None:
            L = fixed_len
        else:
            # variable length strategy
            if min_len <= 0:
                min_len = 1
            upper = 10 if min_len < 10 else min_len + 2
            if upper > 20:
                upper = 20
            if min_len > upper:
                upper = min_len
            L = rng.randint(min_len, upper)
        if L > 20:
            continue
        if L < min_len:
            continue
        chars: List[str] = []
        remaining_slots = L
        invalid = False
        # exact counts
        for t, cnt_val in cc.target_counts.items():
            pool = get_pool(t)
            if cnt_val > remaining_slots:
                invalid = True
                break
            for _ in range(cnt_val):
                chars.append(rng.choice(pool))
            remaining_slots -= cnt_val
        if invalid or remaining_slots < 0:
            continue
        # parity & multiples provisional allocation (guarantee non-zero counts)
        provisional_alloc: Dict[str, int] = {}
        for t, rule in cc.parity_rules.items():
            if t in cc.target_counts:
                continue
            alloc = 2 if rule == "even" else 1
            if alloc > remaining_slots:
                invalid = True
                break
            provisional_alloc[t] = alloc
            remaining_slots -= alloc
        if invalid:
            continue
        for t, k in cc.multiple_rules.items():
            if t in cc.target_counts:
                continue
            if k > remaining_slots:
                invalid = True
                break
            provisional_alloc[t] = provisional_alloc.get(t, 0) + k
            remaining_slots -= k
        if invalid:
            continue
        # place provisional allocations
        for t, alloc in provisional_alloc.items():
            pool = get_pool(t)
            for _ in range(alloc):
                chars.append(rng.choice(pool))
        # fill remaining
        base_pool = cc.allowed_chars if cc.allowed_chars else ALL_BASE_POOL
        for _ in range(remaining_slots):
            chars.append(rng.choice(base_pool))
        # positional
        if cc.positional_type:
            pos_pool = get_pool(cc.positional_type)
            if cc.positional_index_rule == "first" and L >= 1:
                chars[0] = rng.choice(pos_pool)
            elif cc.positional_index_rule == "last" and L >= 1:
                chars[-1] = rng.choice(pos_pool)
            elif cc.positional_index_rule == "even":
                for i in range(0, L, 2):
                    chars[i] = rng.choice(pos_pool)
            elif cc.positional_index_rule == "odd":
                for i in range(1, L, 2):
                    chars[i] = rng.choice(pos_pool)
        # unique
        if cc.unique_exact is not None:
            distinct = list(dict.fromkeys(chars))
            if len(distinct) > cc.unique_exact:
                while len(set(chars)) > cc.unique_exact and len(chars) > 1:
                    idx = rng.randrange(len(chars))
                    chars[idx] = chars[0]
            elif len(distinct) < cc.unique_exact:
                pool = base_pool
                available = [ch for ch in pool if ch not in distinct]
                while len(distinct) < cc.unique_exact and available:
                    ch_new = rng.choice(available)
                    replace_idx = rng.randrange(len(chars))
                    chars[replace_idx] = ch_new
                    distinct = list(dict.fromkeys(chars))
                    available = [ch for ch in pool if ch not in distinct]
                if len(distinct) != cc.unique_exact:
                    continue
        # ordering
        if cc.ordering == "ascending":
            chars = sorted(chars, key=case_key)
        elif cc.ordering == "descending":
            chars = sorted(chars, key=case_key, reverse=True)
        elif cc.ordering == "mixed":
            rng.shuffle(chars)
        s = "".join(chars)
        if s in used:
            continue

        # quick revalidation of counts
        def cnt(t: str) -> int:
            if t == "unique":
                return len(set(s))
            if t == "length":
                return len(s)
            pool = CHARACTER_POOLS.get(t)
            if not pool:
                return 0
            sp = set(pool)
            return sum(1 for ch in s if ch in sp)

        valid = True
        for t, v in cc.target_counts.items():
            if cnt(t) != v:
                valid = False
                break
        if not valid:
            continue
        for t, rule in cc.parity_rules.items():
            val = cnt(t)
            if not (
                (rule == "even" and val % 2 == 0 and val > 0)
                or (rule == "odd" and val % 2 == 1 and val > 0)
            ):
                valid = False
                break
        if not valid:
            continue
        for t, k in cc.multiple_rules.items():
            val = cnt(t)
            if val == 0 or val % k != 0:
                valid = False
                break
        if not valid:
            continue
        if cc.unique_exact is not None and cnt("unique") != cc.unique_exact:
            continue
        used.add(s)
        return s
    raise RuntimeError("Failed to generate cell string after attempts")
