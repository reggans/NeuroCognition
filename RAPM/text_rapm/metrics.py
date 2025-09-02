from __future__ import annotations
from typing import Iterable, Callable
from collections import Counter


def count_type(s: str, predicate: Callable[[str], bool]) -> int:
    return sum(1 for ch in s if predicate(ch))


def unique_count(s: str) -> int:
    return len(set(s))


def is_even(n: int) -> bool:
    return n % 2 == 0


def is_odd(n: int) -> bool:
    return n % 2 == 1


def matches_parity_rule(n: int, rule: str) -> bool:
    if rule == "even":
        return is_even(n)
    if rule == "odd":
        return is_odd(n)
    raise ValueError(f"Unknown parity rule {rule}")


def matches_multiple(n: int, k: int) -> bool:
    return n % k == 0
