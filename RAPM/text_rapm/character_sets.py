from dataclasses import dataclass
from typing import Dict, List, Set
import string

SYMBOLS = list("!@#$%^&*()_+-=[]{}|;:,.<>?")

LETTER_UPPER = list(string.ascii_uppercase)
LETTER_LOWER = list(string.ascii_lowercase)
VOWELS_LOWER = list("aeiou")
VOWELS_UPPER = [c.upper() for c in VOWELS_LOWER]
CONSONANTS_LOWER = [c for c in LETTER_LOWER if c not in VOWELS_LOWER]
CONSONANTS_UPPER = [c.upper() for c in CONSONANTS_LOWER]
DIGITS = list(string.digits)

CHARACTER_POOLS: Dict[str, List[str]] = {
    "letters": LETTER_UPPER + LETTER_LOWER,
    "letters-uppercase": LETTER_UPPER,
    "letters-lowercase": LETTER_LOWER,
    "vowels": VOWELS_UPPER + VOWELS_LOWER,
    "vowels-uppercase": VOWELS_UPPER,
    "vowels-lowercase": VOWELS_LOWER,
    "consonants": CONSONANTS_UPPER + CONSONANTS_LOWER,
    "consonants-uppercase": CONSONANTS_UPPER,
    "consonants-lowercase": CONSONANTS_LOWER,
    "digits": DIGITS,
    "symbols": SYMBOLS,
}

# Types that can be used for positional constraints (unique is excluded)
POSITIONAL_ALLOWED: Set[str] = set(CHARACTER_POOLS.keys()) - {"unique"}

HOMOGENEOUS_TYPES = {"letters", "digits", "symbols"}


def case_key(ch: str) -> str:
    return ch.lower()


def is_homogeneous(pool_types: Set[str]) -> bool:
    # Accept if all ultimately map to a single broad category letters/digits/symbols
    if len(pool_types) == 1:
        t = next(iter(pool_types))
        if t in HOMOGENEOUS_TYPES:
            return True
    return False
