from __future__ import annotations
from typing import Dict, Any
import random
from .generator import generate_grid
from .distractors import generate_distractors
from .util_rng import RNG
from .per_cell_constraints import CellConstraint
from .attribute_audit import audit_and_apply_leaks


def generate_text_rapm_item(
    seed: int | None = None, debug: bool = False
) -> Dict[str, Any]:
    # When seed is None, we want non-deterministic behavior each call. We derive an internal seed
    # only for reproducibility within this single item generation if needed (e.g., for debugging),
    # but do not expose it outside.
    internal_seed = seed if seed is not None else random.getrandbits(64)
    data = generate_grid(seed=internal_seed)
    # Audit for attribute leaks and mutate constraints in place
    row_attr, col_attr, row_leaks, col_leaks = audit_and_apply_leaks(
        data["full_grid"],
        data["raw_constraints"],
        data["row_attribute"],
        data["col_attribute"],
    )
    # Rebuild cell constraint metadata to include leak additions
    updated_cell_meta = {
        f"{r},{c}": data["raw_constraints"][r][c].describe()
        for r in range(3)
        for c in range(3)
    }
    rng = RNG(internal_seed)
    existing = set(ch for row in data["full_grid"] for ch in row)
    answer_constraint: CellConstraint | None = (
        data.get("raw_constraints", [[None] * 3] * 3)[2][2]
        if data.get("raw_constraints")
        else None
    )
    distractors = generate_distractors(
        data["answer"], rng, existing, answer_constraint=answer_constraint
    )
    if len(distractors) < 7:
        base = data["answer"]
        i = 0
        while len(distractors) < 7:
            variant = f"{base}{i}"
            if (
                variant not in distractors
                and variant not in existing
                and variant != base
            ):
                distractors.append(variant)
            i += 1
    options = distractors + [data["answer"]]
    rng.shuffle(options)
    correct_index = options.index(data["answer"])
    out = {
        "question_grid": data["question_grid"],
        "options": options,
        "correct_index": correct_index,
        "answer": data["answer"],
        "row_attribute": row_attr,
        "col_attribute": col_attr,
        "cell_constraints": updated_cell_meta,
        "axis_attempts": data["axis_attempts"],
        "row_leaks": row_leaks,
        "col_leaks": col_leaks,
    }
    if debug:
        out["_internal_seed"] = internal_seed  # only in debug for possible reproduction
        out["full_grid"] = data["full_grid"]
        if data.get("raw_constraints"):
            out["raw_constraints"] = [
                [cc.describe() if cc else None for cc in row]
                for row in data["raw_constraints"]
            ]
        else:
            out["raw_constraints"] = None
        if "validation_attempts_last" in data:
            out["validation_attempts_last"] = data["validation_attempts_last"]
    return out
