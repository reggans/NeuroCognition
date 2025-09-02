from __future__ import annotations
from typing import Dict, Any, List, Set
from .sampling import sample_axis_spec
from .compatibility import combine_and_validate
from .cell_generator import generate_cell_string
from .util_rng import RNG
from .exceptions import IncompatibleAttributesError
from .constraint_specs import spec_to_dict
from .validator import validate_grid

MAX_AXIS_ATTEMPTS = 25
MIN_ROW_HAMMING = 2  # minimal pairwise hamming distance within a row if lengths match
MIN_COL_HAMMING = 2


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        # treat completely different length as sufficiently different
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def enforce_row_col_variation(strings: List[List[str]]) -> bool:
    # Rows
    for r in range(3):
        for i in range(3):
            for j in range(i + 1, 3):
                if hamming(strings[r][i], strings[r][j]) < MIN_ROW_HAMMING:
                    return False
    # Cols
    for c in range(3):
        for i in range(3):
            for j in range(i + 1, 3):
                if hamming(strings[i][c], strings[j][c]) < MIN_COL_HAMMING:
                    return False
    return True


def generate_grid(seed: int | None = None) -> Dict[str, Any]:
    rng = RNG(seed)
    axis_attempts = 0
    last_reasons: List[str] = []
    while axis_attempts < MAX_AXIS_ATTEMPTS:
        axis_attempts += 1
        row_spec = sample_axis_spec(rng)
        col_spec = sample_axis_spec(rng)
        grid_constraints, reasons = combine_and_validate(row_spec, col_spec)
        if reasons:
            last_reasons = reasons
            continue
        used: Set[str] = set()
        strings: List[List[str]] = [["" for _ in range(3)] for _ in range(3)]
        try:
            for r in range(3):
                for c in range(3):
                    # attempt cell generation with variation check retries
                    for _ in range(10):
                        s = generate_cell_string(grid_constraints[r][c], rng, used)
                        strings[r][c] = s
                        # temporary assign and check variation for completed parts of row/col
                        row_ok = True
                        col_ok = True
                        # check row variation so far
                        row_vals = [
                            strings[r][k] for k in range(c + 1) if strings[r][k]
                        ]
                        for i in range(len(row_vals)):
                            for j in range(i + 1, len(row_vals)):
                                if hamming(row_vals[i], row_vals[j]) < MIN_ROW_HAMMING:
                                    row_ok = False
                                    break
                            if not row_ok:
                                break
                        # check column variation so far
                        col_vals = [
                            strings[k][c] for k in range(r + 1) if strings[k][c]
                        ]
                        for i in range(len(col_vals)):
                            for j in range(i + 1, len(col_vals)):
                                if hamming(col_vals[i], col_vals[j]) < MIN_COL_HAMMING:
                                    col_ok = False
                                    break
                            if not col_ok:
                                break
                        if row_ok and col_ok:
                            break
                        # else rollback this cell and try again
                        used.discard(s)
                        strings[r][c] = ""
                    if strings[r][c] == "":
                        raise RuntimeError("variation_generation_failed")
        except RuntimeError:
            last_reasons = ["generation_failed"]
            continue
        if not enforce_row_col_variation(strings):
            last_reasons = ["variation_check_failed"]
            continue
        val_reasons = validate_grid(strings, grid_constraints)
        if val_reasons:
            last_reasons = val_reasons
            continue
        # post-process: shuffle strings of cells without ordering or positional constraints
        for r in range(3):
            for c in range(3):
                cc = grid_constraints[r][c]
                if not cc.ordering and not cc.positional_type:
                    lst = list(strings[r][c])
                    rng.shuffle(lst)
                    strings[r][c] = "".join(lst)
        answer = strings[2][2]
        question_grid = [
            [strings[r][c] if not (r == 2 and c == 2) else None for c in range(3)]
            for r in range(3)
        ]
        constraint_meta = {
            f"{r},{c}": grid_constraints[r][c].describe()
            for r in range(3)
            for c in range(3)
        }
        return {
            "question_grid": question_grid,
            "full_grid": strings,
            "answer": answer,
            "row_attribute": spec_to_dict(row_spec),
            "col_attribute": spec_to_dict(col_spec),
            "axis_attempts": axis_attempts,
            "cell_constraints": constraint_meta,
            "raw_constraints": grid_constraints,
            "validation_attempts_last": last_reasons,
        }
    raise IncompatibleAttributesError(last_reasons)
