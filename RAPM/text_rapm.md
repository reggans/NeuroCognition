Available character types for RAPM evaluation:

- `letters`: Includes all alphabetic characters.
- `letters-uppercase`: Includes all uppercase alphabetic characters.
- `letters-lowercase`: Includes all lowercase alphabetic characters.
- `vowels`: Includes all vowel characters.
- `vowels-uppercase`: Includes all uppercase vowel characters.
- `vowels-lowercase`: Includes all lowercase vowel characters.
- `consonants`: Includes all consonant characters.
- `consonants-uppercase`: Includes all uppercase consonant characters.
- `consonants-lowercase`: Includes all lowercase consonant characters.
- `digits`: Includes all numeric characters.
- `symbols`: Includes all symbol characters.
- `unique`: Unique characters in the string.

Constraint level for RAPM evaluation:

- `string`: string-level constraints.
- `char`: character-level constraints, mostly in specific index/indexes.
- `grid`: grid-level constraints, either row or column. All 3 strings in the row/column must satisfy the constraint.

Below are the initial character/string attributes to build a pattern for text-based RAPM evaluation:

1. Character Set Type

   All chars in the string are (effectively constrained to) a specific character set. (Implementation: intersects with any positional / ordering feasibility checks.)

   - constraint_type: `string`
   - character_set_type: `letters`, `digits`, `symbols`

2. Character Type Count

   The number of characters in the string that are of a specific type must satisfy a rule.

   - constraint_type: `string`
   - character_set_type: all character types (including `unique` for exact count via Quant Constant; parity / multiple not applied to `unique`).
   - count_rule: `even`, `odd`, `multiple_of_N` where N ∈ {2,3,4}.
   - NON-ZERO ENFORCEMENT (current implementation):
     - `even` requires a strictly positive even count (minimum 2) unless an exact target count already sets a positive value.
     - `odd` requires a strictly positive odd count (minimum 1).
     - `multiple_of_N` requires at least one multiple group (minimum N). Zero is NOT allowed for any rule.

3. Quantitative Constant

   The number of characters of a given metric equals a specific positive integer.

   - constraint_type: `string`
   - metric: any character type + `length` + `unique`.
   - constant range (implemented): 2–5 (previous spec 1–10 was tightened).

4. Quantitative Progression

   The number of characters of a given metric follows an arithmetic progression down a row or column.

   - constraint_type: `string` + `grid` (applies across 3 cells on that axis).
   - metric: any character type + `length` + `unique`.
   - start: 1–3.
   - step: 1–3 (tightened from earlier 1–5 spec).
   - All derived values must remain within feasibility bounds (length ≤ 20, ≥ 1, etc.).

5. Sorted Characters

   All characters in the string must be sorted by a defined order.

   - constraint_type: `string`
   - sort_order: `ascending`, `descending`, `mixed`.
   - `mixed` means the string is neither fully ascending nor fully descending (post-generation shuffle).
   - Only allowed when a homogeneous character set context exists (enforced via feasibility checks). Not compatible with Positional Character Type.

6. Positional Character Type

   A character (or set of indexed positions) must be of a specific type.

   - constraint_type: `char`
   - character_set_type: all character types except `unique`.
   - index: `first`, `last`, `even`, `odd`.
   - Not compatible with Sorted Characters.

7. Unique Characters (as a metric)

   - Enforced via Quant Constant or Quant Progression using metric `unique`.
   - Interacts with length feasibility (unique count ≤ length and ≤ pool size after any character set restriction).
   - Letters with different cases (e.g., `A` vs `a`) are considered distinct for uniqueness.

---

Additional Generation Policies (Implemented):

- Variable Length (when no fixed length):

  - A minimum required length is computed: sum of exact target counts + parity minima (even=2, odd=1 if not already exact) + multiples minima (N) + unique_exact (if present).
  - If min < 10: sampled length ∈ [min, 10]; else length ∈ [min, min+2]; capped at 20.
  - Applied per cell when `fixed_length` is absent.

- Post-generation Shuffling:

  - If a cell has neither ordering nor positional constraints, its characters are shuffled after satisfying other constraints to reduce positional cues.
  - The answer cell and any such cells are shuffled under this rule.

- Distractor Generation:

  - Starts from the correct answer string and applies one mutation strategy: positional break, ordering swap, count adjustment, or single-character mutation.
  - Variable length adjustment & shuffling also applied to distractors when (a) no fixed_length, (b) no ordering, (c) no positional constraints for the answer pattern.
  - Each distractor must violate at least one constraint of the answer’s constraint set to avoid accidental correctness.

- Non-zero Enforcement:

  - Parity and multiple rules disallow zero counts (even must be ≥2, odd ≥1, multiple_of_N ≥N) unless an exact target already provides a valid positive number.

- Feasibility Checks:

  - Detect incompatible combinations (e.g., parity conflicts with exact counts, multiple conflicts, sorted + positional conflict, unique exceeding pool, progression out of range).

- Variation Constraint:
  - Rows and columns enforce a minimal Hamming distance between strings (currently ≥2 where lengths match) to avoid trivial repetition.

---

Leak Detection & Consolidation (Implemented):

- Timing: Leaks are inferred AFTER the full 3x3 grid strings are generated under the originally chosen row + column attributes.
- Scope: A candidate leak must hold for ALL 3 strings in a given row (row candidate) or ALL 3 strings in a given column (column candidate).
- Intersection: For rows, we intersect the candidate sets across the 3 rows to retain only leaks universally true for every row; same for columns (across the 3 columns). Only universally valid leaks are reported.
- Supported leak types (mirrors primary attribute set):
  - character_set_type (all characters drawn from letters / digits / symbols)
  - type*count parity / multiple (even, odd, multiple_of*{2,3,4}) for any character type with non‑zero counts
  - quant_constant (exact counts) for any metric (character types + length + unique)
  - quant_progression (arithmetic progression across the 3 cells) for any metric
  - sorted (ascending / descending / mixed) when no positional constraint present
  - positional (first / last / even / odd positions enforce a character type) when no ordering constraint present
- Overlap Simplification:
  - If multiple leak specs differ ONLY by character_type (e.g., positional vowels vs positional letters) we retain the spec whose character pool is largest (preferring the most general) and drop the rest.
  - A second pass performs the same consolidation for leaks that differ ONLY by metric (excluding length/unique) again selecting the largest underlying pool.
  - Result: Each leak appears at most once; no axis-specific duplication.
- Application:
  - Accepted leaks are applied to every cell constraint along the corresponding axis, but only when the specific constraint slot is still unset (e.g., add parity rule only if target count not already fixed and no parity rule present; add ordering only if no positional constraint; add positional only if neither positional nor ordering exists, etc.).
  - Character set leaks narrow allowed character pools by intersection (never expand beyond what the strings already use), guaranteeing the existing strings still satisfy the updated constraints (no regeneration required).
- Reporting:
  - row_leaks: list of consolidated leak specs universal to ALL rows.
  - col_leaks: list of consolidated leak specs universal to ALL columns.
  - No per-row / per-column duplication; each spec is reported once.
- Guarantees:
  - The final answer strings satisfy BOTH the originally chosen attributes AND every reported leak.
  - Distractors are generated only after leak application and must violate at least one (chosen or leaked) constraint to be accepted.

Enhanced Compatibility / Infeasibility Handling:

- Added detection for infeasible parity or multiple rules when the effective character set restriction makes a non‑zero count impossible (e.g., requesting consonant parity inside a symbols‑only character set).
- Detects MULTIPLE_INFEASIBLE alongside existing parity infeasibility flags and prevents construction of contradictory constraints.
- Prevents sorted + positional coexistence (already noted) and disallows zero counts under parity/multiple semantics.
- Ensures unique / length targets do not exceed available pool size after any character set narrowing.

Randomness / Seeding:

- External/user seed input removed; each generation run derives an internal random 64‑bit seed automatically (exposed only in debug metadata if present). This eliminates unintended reproducibility while keeping per‑item stochastic variability.

Distractors (Clarification):

- Validation uses the post‑leak (augmented) constraint set. Any candidate fully satisfying all constraints is rejected and mutated until at least one constraint violation is present (or another strategy chosen), ensuring a single unambiguous correct answer.

---

Notes / Differences from Original Spec:

- Quant Constant range narrowed to 2–5.
- Progression step narrowed to 1–3.
- Zero counts for type_count rules are explicitly disallowed (original spec allowed implicit zero for even). Updated for clarity and user experience.
- Additional post-processing (shuffling & variable sizing) introduced to reduce pattern salience purely from length or position when unconstrained.
