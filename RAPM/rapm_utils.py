import re, os, json
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from RAPM.text_rapm.per_cell_constraints import CellConstraint
from RAPM.text_rapm.validator import cell_satisfies
from shared.model_wrapper import encode_image_to_base64

# ---------------- System prompts ---------------- #
RAPM_BASE_PROMPT = (
    "You are taking the Raven's Progressive Matrices (RPM) test, a non-verbal intelligence test that measures abstract reasoning ability.\n\n"
    "You will see a 3x3 matrix of images with the bottom-right image missing (shown as a question mark), followed by 8 answer choices numbered 1-8.\n\n"
    "Your task is to: \n1. Analyze rows and columns\n2. Infer the governing logical rule(s)\n3. Select the answer choice (1-8) that correctly completes the matrix.\n\n"
)
RAPM_PATTERN_INFO = (
    "The patterns can involve: \n"
    "- Shape transformations (rotation, reflection, scaling)\n"
    "- Position changes (movement, arrangement)\n"
    "- Attribute changes (color, size, number of elements)\n"
    "- Logical operations (addition, subtraction, intersection)\n"
    "- Sequence progressions (systematic changes across rows/columns)\n\n"
)
RAPM_ADDITIONAL_RULES = (
    "Additional common rule types:\n"
    "- Constant-in-row: Same value across a row; varies down columns.\n"
    "- Quantitative step: Fixed +/− increment between adjacent cells (size / count / position offset).\n"
    "- Figure add/subtract: Combine (overlay or juxtapose) or remove elements from two cells to form the third.\n"
    "- Distribution-of-three: Three distinct categorical values appear once each per row (order may permute).\n"
    "- Distribution-of-two: Two values each appear once; third slot is empty / null.\n\n"
    "Look horizontally and vertically; the missing piece must satisfy ALL relevant row and column rules.\n\n"
)
RAPM_ANSWER_SUFFIX = "Your final answer should be a number between 1-8 corresponding to the correct choice.\n"

TEXT_RAPM_BASE_PROMPT = (
    "You are solving a TEXT-BASED 3x3 pattern matrix (Raven-style). Each cell contains a string; the bottom-right cell is missing ('?').\n\n"
    "Goal: Infer the rule(s) acting across rows and columns.\n\n"
)
TEXT_RAPM_PATTERN_INFO = (
    "Possible dimensions (one or more):\n"
    "- Character set restriction (digits / letters / symbols)\n"
    "- Quantitative constant (exact length / count / unique)\n"
    "- Quantitative progression (arithmetic step across row/column)\n"
    "- Parity / multiple rules (all even / all odd / multiples of N)\n"
    "- Positional constraints (first/last/even/odd positions restricted)\n"
    "- Ordering (ascending / descending / mixed)\n"
    "- Layered combinations (e.g. constant + parity, progression + positional)\n\n"
)
TEXT_RAPM_MODE_INSTR_MC = (
    "You will be given 8 answer options (1-8). Select the single option that correctly fills the missing cell while satisfying ALL inferred row and column constraints.\n"
    "Respond with <answer>NUMBER</answer> using just the chosen option number.\n"
)
TEXT_RAPM_MODE_INSTR_GEN = (
    "You must GENERATE the exact missing cell string that satisfies ALL inferred row and column constraints.\n"
    "Respond with <answer>STRING</answer> containing only the candidate string (no quotes or extra text).\n"
)


def build_image_system_prompt(args) -> str:
    s = RAPM_BASE_PROMPT
    if args.patterns:
        s += RAPM_PATTERN_INFO + RAPM_ADDITIONAL_RULES
    s += RAPM_ANSWER_SUFFIX
    if args.cot:
        s += f"\nExplain your thought process (max {args.think_budget} tokens) inside <think>...</think> then give final answer.\n"
    else:
        s += "\nAnswer only with your final answer.\n"
    s += "State your final answer as: <answer>number</answer>\n"
    return s


def build_text_system_prompt(args) -> str:
    s = TEXT_RAPM_BASE_PROMPT
    if args.patterns:
        s += TEXT_RAPM_PATTERN_INFO
    s += TEXT_RAPM_MODE_INSTR_MC if args.answer_mode == "mc" else TEXT_RAPM_MODE_INSTR_GEN
    if args.cot:
        s += f"Explain your thought process (max {args.think_budget} tokens) inside <think>...</think> then final answer.\n"
    else:
        s += "Answer only with your final answer.\n"
    return s


# ---------------- Formatting helpers ---------------- #

def format_text_grid(grid) -> str:
    rows = []
    for r in range(3):
        row_cells = []
        for c in range(3):
            v = grid[r][c]
            row_cells.append("?" if v is None else v)
        rows.append(" | ".join(row_cells))
    return "\n".join(rows)


def format_text_item_prompt(item: Dict, answer_mode: str) -> str:
    grid_text = format_text_grid(item["question_grid"])
    if answer_mode == "mc":
        options = item["options"]
        opt_lines = [f"{i+1}. {o}" for i, o in enumerate(options)]
        return f"Matrix:\n{grid_text}\n\nOptions:\n" + "\n".join(opt_lines) + "\n\nAnswer with <answer>N</answer>."
    return (
        f"Matrix:\n{grid_text}\n\nThe bottom-right cell should be generated. Provide only the inferred string in <answer>...</answer>."
    )


# ---------------- Parsing helpers ---------------- #

def extract_reasoning_and_answer(content: str):
    reasoning = None
    answer = None
    if content:
        think = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if not think:
            think = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        if think:
            reasoning = think.group(1).strip()
        ans = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if ans:
            answer = ans.group(1).strip()
    return reasoning, answer


def parse_image_answer(answer_txt: Optional[str]) -> Optional[int]:
    if not answer_txt:
        return None
    nums = re.findall(r"\d+", answer_txt)
    if not nums:
        return None
    try:
        n = int(nums[0])
        if 1 <= n <= 8:
            return n
    except ValueError:
        return None
    return None


def parse_text_mc(answer_txt: Optional[str]) -> Optional[int]:
    return parse_image_answer(answer_txt)


# ---------------- Constraint reconstruction ---------------- #

def reconstruct_cell_constraint(d: dict) -> CellConstraint:
    return CellConstraint(
        fixed_length=d.get("fixed_length"),
        target_counts=d.get("target_counts", {}) or {},
        parity_rules=d.get("parity_rules", {}) or {},
        multiple_rules=d.get("multiple_rules", {}) or {},
        unique_exact=d.get("unique_exact"),
        ordering=d.get("ordering"),
        positional_type=d.get("positional_type"),
        positional_index_rule=d.get("positional_index_rule"),
    )


# ---------------- Batch request builders ---------------- #

def build_image_batch_requests(args, questions, system_prompt, instruction_text: Optional[str] = None) -> List[dict]:
    base_dir = os.path.dirname(args.eval_data)
    reqs = []
    for idx, q in enumerate(questions):
        img_path = os.path.join(base_dir, q["full_image"])
        b64 = encode_image_to_base64(img_path)
        user_content: List[Dict[str, Any]] = []
        if instruction_text:
            user_content.append({"type": "text", "text": instruction_text})
        user_content.append({"type": "image_url", "image_url": {"url": b64}})
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        reqs.append(
            {
                "custom_id": f"rapm_item_{idx}",
                "messages": msgs,
                "max_completion_tokens": args.max_tokens,
            }
        )
    return reqs


def build_text_batch_requests(args, items, system_prompt, instruction_text: Optional[str] = None) -> List[dict]:
    reqs = []
    for idx, item in enumerate(items):
        prompt = format_text_item_prompt(item, args.answer_mode)
        if instruction_text:
            prompt = f"{prompt.rstrip()}\n\n{instruction_text}"
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        reqs.append(
            {
                "custom_id": f"rapm_item_{idx}",
                "messages": msgs,
                "max_completion_tokens": args.max_tokens,
            }
        )
    return reqs


# ---------------- Scoring ---------------- #

def score_image(questions, responses, args):
    results = []
    reasoning_traces = []
    correct = 0
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    for idx, q in enumerate(questions):
        content = responses.get(idx, {}).get("content") or ""
        reasoning, answer_txt = extract_reasoning_and_answer(content)
        predicted = parse_image_answer(answer_txt)
        correct_answer = q["correct_answer"]
        is_correct = predicted is not None and (predicted - 1) == correct_answer
        dt = q["dataset_type"]
        total_by_type[dt] += 1
        if is_correct:
            correct += 1
            correct_by_type[dt] += 1
        results.append(
            {
                "id": q["id"],
                "dataset_type": dt,
                "correct_answer": correct_answer,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "response": content,
                "image_path": q["full_image"],
            }
        )
        if reasoning:
            reasoning_traces.append({"question_id": q["id"], "reasoning": reasoning})
    total_questions = len(questions)
    summary = {
        "model": args.model,
        "model_source": args.model_source,
        "total_questions": total_questions,
        "total_correct": correct,
        "overall_accuracy": correct / total_questions if total_questions else 0,
        "accuracy_by_type": {
            dt: {
                "correct": correct_by_type[dt],
                "total": total_by_type[dt],
                "accuracy": correct_by_type[dt] / total_by_type[dt] if total_by_type[dt] else 0,
            }
            for dt in total_by_type
        },
    }
    return results, summary, reasoning_traces


def score_text(items, responses, args):
    results = []
    reasoning_traces = []
    total_correct = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    for idx, item in enumerate(items):
        content = responses.get(idx, {}).get("content") or ""
        reasoning, answer_txt = extract_reasoning_and_answer(content)
        predicted = None
        is_correct = False
        matches_gold = False
        constraint_valid = False
        if args.answer_mode == "mc":
            predicted = parse_text_mc(answer_txt)
            correct_index = item["correct_index"]
            is_correct = predicted is not None and (predicted - 1) == correct_index
        else:
            gold_answer = item["raw"].get("answer")
            constraint_desc = (item["raw"].get("cell_constraints") or {}).get("2,2")
            cell_constraint = reconstruct_cell_constraint(constraint_desc) if constraint_desc else None
            if answer_txt:
                gen = answer_txt.strip().strip('"')
                predicted = gen
                if cell_constraint and gen:
                    constraint_valid = cell_satisfies(gen, cell_constraint)
                matches_gold = gen == gold_answer
                is_correct = constraint_valid
            correct_index = None
        raw = item["raw"]
        cats = (
            raw.get("credited_categories")
            or raw.get("assigned_categories")
            or [raw.get("primary_category")]
            if raw.get("primary_category")
            else []
        )
        for c in cats:
            if c:
                cat_total[c] += 1
                if is_correct:
                    cat_correct[c] += 1
        if is_correct:
            total_correct += 1
        results.append(
            {
                "id": item["id"],
                "predicted_answer": predicted,
                "correct_index": correct_index,
                **({"matches_gold": matches_gold, "constraint_valid": constraint_valid} if args.answer_mode == "gen" else {}),
                "is_correct": is_correct,
                "response": content,
                "categories": cats,
            }
        )
        if reasoning:
            reasoning_traces.append({"id": item["id"], "reasoning": reasoning})
    total = len(items)
    summary = {
        "model": args.model,
        "mode": "text",
        "answer_mode": args.answer_mode,
        "total": total,
        "correct": total_correct,
        "overall_accuracy": total_correct / total if total else 0,
        "accuracy_by_category": {
            c: {
                "correct": cat_correct[c],
                "total": cat_total[c],
                "accuracy": cat_correct[c] / cat_total[c] if cat_total[c] else 0,
            }
            for c in cat_total
        },
    }
    return results, summary, reasoning_traces
