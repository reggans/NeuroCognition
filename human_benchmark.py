#!/usr/bin/env python3
"""Human-facing benchmark runner built on top of existing task logic."""

import json
import os
import re
import secrets
import shutil
import sys
import time
import html
import uuid
import base64
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote

from RAPM.rapm_evaluation import load_evaluation_data, load_text_rapm_jsonl
from RAPM.rapm_utils import (
    format_text_item_prompt,
    parse_image_answer,
    parse_text_mc,
    reconstruct_cell_constraint,
)
from RAPM.text_rapm.validator import cell_satisfies
from SWM.swm_env import SWMEnv
from WCST.wcst_env import WCSTEnv


RAPM_TEXT_DATA_PATH = os.path.join("eval_data", "text_rapm_min20_total200.jsonl")
RAPM_IMAGE_DATA_PATH = os.path.join("eval_data", "raven_subset.json")
RAPM_MAX_TEXT_QUESTIONS = 200
RAPM_MAX_IMAGE_QUESTIONS = 140
PARTICIPANT_MODE = "participant"
TASK_SETUP_CHOICES: Dict[str, List[str]] = {
    "wcst": ["text", "image+text"],
    "swm": ["text", "image+text", "image-only"],
    "rapm": ["text", "image"],
}

# Tokenized preset configs for participant links (lives while app process runs).
PRESET_STORE: Dict[str, Dict[str, Any]] = {}
RUN_COUNTERS: Dict[str, int] = {}


def _slugify_token(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "participant"


def _settings_signature(
    task: str,
    swm_boxes: int,
    swm_tokens: int,
    wcst_variant: str,
    wcst_max_trials: int,
    wcst_num_correct: int,
    wcst_bg_color: bool,
    wcst_ambiguous: str,
    rapm_n_questions: int,
) -> str:
    if task == "swm":
        return f"b{int(swm_boxes)}-t{int(swm_tokens)}"
    if task == "wcst":
        variant = _slugify_token(wcst_variant)
        ambiguous = _slugify_token(wcst_ambiguous)
        bg = "1" if wcst_bg_color else "0"
        return (
            f"v{variant}-m{int(wcst_max_trials)}"
            f"-c{int(wcst_num_correct)}-bg{bg}-a{ambiguous}"
        )
    return f"n{int(rapm_n_questions)}"


def _participant_base_id(
    participant_name: str,
    task: str,
    setup: str,
    swm_boxes: int,
    swm_tokens: int,
    wcst_variant: str,
    wcst_max_trials: int,
    wcst_num_correct: int,
    wcst_bg_color: bool,
    wcst_ambiguous: str,
    rapm_n_questions: int,
) -> str:
    name_part = _slugify_token(participant_name)
    task_part = _slugify_token(task)
    setup_part = _slugify_token(setup)
    settings_part = _settings_signature(
        task,
        swm_boxes,
        swm_tokens,
        wcst_variant,
        wcst_max_trials,
        wcst_num_correct,
        wcst_bg_color,
        wcst_ambiguous,
        rapm_n_questions,
    )
    return f"{name_part}_{task_part}_{setup_part}_{settings_part}"


def _detect_max_run_number(base_id: str, output_dir: str) -> int:
    if not output_dir or not os.path.isdir(output_dir):
        return 0

    pattern = re.compile(rf"^{re.escape(base_id)}_run(\\d+)_")
    max_run = 0
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if not match:
            continue
        try:
            run_num = int(match.group(1))
        except ValueError:
            continue
        max_run = max(max_run, run_num)
    return max_run


def _reserve_next_run(base_id: str, output_dir: str) -> int:
    existing_max = _detect_max_run_number(base_id, output_dir)
    current = max(existing_max, RUN_COUNTERS.get(base_id, 0))
    next_run = current + 1
    RUN_COUNTERS[base_id] = next_run
    return next_run


def _generate_participant_identity(
    participant_name: str,
    task: str,
    setup: str,
    swm_boxes: int,
    swm_tokens: int,
    wcst_variant: str,
    wcst_max_trials: int,
    wcst_num_correct: int,
    wcst_bg_color: bool,
    wcst_ambiguous: str,
    rapm_n_questions: int,
    output_dir: str,
) -> Tuple[str, str]:
    display_name = (participant_name or "").strip() or "participant"
    base_id = _participant_base_id(
        display_name,
        task,
        setup,
        swm_boxes,
        swm_tokens,
        wcst_variant,
        wcst_max_trials,
        wcst_num_correct,
        wcst_bg_color,
        wcst_ambiguous,
        rapm_n_questions,
    )
    run_number = _reserve_next_run(base_id, output_dir)
    participant_id = f"{base_id}_run{run_number:02d}"
    return display_name, participant_id


@dataclass
class SessionSummary:
    participant_id: str
    task: str
    mode: str
    started_at: str
    completed_at: Optional[str]
    total_steps: int
    metrics: Dict[str, Any]


def _resolve_mode(task: str, setup: str) -> Tuple[str, bool]:
    if setup == "text":
        return "text", False
    if setup == "image":
        return "image", False
    if setup == "image+text":
        return "image", False
    if setup == "image-only":
        return "image", True
    raise ValueError(f"Unknown setup: {setup}")


def _normalize_setup_for_task(task: str, setup: str) -> str:
    allowed = TASK_SETUP_CHOICES.get(task, ["text"])
    return setup if setup in allowed else allowed[0]


def _wrap_answer(answer: str) -> str:
    text = (answer or "").strip()
    if "<answer>" in text and "</answer>" in text:
        return text
    return f"<answer>{text}</answer>"


def _extract_answer_text(response: str) -> Optional[str]:
    match = re.search(r"<answer>(?s:.*?)</answer>", response)
    if match is None:
        return None
    return re.sub(r"<answer>|</answer>", "", match.group(0)).strip()


def _clean_observation(text: str) -> str:
    return re.sub(r"\[Image:\s*[^\]]+\]", "", text or "").strip()


def _resolve_rapm_image(eval_data_path: str, question: Dict[str, Any]) -> Optional[str]:
    rel = question.get("full_image") or question.get("image")
    if not rel:
        return None
    if os.path.isabs(rel):
        return rel if os.path.exists(rel) else None
    abs_path = os.path.join(os.path.dirname(eval_data_path), rel)
    return abs_path if os.path.exists(abs_path) else None


def _current_rapm_prompt(state: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    idx = state["rapm_index"]
    total = state["rapm_total"]
    mode = state["mode"]
    if mode == "image":
        q = state["rapm_questions"][idx]
        prompt = f"RAPM item {idx + 1}/{total}. Analyze the matrix image and choose the best option (1-8)."
        image_path = _resolve_rapm_image(state["rapm_eval_data"], q)
        return prompt, image_path
    item = state["rapm_items"][idx]
    prompt = f"RAPM item {idx + 1}/{total}\n\n{format_text_item_prompt(item, state['rapm_answer_mode'])}"
    return prompt, None


def _rapm_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    answered = state.get("rapm_answered", 0)
    correct = state.get("rapm_correct", 0)
    return {
        "answered": answered,
        "correct": correct,
        "accuracy": (correct / answered) if answered else 0.0,
        "total": state.get("rapm_total", 0),
    }


def _format_metrics_for_task(task: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize task metrics for participant display and exported summaries."""
    if task != "wcst":
        return metrics

    view = dict(metrics)
    s_wcst = view.get("S_wcst")
    if s_wcst is None:
        s_wcst = view.get("wcst_score")
    if s_wcst is None:
        s_wcst = 0.0

    view["S_wcst"] = s_wcst
    view["wcst_score"] = s_wcst
    view.pop("accuracy", None)
    return view


def _build_preset_config(
    participant_name: str,
    participant_id: str,
    history_enabled: bool,
    task: str,
    setup: str,
    swm_boxes: int,
    swm_tokens: int,
    wcst_variant: str,
    wcst_max_trials: int,
    wcst_num_correct: int,
    wcst_bg_color: bool,
    wcst_ambiguous: str,
    rapm_n_questions: int,
) -> Dict[str, Any]:
    return {
        "participant_name": participant_name,
        "participant_id": participant_id,
        "history_enabled": bool(history_enabled),
        "task": task,
        "setup": setup,
        "swm_boxes": int(swm_boxes),
        "swm_tokens": int(swm_tokens),
        "wcst_variant": wcst_variant,
        "wcst_max_trials": int(wcst_max_trials),
        "wcst_num_correct": int(wcst_num_correct),
        "wcst_bg_color": bool(wcst_bg_color),
        "wcst_ambiguous": wcst_ambiguous,
        "rapm_n_questions": int(rapm_n_questions),
        "created_at": datetime.utcnow().isoformat(),
    }


def _register_preset(config: Dict[str, Any]) -> str:
    token = secrets.token_urlsafe(16)
    PRESET_STORE[token] = dict(config)
    return token


def _get_preset(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    cfg = PRESET_STORE.get(token)
    return dict(cfg) if cfg else None


def _extract_query_params(request: Any) -> Dict[str, str]:
    if request is None:
        return {}

    candidates = [
        getattr(request, "query_params", None),
        getattr(getattr(request, "request", None), "query_params", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return {str(k): str(v) for k, v in candidate.items()}
        except Exception:
            pass

    req_obj = getattr(request, "request", None)
    url = getattr(req_obj, "url", None) or getattr(request, "url", None)
    if url is not None:
        raw_query = getattr(url, "query", "")
        parsed = parse_qs(raw_query)
        return {k: (v[-1] if v else "") for k, v in parsed.items()}

    return {}


def _infer_request_origin(request: Any, args: Any) -> str:
    req_obj = getattr(request, "request", request)
    headers = getattr(req_obj, "headers", {}) or {}

    scheme = headers.get("x-forwarded-proto")
    host = headers.get("x-forwarded-host") or headers.get("host")

    url = getattr(req_obj, "url", None)
    if url is not None:
        if not scheme:
            scheme = getattr(url, "scheme", None)
        if not host:
            host = getattr(url, "netloc", None)

    if scheme and host:
        return f"{scheme}://{host}"

    host_arg = getattr(args, "host", "127.0.0.1")
    if host_arg in ("0.0.0.0", "::"):
        host_arg = "127.0.0.1"
    port_arg = getattr(args, "port", 7860)
    return f"http://{host_arg}:{port_arg}"


def _build_filename(state: Dict[str, Any]) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pid = re.sub(r"[^a-zA-Z0-9_-]", "_", state["participant_id"]) or "participant"
    return f"{pid}_{state['task']}_{state['mode']}_{ts}.json"


def _session_label(state: Dict[str, Any]) -> str:
    name = (state.get("participant_name") or state.get("participant_id") or "").strip()
    if not name:
        name = "participant"
    task = str(state.get("task", "")).strip()
    setup = str(state.get("setup", "")).strip()
    return f"{name} | {task} | {setup}"


def _encode_image_data_url(image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as fh:
            raw = fh.read()
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif ext == ".gif":
            mime = "image/gif"
        elif ext == ".webp":
            mime = "image/webp"
        else:
            mime = "image/png"
        encoded = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def _render_move_history(state: Dict[str, Any]) -> str:
    entries = state.get("move_history") or []
    if not entries:
        return ""

    blocks: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        answer = html.escape(str(entry.get("answer", "")))
        feedback = html.escape(str(entry.get("feedback", "")))
        ts = html.escape(str(entry.get("timestamp", "")))
        image_data_url = entry.get("image_data_url")
        image_path = entry.get("image_path")
        image_html = ""
        if image_data_url:
            image_html = (
                f"<div style='margin-top:6px;'><img src='{image_data_url}' "
                "style='max-width:100%; border-radius:6px;'></div>"
            )
        elif image_path and os.path.exists(image_path):
            image_url = "/gradio_api/file=" + quote(str(image_path), safe="/")
            image_html = (
                f"<div style='margin-top:6px;'><img src='{image_url}' "
                "style='max-width:100%; border-radius:6px;'></div>"
            )
        blocks.append(
            "<div style='padding:10px; border:1px solid #ccc; border-radius:8px; margin-bottom:8px;'>"
            f"<div><strong>Move {idx}</strong> <span style='opacity:0.75;'>{ts}</span></div>"
            f"<div><strong>User:</strong> {answer}</div>"
            f"<div><strong>System:</strong> {feedback}</div>"
            f"{image_html}</div>"
        )
    return "".join(blocks)


def _snapshot_history_image(state: Dict[str, Any], image_path: Optional[str]) -> Optional[str]:
    if not state.get("history_enabled"):
        return None
    if not image_path or not os.path.exists(image_path):
        return None

    session_dir = state.get("history_image_dir")
    if not session_dir:
        return None

    os.makedirs(session_dir, exist_ok=True)
    idx = len(state.get("move_history") or []) + 1
    ext = os.path.splitext(str(image_path))[1] or ".png"
    dst = os.path.join(session_dir, f"move_{idx:03d}{ext}")
    try:
        shutil.copy2(str(image_path), dst)
        return dst
    except Exception:
        return None


def _cleanup_session_images(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    session_dir = state.get("history_image_dir")
    if session_dir and os.path.isdir(session_dir):
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass


def _append_move_history(
    state: Dict[str, Any], answer_text: str, feedback_text: str, image_path: Optional[str]
) -> None:
    if not state.get("history_enabled"):
        return
    entries = state.setdefault("move_history", [])
    image_data_url = None
    if image_path and os.path.exists(image_path):
        image_data_url = _encode_image_data_url(image_path)

    entries.append(
        {
            "answer": answer_text,
            "feedback": feedback_text,
            "image_path": image_path,
            "image_data_url": image_data_url,
            "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
        }
    )


def _persist_session(state: Dict[str, Any]) -> str:
    output_dir = state["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    if state["task"] == "rapm":
        final_metrics = _rapm_metrics(state)
    else:
        final_metrics = _format_metrics_for_task(
            state["task"], state["env"].get_metrics()
        )
    summary = SessionSummary(
        participant_id=state["participant_id"],
        task=state["task"],
        mode=state["mode"],
        started_at=state["started_at"],
        completed_at=datetime.utcnow().isoformat(),
        total_steps=len(state["turn_logs"]),
        metrics=final_metrics,
    )
    payload = {
        "config": state["config"],
        "summary": summary.__dict__,
        "turn_logs": state["turn_logs"],
    }
    out_path = os.path.join(output_dir, _build_filename(state))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


def _start_session(
    participant_name: str,
    participant_id: str,
    history_enabled: bool,
    task: str,
    setup: str,
    swm_boxes: int,
    swm_tokens: int,
    wcst_variant: str,
    wcst_max_trials: int,
    wcst_num_correct: int,
    wcst_bg_color: bool,
    wcst_ambiguous: str,
    rapm_n_questions: int,
    output_dir: str,
) -> Tuple[Dict[str, Any], str, Optional[str], str, Dict[str, Any], str]:
    participant_name = participant_name.strip() or "participant"
    participant_id = participant_id.strip() or "participant"
    session_id = uuid.uuid4().hex
    env_mode, image_only = _resolve_mode(task, setup)
    state: Dict[str, Any] = {
        "participant_name": participant_name,
        "participant_id": participant_id,
        "session_id": session_id,
        "history_enabled": bool(history_enabled),
        "task": task,
        "setup": setup,
        "mode": env_mode,
        "image_only": image_only,
        "output_dir": output_dir,
        "started_at": datetime.utcnow().isoformat(),
        "turn_logs": [],
        "done": False,
        "prompt_shown_at": time.time(),
        "history_image_dir": os.path.join(output_dir, ".session_images", session_id),
        "move_history": [],
        "config": {
            "participant_name": participant_name,
            "participant_id": participant_id,
            "session_id": session_id,
            "history_enabled": bool(history_enabled),
            "task": task,
            "setup": setup,
            "mode": env_mode,
            "image_only": image_only,
            "swm_boxes": swm_boxes,
            "swm_tokens": swm_tokens,
            "wcst_variant": wcst_variant,
            "wcst_max_trials": wcst_max_trials,
            "wcst_num_correct": wcst_num_correct,
            "wcst_bg_color": wcst_bg_color,
            "wcst_ambiguous": wcst_ambiguous,
            "rapm_n_questions": rapm_n_questions,
            "rapm_text_data": RAPM_TEXT_DATA_PATH,
            "rapm_image_data": RAPM_IMAGE_DATA_PATH,
        },
    }

    if task == "swm":
        env = SWMEnv(
            n_boxes=swm_boxes,
            n_tokens=swm_tokens,
            mode=env_mode,
            image_path=os.path.join("SWM", "images"),
            image_only=image_only,
        )
        observation = env.reset()
        state["env"] = env
        image_path = env.get_current_image_path() if env_mode == "image" else None
        metrics = _format_metrics_for_task(task, env.get_metrics())
        visible_obs = "" if image_only else _clean_observation(observation)
        return state, visible_obs, image_path, "Session started.", metrics, "Running"

    if task == "wcst":
        env = WCSTEnv(
            variant=wcst_variant,
            max_trials=wcst_max_trials,
            num_correct=wcst_num_correct,
            bg_color=wcst_bg_color,
            ambiguous_mode=wcst_ambiguous,
            image_mode=(env_mode == "image"),
            image_path=os.path.join("WCST", "images"),
            image_only=image_only,
        )
        observation = env.reset()
        state["env"] = env
        image_path = env.get_current_image_path() if env_mode == "image" else None
        metrics = _format_metrics_for_task(task, env.get_metrics())
        visible_obs = "" if image_only else _clean_observation(observation)
        return state, visible_obs, image_path, "Session started.", metrics, "Running"

    if task == "rapm" and image_only:
        return (
            state,
            "",
            None,
            "RAPM image-only is not supported in this runner yet.",
            {},
            "Error",
        )

    state["rapm_answer_mode"] = "mc"
    state["rapm_index"] = 0
    state["rapm_answered"] = 0
    state["rapm_correct"] = 0

    n_questions = max(1, int(rapm_n_questions))

    if env_mode == "image":
        if not os.path.exists(RAPM_IMAGE_DATA_PATH):
            return (
                state,
                "",
                None,
                f"RAPM image data not found: {RAPM_IMAGE_DATA_PATH}",
                {},
                "Error",
            )
        state["rapm_eval_data"] = RAPM_IMAGE_DATA_PATH
        questions = load_evaluation_data(RAPM_IMAGE_DATA_PATH)
        limit = min(n_questions, RAPM_MAX_IMAGE_QUESTIONS, len(questions))
        state["rapm_questions"] = questions[:limit]
        state["rapm_total"] = len(state["rapm_questions"])
    else:
        if not os.path.exists(RAPM_TEXT_DATA_PATH):
            return (
                state,
                "",
                None,
                f"RAPM text data not found: {RAPM_TEXT_DATA_PATH}",
                {},
                "Error",
            )
        state["rapm_eval_data"] = RAPM_TEXT_DATA_PATH
        items = load_text_rapm_jsonl(RAPM_TEXT_DATA_PATH)
        limit = min(n_questions, RAPM_MAX_TEXT_QUESTIONS, len(items))
        state["rapm_items"] = items[:limit]
        state["rapm_total"] = len(state["rapm_items"])

    if state["rapm_total"] == 0:
        return state, "", None, "RAPM dataset is empty.", {}, "Error"

    observation, image_path = _current_rapm_prompt(state)
    return (
        state,
        observation,
        image_path,
        "Session started.",
        _rapm_metrics(state),
        "Running",
    )


def _step_rapm(
    state: Dict[str, Any], wrapped_answer: str, response_time_s: float
) -> Tuple[str, Optional[str], str, Dict[str, Any], str]:
    idx = state["rapm_index"]
    mode = state["mode"]
    answer_mode = state["rapm_answer_mode"]
    answer_text = _extract_answer_text(wrapped_answer)
    is_correct = False
    expected: Any = None
    parsed: Any = None

    if mode == "image":
        q = state["rapm_questions"][idx]
        parsed = parse_image_answer(answer_text)
        expected = int(q["correct_answer"]) + 1
        is_correct = parsed is not None and parsed == expected
        item_id = q.get("id", f"image_{idx}")
    else:
        item = state["rapm_items"][idx]
        item_id = item.get("id", f"text_{idx}")
        if answer_mode == "mc":
            parsed = parse_text_mc(answer_text)
            expected = int(item["correct_index"]) + 1
            is_correct = parsed is not None and parsed == expected
        else:
            parsed = answer_text
            expected = item["raw"].get("answer")
            constraint_desc = (item["raw"].get("cell_constraints") or {}).get("2,2")
            if parsed and constraint_desc:
                constraint = reconstruct_cell_constraint(constraint_desc)
                is_correct = cell_satisfies(parsed, constraint)
            else:
                is_correct = parsed == expected

    state["rapm_answered"] += 1
    if is_correct:
        state["rapm_correct"] += 1

    state["turn_logs"].append(
        {
            "step": state["rapm_answered"],
            "item_id": item_id,
            "raw_answer": wrapped_answer,
            "parsed_answer": parsed,
            "expected": expected,
            "is_correct": is_correct,
            "response_time_s": response_time_s,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    state["rapm_index"] += 1
    done = state["rapm_index"] >= state["rapm_total"]
    state["done"] = done

    feedback = "Correct." if is_correct else f"Incorrect. Expected: {expected}"
    metrics = _rapm_metrics(state)

    if done:
        out_path = _persist_session(state)
        return (
            "Session complete.",
            None,
            f"{feedback} Results saved: {out_path}",
            metrics,
            "Completed",
        )

    next_observation, next_image_path = _current_rapm_prompt(state)
    state["prompt_shown_at"] = time.time()
    return next_observation, next_image_path, feedback, metrics, "Running"


def _submit_answer(
    answer: str, state: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], str, Optional[str], str, Dict[str, Any], str, str]:
    if not state or state.get("done"):
        return (
            state or {},
            "",
            None,
            "No active session. Start a session first.",
            {},
            "Idle",
            "",
        )

    wrapped = _wrap_answer(answer)
    dt = max(0.0, time.time() - float(state.get("prompt_shown_at", time.time())))

    task = state["task"]
    if task == "rapm":
        obs, img, feedback, metrics, status = _step_rapm(state, wrapped, dt)
        return state, obs, img, feedback, metrics, status, ""

    env = state["env"]
    step = env.step(wrapped)
    image_path = env.get_current_image_path() if state["mode"] == "image" else None
    metrics = _format_metrics_for_task(task, env.get_metrics())
    feedback = f"Status: {step.info.get('status')} | Reward: {step.reward}"

    state["turn_logs"].append(
        {
            "step": len(state["turn_logs"]) + 1,
            "raw_answer": wrapped,
            "status": step.info.get("status"),
            "reward": step.reward,
            "info": step.info,
            "response_time_s": dt,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    if step.done:
        state["done"] = True
        out_path = _persist_session(state)
        return (
            state,
            "Session complete.",
            image_path,
            f"{feedback}. Results saved: {out_path}",
            metrics,
            "Completed",
            "",
        )

    state["prompt_shown_at"] = time.time()
    next_obs = "" if state.get("image_only") else _clean_observation(step.observation)
    return state, next_obs, image_path, feedback, metrics, "Running", ""


def _end_session(
    state: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str, Optional[str], str, Dict[str, Any], str]:
    if not state:
        return {}, "", None, "No session to stop.", {}, "Idle"
    if not state.get("done") and state.get("turn_logs"):
        out_path = _persist_session(state)
        status = f"Stopped and saved partial session: {out_path}"
    else:
        status = "Session closed."
    state["done"] = True
    return state, "", None, status, {}, "Idle"


def _task_instructions(task: str) -> str:
    """Return task instructions for participant."""
    instrs = {
        "wcst": """**Wisconsin Card Sorting Test (WCST)**

You will be shown a given card and four option cards. Your task is to match the given card to one of the four options according to an attribute you must figure out.

**Attributes:**
- Number of symbols (one, two, three, four)
- Color (red, green, blue, yellow)
- Shape (circle, triangle, star, square)
- Background color (only if enabled by the test administrator)

**How it works:**
1. The system will tell you "Correct!" or "Incorrect. Please try again."
2. If incorrect, either you made a mistake OR the rule has changed.
3. If you think you made a mistake, adjust and try again.
4. If you think the rule changed, infer the new rule from feedback.
5. Your answer should be a single number: 1, 2, 3, or 4.
""",
        "swm": """**Spatial Working Memory (SWM) Test**

Your task is to find token types hidden in boxes.

Depending on the setup:
- In **text** setup, you will see numbered boxes and answer with a box number.
- In **image+text** or **image-only** setup, you will see a grid and answer with coordinates.

**How it works:**
1. There can be one or multiple token types, depending on the setup.
2. A box may contain multiple token types, but each token type can appear at most once in that box.
3. Once a token type is found in a box, that same token type will never appear in that box again.
4. You must use feedback from previous choices to guide your next choice.

**Answering:**
- In **text** setup: enter a single box number (e.g., 3).
- In **image+text/image-only** setup: enter coordinates in the format row,col (e.g., "0,1").
- For coordinates, rows and columns are **0-indexed** (starting from 0).
""",
        "rapm": """**Raven's Advanced Progressive Matrices (RAPM)**

You will be shown matrix puzzles. Each puzzle shows a pattern with one cell missing.
Your task is to find the pattern and select the correct option that completes the matrix.

**How it works:**
1. Analyze the matrix image to identify the pattern.
2. Choose the option number (1-8) that best completes the pattern.
3. After you submit an answer, the test automatically moves to the next question.
4. There are no correct answers provided; you must find the pattern yourself.
""",
    }
    return instrs.get(task, "")


def launch_human_benchmark(args: Any) -> None:
    import gradio as gr

    with gr.Blocks(title="NeuroCognition Human Benchmark") as demo:
        gr.Markdown("# NeuroCognition Human Benchmark")
        session_state = gr.State(value={})
        output_dir_state = gr.State(value=getattr(args, "output_dir", "human_data"))

        with gr.Tabs() as tab_group:
            # ========== SETUP TAB ==========
            with gr.Tab("Setup", id="setup") as setup_tab:
                gr.Markdown("### Enter Your Information & Select Task")

                with gr.Group():
                    participant_name = gr.Textbox(
                        label="Participant Name",
                        placeholder="e.g., faiz",
                        value="participant",
                    )

                with gr.Group():
                    task = gr.Dropdown(
                        choices=["wcst", "swm", "rapm"],
                        value="wcst",
                        label="Select Task",
                    )
                    setup_type = gr.Dropdown(
                        choices=["text", "image+text"],
                        value="text",
                        label="Setup Type",
                        allow_custom_value=True,
                    )

                # Task-specific public settings
                with gr.Group():
                    gr.Markdown("#### Task Settings")
                    with gr.Group(visible=True) as wcst_task_group:
                        wcst_variant_setup = gr.Dropdown(
                            choices=["card", "card-random", "string", "empty"],
                            value="card",
                            label="WCST: Variant",
                        )
                    with gr.Group(visible=False) as swm_task_group:
                        swm_boxes_setup = gr.Slider(
                            minimum=4,
                            maximum=12,
                            step=1,
                            value=8,
                            label="SWM: Number of Boxes",
                        )
                    with gr.Group(visible=False) as rapm_task_group:
                        gr.Markdown(
                            "RAPM uses fixed datasets: text from eval_data/text_rapm_min20_total200.jsonl and image from eval_data/raven_subset.json (images in eval_data/images)."
                        )
                        rapm_n_questions_setup = gr.Slider(
                            minimum=1,
                            maximum=RAPM_MAX_TEXT_QUESTIONS,
                            step=1,
                            value=20,
                            label="RAPM: Number of Questions",
                        )

                # Researcher-only settings (hidden by default)
                with gr.Group(visible=False) as researcher_group:
                    gr.Markdown("#### Researcher Settings (Advanced)")
                    with gr.Group(visible=True) as wcst_research_group:
                        wcst_max_trials_adv = gr.Slider(
                            minimum=16,
                            maximum=128,
                            step=1,
                            value=64,
                            label="WCST Max Trials",
                        )
                        wcst_num_correct_adv = gr.Slider(
                            minimum=3,
                            maximum=8,
                            step=1,
                            value=5,
                            label="WCST Num Correct",
                        )
                        wcst_bg_color_adv = gr.Checkbox(
                            value=False, label="WCST Background Color"
                        )
                        wcst_ambiguous_adv = gr.Dropdown(
                            choices=["off", "first", "rest"],
                            value="off",
                            label="WCST Ambiguous",
                        )

                    with gr.Group(visible=False) as swm_research_group:
                        swm_tokens_adv = gr.Slider(
                            minimum=1, maximum=4, step=1, value=1, label="SWM Tokens"
                        )

                    with gr.Group(visible=False) as rapm_research_group:
                        gr.Markdown(
                            f"Fixed RAPM datasets:\n- Text: {RAPM_TEXT_DATA_PATH}\n- Image: {RAPM_IMAGE_DATA_PATH}"
                        )

                researcher_toggle = gr.Checkbox(
                    value=False, label="Show Researcher Settings", interactive=True
                )

                history_enabled_toggle = gr.Checkbox(
                    value=False,
                    label="Enable Move History Panel",
                    info="If enabled, participants can view their past moves and feedback during the test.",
                    interactive=True,
                )

                # Task description
                task_description = gr.Markdown(_task_instructions("wcst"))

                def update_description(t):
                    return gr.update(value=_task_instructions(t))

                task.change(
                    fn=update_description, inputs=[task], outputs=[task_description]
                )

                # Visibility logic for task-specific groups and setup choices.
                def update_task_controls(t, show_researcher):
                    if t == "wcst":
                        setup = gr.update(
                            choices=TASK_SETUP_CHOICES["wcst"], value="text"
                        )
                        researcher_panel = gr.update(visible=show_researcher)
                        wcst_task = gr.update(visible=True)
                        swm_task = gr.update(visible=False)
                        rapm_task = gr.update(visible=False)
                        wcst_research = gr.update(visible=show_researcher)
                        swm_research = gr.update(visible=False)
                        rapm_research = gr.update(visible=False)
                    elif t == "swm":
                        setup = gr.update(choices=TASK_SETUP_CHOICES["swm"], value="text")
                        researcher_panel = gr.update(visible=show_researcher)
                        wcst_task = gr.update(visible=False)
                        swm_task = gr.update(visible=True)
                        rapm_task = gr.update(visible=False)
                        wcst_research = gr.update(visible=False)
                        swm_research = gr.update(visible=show_researcher)
                        rapm_research = gr.update(visible=False)
                    else:
                        setup = gr.update(choices=TASK_SETUP_CHOICES["rapm"], value="text")
                        researcher_panel = gr.update(visible=False)
                        wcst_task = gr.update(visible=False)
                        swm_task = gr.update(visible=False)
                        rapm_task = gr.update(visible=True)
                        wcst_research = gr.update(visible=False)
                        swm_research = gr.update(visible=False)
                        rapm_research = gr.update(visible=False)

                    return (
                        setup,
                        wcst_task,
                        swm_task,
                        rapm_task,
                        researcher_panel,
                        wcst_research,
                        swm_research,
                        rapm_research,
                    )

                def update_rapm_question_limit(tsk, setup, current_value):
                    if tsk != "rapm":
                        return gr.update()
                    if setup == "image":
                        max_q = RAPM_MAX_IMAGE_QUESTIONS
                    else:
                        max_q = RAPM_MAX_TEXT_QUESTIONS
                    value = int(current_value) if current_value is not None else max_q
                    value = max(1, min(value, max_q))
                    return gr.update(maximum=max_q, value=value)

                task.change(
                    fn=update_task_controls,
                    inputs=[task, researcher_toggle],
                    outputs=[
                        setup_type,
                        wcst_task_group,
                        swm_task_group,
                        rapm_task_group,
                        researcher_group,
                        wcst_research_group,
                        swm_research_group,
                        rapm_research_group,
                    ],
                    queue=False,
                )

                researcher_toggle.change(
                    fn=update_task_controls,
                    inputs=[task, researcher_toggle],
                    outputs=[
                        setup_type,
                        wcst_task_group,
                        swm_task_group,
                        rapm_task_group,
                        researcher_group,
                        wcst_research_group,
                        swm_research_group,
                        rapm_research_group,
                    ],
                    queue=False,
                )

                setup_type.change(
                    fn=update_rapm_question_limit,
                    inputs=[task, setup_type, rapm_n_questions_setup],
                    outputs=[rapm_n_questions_setup],
                    queue=False,
                )

                status_setup = gr.Textbox(
                    label="Status", interactive=False, value="Ready"
                )

                generate_link_btn = gr.Button(
                    "Generate Participant Link", variant="secondary"
                )
                participant_link_box = gr.Textbox(
                    label="Participant Link",
                    interactive=False,
                    visible=False,
                    placeholder="Generate link after setting task and options.",
                )

                def start_task(
                    name,
                    tsk,
                    setup,
                    history_enabled,
                    swm_boxes,
                    swm_tokens,
                    wcst_var,
                    wcst_max,
                    wcst_num,
                    wcst_bg,
                    wcst_amb,
                    rapm_n_questions,
                    out_dir,
                ):
                    try:
                        setup = _normalize_setup_for_task(tsk, setup)
                        resolved_name, resolved_id = _generate_participant_identity(
                            participant_name=name,
                            task=tsk,
                            setup=setup,
                            swm_boxes=swm_boxes,
                            swm_tokens=swm_tokens,
                            wcst_variant=wcst_var,
                            wcst_max_trials=wcst_max,
                            wcst_num_correct=wcst_num,
                            wcst_bg_color=wcst_bg,
                            wcst_ambiguous=wcst_amb,
                            rapm_n_questions=rapm_n_questions,
                            output_dir=out_dir,
                        )
                        st, obs, img, fb, met, stat = _start_session(
                            resolved_name,
                            resolved_id,
                            history_enabled,
                            tsk,
                            setup,
                            swm_boxes,
                            swm_tokens,
                            wcst_var,
                            wcst_max,
                            wcst_num,
                            wcst_bg,
                            wcst_amb,
                            rapm_n_questions,
                            out_dir,
                        )
                        # Store observation and image in state for sync to test tab
                        st["_initial_observation"] = obs
                        st["_initial_image"] = img
                        st["_current_observation"] = obs
                        st["_current_image"] = img
                        return st, stat
                    except Exception as e:
                        return {}, f"Error: {str(e)}"

                def generate_participant_link(
                    name,
                    tsk,
                    setup,
                    history_enabled,
                    swm_boxes,
                    swm_tokens,
                    wcst_var,
                    wcst_max,
                    wcst_num,
                    wcst_bg,
                    wcst_amb,
                    rapm_n_questions,
                    out_dir,
                    request: gr.Request,
                ):
                    setup = _normalize_setup_for_task(tsk, setup)
                    resolved_name, resolved_id = _generate_participant_identity(
                        participant_name=name,
                        task=tsk,
                        setup=setup,
                        swm_boxes=swm_boxes,
                        swm_tokens=swm_tokens,
                        wcst_variant=wcst_var,
                        wcst_max_trials=wcst_max,
                        wcst_num_correct=wcst_num,
                        wcst_bg_color=wcst_bg,
                        wcst_ambiguous=wcst_amb,
                        rapm_n_questions=rapm_n_questions,
                        output_dir=out_dir,
                    )
                    cfg = _build_preset_config(
                        participant_name=resolved_name,
                        participant_id=resolved_id,
                        history_enabled=history_enabled,
                        task=tsk,
                        setup=setup,
                        swm_boxes=swm_boxes,
                        swm_tokens=swm_tokens,
                        wcst_variant=wcst_var,
                        wcst_max_trials=wcst_max,
                        wcst_num_correct=wcst_num,
                        wcst_bg_color=wcst_bg,
                        wcst_ambiguous=wcst_amb,
                        rapm_n_questions=rapm_n_questions,
                    )
                    token = _register_preset(cfg)
                    origin = _infer_request_origin(request, args)
                    link = f"{origin}/?mode={PARTICIPANT_MODE}&token={token}"
                    return gr.update(value=link, visible=True)

                start_btn = gr.Button("Start Test", variant="primary", size="lg")
                start_inputs = [
                    participant_name,
                    task,
                    setup_type,
                    history_enabled_toggle,
                    swm_boxes_setup,
                    swm_tokens_adv,
                    wcst_variant_setup,
                    wcst_max_trials_adv,
                    wcst_num_correct_adv,
                    wcst_bg_color_adv,
                    wcst_ambiguous_adv,
                    rapm_n_questions_setup,
                    output_dir_state,
                ]

                generate_link_btn.click(
                    fn=generate_participant_link,
                    inputs=[
                        participant_name,
                        task,
                        setup_type,
                        history_enabled_toggle,
                        swm_boxes_setup,
                        swm_tokens_adv,
                        wcst_variant_setup,
                        wcst_max_trials_adv,
                        wcst_num_correct_adv,
                        wcst_bg_color_adv,
                        wcst_ambiguous_adv,
                        rapm_n_questions_setup,
                        output_dir_state,
                    ],
                    outputs=[participant_link_box],
                    queue=False,
                )

            # ========== TEST TAB ==========
            with gr.Tab("Test", id="test") as test_tab:
                gr.Markdown("### Complete the Task")

                session_info = gr.Textbox(label="Session", interactive=False, value="")

                observation = gr.Textbox(
                    label="Prompt", lines=8, interactive=False, value=""
                )
                stimulus_image = gr.Image(
                    label="Image", type="filepath", interactive=False
                )

                answer = gr.Textbox(
                    label="Your Answer",
                    lines=1,
                    placeholder="Enter your answer here",
                    interactive=True,
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "Submit Answer", variant="primary", size="lg"
                    )
                    stop_btn = gr.Button("End Session")

                feedback = gr.Textbox(
                    label="Feedback", interactive=False, value="", visible=False
                )
                metrics = gr.JSON(label="Progress", visible=False)
                move_history_panel = gr.HTML(value="", visible=False)
                status = gr.Textbox(label="Status", interactive=False, value="Idle")

                def _history_panel_update(state):
                    if not state or not state.get("history_enabled"):
                        return gr.update(value="", visible=False)
                    return gr.update(value=_render_move_history(state), visible=True)

                def on_submit(ans, state):
                    if not state or state.get("done"):
                        return (
                            state or {},
                            "",
                            None,
                            gr.update(value="", visible=False),
                            gr.update(value={}, visible=False),
                            "Idle",
                            gr.update(value="", visible=False),
                            "",
                        )

                    if not (ans or "").strip():
                        current_obs = state.get("_current_observation", "")
                        current_img = state.get("_current_image")
                        return (
                            state,
                            current_obs,
                            current_img,
                            gr.update(value="", visible=False),
                            gr.update(value={}, visible=False),
                            "Running",
                            _history_panel_update(state),
                            "",
                        )

                    wrapped = _wrap_answer(ans)
                    dt = max(
                        0.0,
                        time.time() - float(state.get("prompt_shown_at", time.time())),
                    )
                    task_name = state.get("task", "")
                    current_image_before_step = state.get("_current_image")

                    if task_name == "rapm":
                        obs, img, fb, met, stat = _step_rapm(state, wrapped, dt)
                        _append_move_history(
                            state,
                            answer_text=(ans or "").strip(),
                            feedback_text=fb,
                            image_path=_snapshot_history_image(
                                state, current_image_before_step
                            ),
                        )
                        state["_current_observation"] = obs
                        state["_current_image"] = img
                        if state.get("done"):
                            _cleanup_session_images(state)
                            return (
                                state,
                                "",
                                img,
                                gr.update(value="", visible=False),
                                gr.update(value=met, visible=False),
                                "Completed",
                                _history_panel_update(state),
                                "",
                            )
                        state["prompt_shown_at"] = time.time()
                        return (
                            state,
                            obs,
                            img,
                            gr.update(value="", visible=False),
                            gr.update(value=met, visible=False),
                            stat,
                            _history_panel_update(state),
                            "",
                        )
                    else:
                        env = state.get("env")
                        if not env:
                            return (
                                state,
                                "",
                                None,
                                gr.update(value="", visible=False),
                                gr.update(value={}, visible=False),
                                "Error",
                                _history_panel_update(state),
                                "",
                            )

                        wcst_pre_step_history_image = None
                        if task_name == "wcst":
                            wcst_pre_step_history_image = _snapshot_history_image(
                                state, current_image_before_step
                            )

                        step = env.step(wrapped)
                        image_path = (
                            env.get_current_image_path()
                            if state.get("mode") == "image"
                            else None
                        )
                        met = env.get_metrics()
                        met = _format_metrics_for_task(task_name, met)
                        obs_feedback = _clean_observation(step.observation)
                        if obs_feedback:
                            fb = obs_feedback
                        else:
                            status_txt = str(step.info.get("status") or "").strip()
                            fb = status_txt or "Result recorded."

                        # WCST history should reflect the pre-step card, while
                        # SWM history should reflect the post-step board state.
                        if task_name == "swm":
                            history_image_path = _snapshot_history_image(state, image_path)
                        elif task_name == "wcst":
                            history_image_path = wcst_pre_step_history_image
                        else:
                            history_image_path = _snapshot_history_image(
                                state, current_image_before_step
                            )

                        _append_move_history(
                            state,
                            answer_text=(ans or "").strip(),
                            feedback_text=fb,
                            image_path=history_image_path,
                        )

                        state["turn_logs"].append(
                            {
                                "step": len(state["turn_logs"]) + 1,
                                "raw_answer": wrapped,
                                "status": step.info.get("status"),
                                "reward": step.reward,
                                "info": step.info,
                                "response_time_s": dt,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

                        if step.done:
                            state["done"] = True
                            out_path = _persist_session(state)
                            state["_current_observation"] = ""
                            state["_current_image"] = image_path
                            _cleanup_session_images(state)
                            return (
                                state,
                                "",
                                image_path,
                                gr.update(value="", visible=False),
                                gr.update(value=met, visible=False),
                                "Completed",
                                _history_panel_update(state),
                                "",
                            )

                        state["prompt_shown_at"] = time.time()
                        next_obs = (
                            ""
                            if state.get("image_only")
                            else _clean_observation(step.observation)
                        )
                        state["_current_observation"] = next_obs
                        state["_current_image"] = image_path
                        return (
                            state,
                            next_obs,
                            image_path,
                            gr.update(value="", visible=False),
                            gr.update(value=met, visible=False),
                            "Running",
                            _history_panel_update(state),
                            "",
                        )

                submit_btn.click(
                    fn=on_submit,
                    inputs=[answer, session_state],
                    outputs=[
                        session_state,
                        observation,
                        stimulus_image,
                        feedback,
                        metrics,
                        status,
                        move_history_panel,
                        answer,
                    ],
                    queue=False,
                )
                answer.submit(
                    fn=on_submit,
                    inputs=[answer, session_state],
                    outputs=[
                        session_state,
                        observation,
                        stimulus_image,
                        feedback,
                        metrics,
                        status,
                        move_history_panel,
                        answer,
                    ],
                    queue=False,
                )

                def on_end(state):
                    if not state or not state.get("participant_id"):
                        return (
                            {},
                            "",
                            None,
                            gr.update(value="", visible=False),
                            gr.update(value={}, visible=False),
                            "Idle",
                            gr.update(value="", visible=False),
                        )
                    if not state.get("done") and state.get("turn_logs"):
                        out_path = _persist_session(state)
                        status_msg = f"Session saved: {out_path}"
                    else:
                        status_msg = "Session closed."
                    state["done"] = True
                    _cleanup_session_images(state)
                    return (
                        state,
                        "",
                        None,
                        gr.update(value="", visible=False),
                        gr.update(value={}, visible=False),
                        "Idle",
                        _history_panel_update(state),
                    )

                stop_btn.click(
                    fn=on_end,
                    inputs=[session_state],
                    outputs=[
                        session_state,
                        observation,
                        stimulus_image,
                        feedback,
                        metrics,
                        status,
                        move_history_panel,
                    ],
                    queue=False,
                )

                # Sync session state on load/change
                def sync_session(state):
                    if not state or not state.get("participant_id"):
                        return (
                            "",
                            "",
                            None,
                            gr.update(value="", visible=False),
                            gr.update(value={}, visible=False),
                            "Idle",
                            gr.update(value="", visible=False),
                        )

                    info = _session_label(state)
                    obs = state.get(
                        "_current_observation", state.get("_initial_observation", "")
                    )
                    img = state.get("_current_image", state.get("_initial_image"))
                    if state.get("task") == "rapm":
                        current_metrics = _rapm_metrics(state)
                    else:
                        env = state.get("env")
                        current_metrics = (
                            _format_metrics_for_task(
                                state.get("task", ""), env.get_metrics()
                            )
                            if env
                            else {}
                        )
                    return (
                        info,
                        obs,
                        img,
                        gr.update(value="", visible=False),
                        gr.update(value=current_metrics, visible=False),
                        "Running",
                        _history_panel_update(state),
                    )

                start_btn.click(
                    fn=start_task,
                    inputs=start_inputs,
                    outputs=[session_state, status_setup],
                    queue=False,
                ).then(
                    fn=sync_session,
                    inputs=[session_state],
                    outputs=[
                        session_info,
                        observation,
                        stimulus_image,
                        feedback,
                        metrics,
                        status,
                        move_history_panel,
                    ],
                    queue=False,
                )

                session_state.change(
                    fn=sync_session,
                    inputs=[session_state],
                    outputs=[
                        session_info,
                        observation,
                        stimulus_image,
                        feedback,
                        metrics,
                        status,
                        move_history_panel,
                    ],
                    queue=False,
                )

                def initialize_from_link(out_dir, request: gr.Request):
                    params = _extract_query_params(request)
                    mode = (params.get("mode") or "").strip().lower()
                    token = (params.get("token") or "").strip()

                    # Default: researcher mode, no auto-start.
                    if mode != PARTICIPANT_MODE:
                        return (
                            {},
                            "Ready",
                            "",
                            "",
                            None,
                            gr.update(value="", visible=False),
                            gr.update(value={}, visible=False),
                            "Idle",
                            gr.update(interactive=True),
                            gr.update(interactive=True),
                            gr.update(interactive=True),
                            gr.update(value=False, interactive=True),
                            gr.update(interactive=True),
                            gr.update(interactive=True),
                            gr.update(interactive=True),
                            gr.update(value=False, visible=True, interactive=True),
                            gr.update(visible=False),
                            gr.update(visible=True, interactive=True),
                            gr.update(visible=True, interactive=True),
                            gr.update(visible=False),
                            gr.update(value="", visible=False),
                            gr.update(),
                            gr.update(selected="setup"),
                        )

                    cfg = _get_preset(token)
                    if cfg is None:
                        msg = "Invalid participant link. Ask the researcher for a new link."
                        return (
                            {},
                            "Participant mode (invalid link)",
                            "",
                            "",
                            None,
                            gr.update(value="", visible=False),
                            gr.update(value={}, visible=False),
                            "Error",
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(value=False, interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(visible=False, interactive=False),
                            gr.update(visible=False),
                            gr.update(visible=False, interactive=False),
                            gr.update(visible=False, interactive=False),
                            gr.update(visible=False),
                            gr.update(value="", visible=False),
                            gr.update(),
                            gr.update(selected="setup"),
                        )

                    st, obs, img, _fb, met, stat = _start_session(
                        cfg.get("participant_name", cfg["participant_id"]),
                        cfg["participant_id"],
                        cfg.get("history_enabled", False),
                        cfg["task"],
                        cfg["setup"],
                        cfg["swm_boxes"],
                        cfg["swm_tokens"],
                        cfg["wcst_variant"],
                        cfg["wcst_max_trials"],
                        cfg["wcst_num_correct"],
                        cfg["wcst_bg_color"],
                        cfg["wcst_ambiguous"],
                        cfg["rapm_n_questions"],
                        out_dir,
                    )

                    st["_initial_observation"] = obs
                    st["_initial_image"] = img
                    st["_current_observation"] = obs
                    st["_current_image"] = img

                    if stat != "Running":
                        return (
                            st,
                            "Participant mode (failed to start)",
                            "",
                            "",
                            None,
                            gr.update(value="", visible=False),
                            gr.update(value=met, visible=False),
                            "Error",
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(
                                value=cfg.get("history_enabled", False),
                                interactive=False,
                            ),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(visible=False, interactive=False),
                            gr.update(visible=False),
                            gr.update(visible=False, interactive=False),
                            gr.update(visible=False, interactive=False),
                            gr.update(visible=False),
                            _history_panel_update(st),
                            gr.update(value=_task_instructions(cfg["task"])),
                            gr.update(selected="setup"),
                        )

                    info = _session_label(st)
                    return (
                        st,
                        "Participant mode (locked)",
                        info,
                        obs,
                        img,
                        gr.update(value="", visible=False),
                        gr.update(value=met, visible=False),
                        "Running",
                        gr.update(
                            value=cfg.get("participant_name", cfg["participant_id"]),
                            interactive=False,
                        ),
                        gr.update(value=cfg["task"], interactive=False),
                        gr.update(value=cfg["setup"], interactive=False),
                        gr.update(
                            value=cfg.get("history_enabled", False),
                            interactive=False,
                        ),
                        gr.update(value=cfg["swm_boxes"], interactive=False),
                        gr.update(value=cfg["wcst_variant"], interactive=False),
                        gr.update(value=cfg["rapm_n_questions"], interactive=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(visible=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(visible=False),
                        _history_panel_update(st),
                        gr.update(value=_task_instructions(cfg["task"])),
                        gr.update(selected="test"),
                    )

                demo.load(
                    fn=initialize_from_link,
                    inputs=[output_dir_state],
                    outputs=[
                        session_state,
                        status_setup,
                        session_info,
                        observation,
                        stimulus_image,
                        feedback,
                        metrics,
                        status,
                        participant_name,
                        task,
                        setup_type,
                        history_enabled_toggle,
                        swm_boxes_setup,
                        wcst_variant_setup,
                        rapm_n_questions_setup,
                        researcher_toggle,
                        researcher_group,
                        start_btn,
                        generate_link_btn,
                        participant_link_box,
                        move_history_panel,
                        task_description,
                        tab_group,
                    ],
                    queue=False,
                )

    inbrowser = bool(getattr(args, "inbrowser", False))
    if inbrowser and not sys.stdout.isatty():
        inbrowser = False
    if inbrowser and os.name != "nt":
        has_gui = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_gui:
            inbrowser = False

    launch_kwargs = {
        "server_name": getattr(args, "host", "127.0.0.1"),
        "server_port": getattr(args, "port", 7860),
        "share": bool(getattr(args, "share", False)),
        "inbrowser": inbrowser,
    }
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    local_args = SimpleNamespace(
        host="127.0.0.1",
        port=7860,
        share=False,
        inbrowser=True,
        output_dir="human_data",
    )
    launch_human_benchmark(local_args)
