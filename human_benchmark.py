#!/usr/bin/env python3
"""Human-facing benchmark runner built on top of existing task logic."""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

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
    if setup == "image+text":
        return "image", False
    if setup == "image-only":
        return "image", True
    raise ValueError(f"Unknown setup: {setup}")


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


def _build_filename(state: Dict[str, Any]) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pid = re.sub(r"[^a-zA-Z0-9_-]", "_", state["participant_id"]) or "participant"
    return f"{pid}_{state['task']}_{state['mode']}_{ts}.json"


def _persist_session(state: Dict[str, Any]) -> str:
    output_dir = state["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    final_metrics = (
        _rapm_metrics(state) if state["task"] == "rapm" else state["env"].get_metrics()
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
    participant_id: str,
    task: str,
    setup: str,
    swm_boxes: int,
    swm_tokens: int,
    wcst_variant: str,
    wcst_max_trials: int,
    wcst_num_correct: int,
    wcst_bg_color: bool,
    wcst_ambiguous: str,
    rapm_eval_data: str,
    rapm_answer_mode: str,
    output_dir: str,
) -> Tuple[Dict[str, Any], str, Optional[str], str, Dict[str, Any], str]:
    participant_id = participant_id.strip() or "participant"
    env_mode, image_only = _resolve_mode(task, setup)
    state: Dict[str, Any] = {
        "participant_id": participant_id,
        "task": task,
        "setup": setup,
        "mode": env_mode,
        "image_only": image_only,
        "output_dir": output_dir,
        "started_at": datetime.utcnow().isoformat(),
        "turn_logs": [],
        "done": False,
        "prompt_shown_at": time.time(),
        "config": {
            "participant_id": participant_id,
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
            "rapm_eval_data": rapm_eval_data,
            "rapm_answer_mode": rapm_answer_mode,
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
        metrics = env.get_metrics()
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
        metrics = env.get_metrics()
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

    if not rapm_eval_data or not os.path.exists(rapm_eval_data):
        return (
            state,
            "",
            None,
            "RAPM eval_data path is missing or invalid.",
            {},
            "Error",
        )

    state["rapm_eval_data"] = rapm_eval_data
    state["rapm_answer_mode"] = rapm_answer_mode
    state["rapm_index"] = 0
    state["rapm_answered"] = 0
    state["rapm_correct"] = 0

    if env_mode == "image":
        state["rapm_questions"] = load_evaluation_data(rapm_eval_data)
        state["rapm_total"] = len(state["rapm_questions"])
    else:
        state["rapm_items"] = load_text_rapm_jsonl(rapm_eval_data)
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
    metrics = env.get_metrics()
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

**How it works:**
1. The system will tell you "Correct!" or "Incorrect. Please try again."
2. If incorrect, either you made a mistake OR the rule has changed.
3. When you get 5 correct answers in a row, the sorting rule changes.
4. Your answer should be a single number: 1, 2, 3, or 4.
""",
        "swm": """**Spatial Working Memory (SWM) Test**

You will see a grid of closed boxes. Your task is to find tokens (one at a time) hidden in the boxes.

**How it works:**
1. Each box contains at most one token.
2. Once you find a token in a box, that box will never contain another token.
3. You must remember which boxes you've already opened.
4. Try to find all tokens with as few errors as possible.

**Answering:**
- Enter the coordinates of a box to open it.
- Coordinates are in format: row,col (e.g., "0,1" for row 0, column 1)
- Rows and columns are **0-indexed** (starting from 0)
""",
        "rapm": """**Raven's Advanced Progressive Matrices (RAPM)**

You will be shown matrix puzzles. Each puzzle shows a pattern with one cell missing.
Your task is to find the pattern and select the correct option that completes the matrix.

**How it works:**
1. Analyze the matrix image to identify the pattern.
2. Choose the option number (1-8) that best completes the pattern.
3. There are no correct answers provided; you must find the pattern yourself.
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
            with gr.Tab("Setup") as setup_tab:
                gr.Markdown("### Enter Your Information & Select Task")

                with gr.Group():
                    participant_id = gr.Textbox(
                        label="Participant ID",
                        placeholder="e.g., P001 or your name",
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
                        rapm_answer_mode_setup = gr.Dropdown(
                            choices=["mc", "gen"],
                            value="mc",
                            label="RAPM: Answer Mode",
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
                        rapm_eval_data_adv = gr.Textbox(
                            label="RAPM eval_data path",
                            placeholder="Path to RAPM evaluation data JSON",
                            value="",
                        )

                researcher_toggle = gr.Checkbox(
                    value=False, label="Show Researcher Settings", interactive=True
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
                        setup = gr.update(choices=["text", "image+text"], value="text")
                        researcher_panel = gr.update(visible=show_researcher)
                        wcst_task = gr.update(visible=True)
                        swm_task = gr.update(visible=False)
                        rapm_task = gr.update(visible=False)
                        wcst_research = gr.update(visible=show_researcher)
                        swm_research = gr.update(visible=False)
                        rapm_research = gr.update(visible=False)
                    elif t == "swm":
                        setup = gr.update(
                            choices=["text", "image+text", "image-only"], value="text"
                        )
                        researcher_panel = gr.update(visible=show_researcher)
                        wcst_task = gr.update(visible=False)
                        swm_task = gr.update(visible=True)
                        rapm_task = gr.update(visible=False)
                        wcst_research = gr.update(visible=False)
                        swm_research = gr.update(visible=show_researcher)
                        rapm_research = gr.update(visible=False)
                    else:
                        setup = gr.update(choices=["text", "image+text"], value="text")
                        researcher_panel = gr.update(visible=show_researcher)
                        wcst_task = gr.update(visible=False)
                        swm_task = gr.update(visible=False)
                        rapm_task = gr.update(visible=True)
                        wcst_research = gr.update(visible=False)
                        swm_research = gr.update(visible=False)
                        rapm_research = gr.update(visible=show_researcher)

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

                status_setup = gr.Textbox(
                    label="Status", interactive=False, value="Ready"
                )

                def start_task(
                    pid,
                    tsk,
                    setup,
                    swm_boxes,
                    swm_tokens,
                    wcst_var,
                    wcst_max,
                    wcst_num,
                    wcst_bg,
                    wcst_amb,
                    rapm_data,
                    rapm_mode,
                    out_dir,
                ):
                    try:
                        st, obs, img, fb, met, stat = _start_session(
                            pid,
                            tsk,
                            setup,
                            swm_boxes,
                            swm_tokens,
                            wcst_var,
                            wcst_max,
                            wcst_num,
                            wcst_bg,
                            wcst_amb,
                            rapm_data,
                            rapm_mode,
                            out_dir,
                        )
                        # Store observation and image in state for sync to test tab
                        st["_initial_observation"] = obs
                        st["_initial_image"] = img
                        return st, stat
                    except Exception as e:
                        return {}, f"Error: {str(e)}"

                start_btn = gr.Button("Start Test", variant="primary", size="lg")
                start_btn.click(
                    fn=start_task,
                    inputs=[
                        participant_id,
                        task,
                        setup_type,
                        swm_boxes_setup,
                        swm_tokens_adv,
                        wcst_variant_setup,
                        wcst_max_trials_adv,
                        wcst_num_correct_adv,
                        wcst_bg_color_adv,
                        wcst_ambiguous_adv,
                        rapm_eval_data_adv,
                        rapm_answer_mode_setup,
                        output_dir_state,
                    ],
                    outputs=[session_state, status_setup],
                    queue=False,
                )

            # ========== TEST TAB ==========
            with gr.Tab("Test") as test_tab:
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

                feedback = gr.Textbox(label="Feedback", interactive=False, value="")
                metrics = gr.JSON(label="Progress")
                status = gr.Textbox(label="Status", interactive=False, value="Idle")

                def on_submit(ans, state):
                    if not state or state.get("done"):
                        return state or {}, "", None, "", {}, "Idle", ""

                    wrapped = _wrap_answer(ans)
                    dt = max(
                        0.0,
                        time.time() - float(state.get("prompt_shown_at", time.time())),
                    )
                    task_name = state.get("task", "")

                    if task_name == "rapm":
                        obs, img, fb, met, stat = _step_rapm(state, wrapped, dt)
                        if state.get("done"):
                            return state, "", img, fb, met, "Completed", ""
                        state["prompt_shown_at"] = time.time()
                        return state, obs, img, fb, met, stat, ""
                    else:
                        env = state.get("env")
                        if not env:
                            return (
                                state,
                                "",
                                None,
                                "No environment loaded",
                                {},
                                "Error",
                                "",
                            )

                        step = env.step(wrapped)
                        image_path = (
                            env.get_current_image_path()
                            if state.get("mode") == "image"
                            else None
                        )
                        met = env.get_metrics()
                        fb = f"Reward: {step.reward}"

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
                                "",
                                image_path,
                                f"{fb}. Results saved: {out_path}",
                                met,
                                "Completed",
                                "",
                            )

                        state["prompt_shown_at"] = time.time()
                        next_obs = (
                            ""
                            if state.get("image_only")
                            else _clean_observation(step.observation)
                        )
                        return state, next_obs, image_path, fb, met, "Running", ""

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
                        answer,
                    ],
                    queue=False,
                )

                def on_end(state):
                    if not state or not state.get("participant_id"):
                        return {}, "", None, "No active session.", {}, "Idle"
                    if not state.get("done") and state.get("turn_logs"):
                        out_path = _persist_session(state)
                        status_msg = f"Session saved: {out_path}"
                    else:
                        status_msg = "Session closed."
                    state["done"] = True
                    return state, "", None, status_msg, {}, "Idle"

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
                    ],
                    queue=False,
                )

                # Sync session state on load/change
                def sync_session(state):
                    if not state or not state.get("participant_id"):
                        return "", "", None, "", {}, "Idle"

                    info = f"{state['participant_id']} | {state['task']}"
                    obs = state.get("_initial_observation", "")
                    img = state.get("_initial_image")
                    fb = "Ready for your first answer"
                    return info, obs, img, fb, {}, "Running"

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
