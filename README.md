# Cognitive Evaluation Suite

A single entrypoint (`main.py`) to run three cognitive tests against LLMs:

- WCST — Wisconsin Card Sorting Test (text or image)
- SWM — Spatial Working Memory (text or image)
- RAPM — Raven’s Progressive Matrices (image or text; supports OpenAI Batch API)

Run `python3 main.py <subcommand> [args]` with one of: `wcst`, `swm`, `rapm`.

## Setup

- Python 3.10+ recommended.
- Install deps (pip):
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `pip install openai` (ModelWrapper uses the OpenAI client for all sources)
- Or use Conda: `conda env create -f environment.yml && conda activate swm`

Environment variables by source:

- OPENAI_API_KEY (for `--model_source openai`)
- OPENROUTER_API_KEY (for `--model_source openrouter`)
- VLLM_URL (for `--model_source vllm`; server at http://$VLLM_URL:8877/v1)

Show help:

- `python3 main.py -h`
- `python3 main.py wcst -h`
- `python3 main.py swm -h`
- `python3 main.py rapm -h`

Common flags (all tasks):

- `--model` name (default varies by source)
- `--model_source` `vllm|openai|openrouter` (default `vllm`)
- `--max_tokens` int (default 512)
- `--think_budget` int (reasoning budget; 64 for WCST/SWM, 256 for RAPM)
- `--api_key` (optional; otherwise read from env)
- `--cot` enable chain-of-thought (<think>…</think> before <answer>…</answer>)

Note: All tasks expect final answers inside `<answer>...</answer>`.

---

## WCST — Wisconsin Card Sorting Test

Run examples:

- Text WCST: `python3 main.py wcst --model_source openrouter --model google/gemini-2.5-pro`
- Image WCST: `python3 main.py wcst --image --model_source openrouter --model google/gemini-2.5-pro`

Arguments:

- `--model` (default "llama")
- `--variant` `card|card-random|string|empty` (default `card`)
- `--max_trials` int (64)
- `--num_correct` int (5)
- `--repeats` int (1)
- `--ambiguous` `off|first|rest` (default `off`) — ambiguous rule handling
- `--few_shot` add demonstrations
- `--cot` add reasoning (<think>)
- `--hint` include rule hint
- `--notes` enable note-taking assistance
- `--notes-window` int (6) — note-taking window size
- `--image` image-based WCST
- `--bg-color` enable background color rule
- `--model_source` `vllm|openai|openrouter` (default `vllm`)
- `--max_tokens` (512), `--think_budget` (64), `--api_key`, `--verbose`

Outputs:

- `wcst_data/{source}_{model}_{variant}_{max_trials}-{num_correct}[...].json`
- Also `..._history.json` and `..._reasoning.json` when applicable.

---

## SWM — Spatial Working Memory

Run examples:

- Text SWM: `python3 main.py swm --model_source openrouter --model google/gemini-2.5-pro --n_boxes 6 --n_tokens 1 --runs 1`
- Image SWM: `python3 main.py swm --image --model_source openrouter --model google/gemini-2.5-pro --n_boxes 8 --n_tokens 1 --runs 1`

Arguments:

- `--model` (default inferred from source if omitted)
- `--model_source` `vllm|openai|openrouter` (default `vllm`)
- `--n_boxes` int (6)
- `--n_tokens` int (1)
- `--cot` enable reasoning
- `--runs` int (1) — repeat full run, report average score
- `--max_tokens` (512), `--think_budget` (64)
- `--notes` note-taking assistance
- `--image` image mode
- `--image-only` enable image-only mode (requires `--image`)
- `--api_key`

Outputs:

- Prints average score; temporary chat history in `data/temp_history.json`.
- Image mode uses `SWM/images` for generated grids.

---

## RAPM — Raven’s Progressive Matrices

Supports image JSON and text JSONL datasets. Example files:

- Image: `RAPM/test_rapm_data.json`
- Text: `RAPM/sample_text_rapm.jsonl`

Quick starts:

- Image RAPM: `python3 main.py rapm --model_source openrouter --model google/gemini-2.5-pro --mode image --eval_data RAPM/test_rapm_data.json --cot --patterns`
- Text RAPM (MC): `python3 main.py rapm --model_source openrouter --model google/gemini-2.5-pro --mode text --eval_data RAPM/sample_text_rapm.jsonl --answer_mode mc --cot --patterns`
- Text RAPM (Gen): `python3 main.py rapm --model_source openrouter --model google/gemini-2.5-pro --mode text --eval_data RAPM/sample_text_rapm.jsonl --answer_mode gen --cot --patterns`

Arguments:

- `--model`, `--model_source` as above
- `--mode` `image|text` (default `image`)
- `--eval_data` path to JSON (image) or JSONL (text)
- `--cot`, `--patterns`, `--max_tokens` (512), `--think_budget` (256), `--api_key`
- `--limit_per_type` int (image only; default 100; 0 = no limit)
- `--output_dir` (default `rapm_data`), `--verbose`
- Text-only: `--answer_mode` `mc|gen` (default `mc`)
- Batch API: `--batch_mode` `off|submit|collect` (default `off`)
- Batch API: `--batch_requests_path`, `--batch_id`, `--batch_id_path`, `--batch_output_jsonl`, `--batch_completion_window` (default `24h`)

Outputs:

- Writes `{base}_results.json`, `{base}_summary.json`, and optionally `{base}_reasoning.json` in `--output_dir`.
- Base name encodes source/model/mode and flags like `_gen`, `_pat`, `_cot`.

### OpenAI Batch API (RAPM)

Requires `--model_source openai` and `OPENAI_API_KEY`.

1. Submit:

```
python3 main.py rapm \
  --model o4-mini-2025-04-16 --model_source openai \
  --mode text --eval_data RAPM/sample_text_rapm.jsonl \
  --answer_mode mc --cot --patterns \
  --max_tokens 32768 --think_budget 30000 \
  --batch_mode submit \
  --batch_requests_path rapm_data/batches/o4-mini_text_mc_requests.jsonl \
  --batch_id_path rapm_data/batches/o4-mini_text_mc_id.txt \
  --batch_output_jsonl rapm_data/batches/o4-mini_text_mc_output.jsonl
```

2. Collect:

```
python3 main.py rapm \
  --model o4-mini-2025-04-16 --model_source openai \
  --mode text --eval_data RAPM/sample_text_rapm.jsonl \
  --answer_mode mc --cot --patterns \
  --batch_mode collect \
  --batch_id_path rapm_data/batches/o4-mini_text_mc_id.txt \
  --batch_output_jsonl rapm_data/batches/o4-mini_text_mc_output.jsonl
```

Helper script: `./rapm_batch.sh submit|collect` shows multi-model batch runs.

---

## Defaults when --model omitted

- vllm → `Qwen/Qwen3-32B`
- openai → `o4-mini-2025-04-16`
- openrouter → `qwen/qwen3-235b-a22b-07-25`

## Repo map

- `main.py` — orchestrates WCST/SWM/RAPM
- `WCST/` — WCST implementation
- `SWM/swm.py`, `SWM/image.py` — SWM (text/image)
- `RAPM/rapm_evaluation.py`, `RAPM/rapm_utils.py` — RAPM logic
- `shared/model_wrapper.py` — OpenAI/OpenRouter/vLLM wrapper + Batch helpers
