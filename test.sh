# python main.py --model unsloth/QwQ-32B-unsloth-bnb-4bit --model_source hf --max_tokens 2048 --runs 5 --n_boxes 8

# LiteLLM
python main.py --model gpt-4.5-preview-2025-02-27 --model_source litellm --runs 5 --n_boxes 8 --n_tokens 2
python main.py --model gpt-4o-mini-2024-07-18 --model_source litellm --runs 5 --n_boxes 8 --n_tokens 2
python main.py --model gemini/gemini-2.0-flash --model_source litellm --runs 5 --n_boxes 8 --n_tokens 2
python main.py --model gemini/gemini-1.5-pro --model_source litellm --runs 5 --n_boxes 8 --n_tokens 2
python main.py --model anthropic/claude-3-7-sonnet-latest --model_source litellm --runs 5 --n_boxes 8 --n_tokens 2
python main.py --model anthropic/claude-3-5-haiku-latest --model_source litellm --runs 5 --n_boxes 8 --n_tokens 2

python main.py --model gpt-4.5-preview-2025-02-27 --model_source litellm --runs 5 --n_boxes 8 --cot implicit --n_tokens 2
python main.py --model gpt-4o-mini-2024-07-18 --model_source litellm --runs 5 --n_boxes 8 --cot implicit --n_tokens 2
python main.py --model gemini/gemini-2.0-flash --model_source litellm --runs 5 --n_boxes 8 --cot implicit --n_tokens 2
python main.py --model gemini/gemini-1.5-flash --model_source litellm --runs 5 --n_boxes 8 --cot implicit --n_tokens 2
python main.py --model anthropic/claude-3-7-sonnet-latest --model_source litellm --runs 5 --n_boxes 8 --cot implicit --n_tokens 2
python main.py --model anthropic/claude-3-5-haiku-latest --model_source litellm --runs 5 --n_boxes 8 --cot implicit --n_tokens 2

python main.py --model gpt-4.5-preview-2025-02-27 --model_source litellm --runs 5 --n_boxes 8 --cot explicit --n_tokens 2
python main.py --model gpt-4o-mini-2024-07-18 --model_source litellm --runs 5 --n_boxes 8 --cot explicit --n_tokens 2
python main.py --model gemini/gemini-2.0-flash --model_source litellm --runs 5 --n_boxes 8 --cot explicit --n_tokens 2
python main.py --model gemini/gemini-1.5-pro --model_source litellm --runs 5 --n_boxes 8 --cot explicit --n_tokens 2
python main.py --model anthropic/claude-3-7-sonnet-latest --model_source litellm --runs 5 --n_boxes 8 --cot explicit --n_tokens 2
python main.py --model anthropic/claude-3-5-haiku-latest --model_source litellm --runs 5 --n_boxes 8 --cot explicit --n_tokens 2