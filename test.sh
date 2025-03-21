# python main.py --model unsloth/QwQ-32B-unsloth-bnb-4bit --model_source hf --max_tokens 2048 --runs 5 --n_boxes 8
python main.py --model gpt-4.5-preview-2025-02-27 --model_source litellm --runs 5 --n_boxes 8
python main.py --model gemini/gemini-2.0-pro-exp-02-05 --model_source litellm --runs 5 --n_boxes 8
python main.py --model claude-3-7-sonnet-latest --model_source litellm --runs 5 --n_boxes 8