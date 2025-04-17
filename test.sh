# python main.py --model unsloth/QwQ-32B-unsloth-bnb-4bit --model_source hf --max_tokens 2048 --runs 5 --n_boxes 8

for box in 8 10 12
do
    for token in 1 2
    do
        # LiteLLM
        python main.py --model gpt-4.5-preview-2025-02-27 --model_source litellm --runs 5 --n_boxes $box --n_tokens $token
        # python main.py --model gpt-4o-mini-2024-07-18 --model_source litellm --runs 5 --n_boxes $box --n_tokens $token
        python main.py --model gemini/gemini-2.0-flash --model_source litellm --runs 5 --n_boxes $box --n_tokens $token
        # python main.py --model gemini/gemini-1.5-pro --model_source litellm --runs 5 --n_boxes $box --n_tokens $token
        python main.py --model anthropic/claude-3-7-sonnet-latest --model_source litellm --runs 5 --n_boxes $box --n_tokens $token
        # python main.py --model anthropic/claude-3-5-haiku-latest --model_source litellm --runs 5 --n_boxes $box --n_tokens $token

        python main.py --model gpt-4.5-preview-2025-02-27 --model_source litellm --runs 5 --n_boxes $box --cot implicit --n_tokens $token
        # python main.py --model gpt-4o-mini-2024-07-18 --model_source litellm --runs 5 --n_boxes $box --cot implicit --n_tokens $token
        python main.py --model gemini/gemini-2.0-flash --model_source litellm --runs 5 --n_boxes $box --cot implicit --n_tokens $token
        # python main.py --model gemini/gemini-1.5-flash --model_source litellm --runs 5 --n_boxes $box --cot implicit --n_tokens $token
        python main.py --model anthropic/claude-3-7-sonnet-latest --model_source litellm --runs 5 --n_boxes $box --cot implicit --n_tokens $token
        # python main.py --model anthropic/claude-3-5-haiku-latest --model_source litellm --runs 5 --n_boxes $box --cot implicit --n_tokens $token

        python main.py --model gpt-4.5-preview-2025-02-27 --model_source litellm --runs 5 --n_boxes $box --cot explicit --n_tokens $token
        # python main.py --model gpt-4o-mini-2024-07-18 --model_source litellm --runs 5 --n_boxes $box --cot explicit --n_tokens $token
        python main.py --model gemini/gemini-2.0-flash --model_source litellm --runs 5 --n_boxes $box --cot explicit --n_tokens $token
        # python main.py --model gemini/gemini-1.5-pro --model_source litellm --runs 5 --n_boxes $box --cot explicit --n_tokens $token
        python main.py --model anthropic/claude-3-7-sonnet-latest --model_source litellm --runs 5 --n_boxes $box --cot explicit --n_tokens $token
        # python main.py --model anthropic/claude-3-5-haiku-latest --model_source litellm --runs 5 --n_boxes $box --cot explicit --n_tokens $token
    done
done