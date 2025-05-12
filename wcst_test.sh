for variant in "card" "card-random" "string"
do
    # python wcst.py --model_source google --model gemini-2.0-flash --variant $variant --repeat 3

    python wcst.py --model_source hf --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --variant $variant --repeat 3
done