#!/bin/bash

set -e  # Exit on error

# Model configuration
MODEL_SOURCE="openrouter"
MODEL="google/gemini-2.5-pro"

echo "==================================="
echo "CognitiveEval Test Suite"
echo "==================================="
echo "Model Source: $MODEL_SOURCE"
echo "Model: $MODEL"
echo ""

echo "Testing Wisconsin Card Sorting Test (WCST)..."
python main.py wcst \
    --model_source "$MODEL_SOURCE" \
    --model "$MODEL" \
    --cot \
    --max_trials 64 \
    --num_correct 5 \
    --repeats 3 \
    --ambiguous off

echo ""
echo "Testing Spatial Working Memory (SWM)..."
python main.py swm \
    --model_source "$MODEL_SOURCE" \
    --model "$MODEL" \
    --cot \
    --n_boxes 8 \
    --n_tokens 1 \
    --runs 3

echo ""
echo "Testing Raven's Progressive Matrices (RAPM)..."

python main.py rapm \
    --model_source "$MODEL_SOURCE" \
    --model "$MODEL" \
    --mode text \
    --eval_data RAPM/sample_text_rapm.jsonl \
    --cot \
    --answer_mode mc

echo ""
echo "All tests completed!"
