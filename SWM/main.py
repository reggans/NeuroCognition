from huggingface_hub import login
import transformers
import torch
import numpy as np
from tqdm.auto import tqdm

import random, json, re, argparse, os
import string

from ..shared.model_wrapper import ModelWrapper
from .swm import image_swm, text_swm, score

def swm_main(model=None, model_source="hf", n_boxes=6, n_tokens=1, cot=False, runs=1, 
                 max_tokens=512, think_budget=64, notes=False, image=False, api_key=None):
    """
    Run the Spatial Working Memory (SWM) test.
    
    Args:
        model: The model to use
        model_source: The source of the model ("hf", "google", "litellm", "vllm")
        n_boxes: The number of boxes in the test
        n_tokens: The number of different tokens present at the same time
        cot: Whether to use chain-of-thought reasoning
        runs: The number of runs to perform
        max_tokens: Maximum number of tokens to generate
        think_budget: Budget tokens for reasoning
        notes: Whether to use note-taking assistance
        image: Whether to use image mode
        api_key: API key to use
    """
    # Input validation
    if model_source not in ["hf", "google", "litellm", "vllm"]:
        raise ValueError("Model source must be either 'hf', 'google', 'litellm', or 'vllm'.")

    if model is None:
        if model_source == "hf":
            model = "unsloth/Meta-Llama-3.1-8B-Instruct"
        elif model_source == "google":
            model = "gemini-1.5-flash-8b"
        elif model_source == "litellm":
            model = "gpt-4o-mini-2024-07-18"
        elif model_source == "vllm":
            model = "meta-llama/Llama-2-7b-chat-hf"
    
    if image:
        if notes:
            os.makedirs("SWM/data/image/baselines", exist_ok=True)
        else:
            os.makedirs("SWM/data/image", exist_ok=True)
        img_path = "SWM/images"
        os.makedirs(img_path, exist_ok=True)
    else:
        if notes:
            os.makedirs("SWM/data/text/baselines", exist_ok=True)
        else:
            os.makedirs("SWM/data/text", exist_ok=True)
        img_path = None

    file_header = f"SWM/data/{'image/' if image else 'text/'}{'baselines/' if notes else ''}{model_source}_{model.replace('/', '-')}{'_cot' if cot else ''}_{n_boxes}_{n_tokens}_"
    print(f"Saving to: {file_header}")

    # Check if results already exist
    stats_file = file_header + "run_stats.json"
    history_file = file_header + "run_history.json"
    
    if os.path.exists(stats_file) and os.path.exists(history_file):
        print(f"Results already exist at {stats_file}")
        with open(stats_file, 'r') as f:
            run_stats = json.load(f)
            
        # Calculate and display statistics
        avg_stats = {}
        for key in run_stats["run_1"].keys():
            if type(run_stats["run_1"][key]) == int:
                avg_stats[key] = np.mean([stats[key] for stats in run_stats.values()])
        
        for key, value in avg_stats.items():
            print(f"{key}: {value}")
        
        tot_score = 0
        for stats in run_stats.values():
            tot_score += score(stats)
        print(f'Score: {tot_score / len(run_stats.keys())}')
        return

    run_stats = {}
    run_history = {}
    run_reasoning = {}  # Add new dictionary for reasoning traces
    
    for i in range(runs):
        model_instance = None
        torch.cuda.empty_cache()
        
        model_instance = ModelWrapper(model, model_source, api_key=api_key, max_new_tokens=max_tokens, image_input=image, image_path=img_path)

        print(f"Run {i+1}/{runs}")
        if image:
            run_stats[f"run_{i+1}"] = image_swm(model_instance, n_boxes, n_tokens=n_tokens, cot=cot, think_budget=think_budget, note_assist=notes)
        else:
            run_stats[f"run_{i+1}"] = text_swm(model_instance, n_boxes, n_tokens=n_tokens, cot=cot, think_budget=think_budget, note_assist=notes)
        run_history[f"run_{i+1}"] = model_instance.history
        run_reasoning[f"run_{i+1}"] = model_instance.reasoning_trace  # Save reasoning trace

        with open(file_header + "run_stats.json", "w") as f:
            json.dump(run_stats, f, indent=4)
        
        with open(file_header + "run_history.json", "w") as f:
            json.dump(run_history, f, indent=4)
            
        with open(file_header + "run_reasoning.json", "w") as f:  # Save reasoning traces
            json.dump(run_reasoning, f, indent=4)

    avg_stats = {}
    for key in run_stats["run_1"].keys():
        if type(run_stats["run_1"][key]) == int:
            avg_stats[key] = np.mean([stats[key] for stats in run_stats.values()])
    
    for key, value in avg_stats.items():
        print(f"{key}: {value}")
    
    tot_score = 0
    for stats in run_stats.values():
        tot_score += score(stats)
    print(f'Score: {tot_score / len(run_stats.keys())}')