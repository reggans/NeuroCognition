import google.generativeai as genai
from huggingface_hub import login
import transformers
import torch
import numpy as np
from tqdm.auto import tqdm

import random, json, re, argparse, os

from model_wrapper import ModelWrapper

def run_swm(model, n_boxes, cot=None):
    """
    Run the Spatial Working Memory (SWM) test with the given model.
    Args:
        model (ModelWrapper): The model to use.
        n_boxes (int): The number of boxes in the test.
        cot (str): The type of CoT to use. Either "implicit" or "explicit".
        verbose (bool): Whether to print verbose output.
    Returns:
        dict: The run statistics.
    """
    # Initiate w/ task prompt
    task_prompt = f"""You will be performing a text version of the Spatial Working Memory (SWM) test.
There are {n_boxes} boxes, one of which contains a token.
Your goal is to find {n_boxes} tokens by repeatedly selecting a box to open.
If the box contains a token, I will respond with "TOKEN".
If the box is empty, I will respond with "EMPTY".
Once the token is found, the token will be regenerated in another box.
The token will be generated in a box that has never contained a token before in the trial.
The token may be generated in a box that has been opened and found empty before, as long as it never contained the token previously.
Your final answer should be a number from 1-{n_boxes}, the index of the box you selected.
"""
    model.init_chat(task_prompt)

    # Configure the question presented each turn and CoT prompt
    if cot is not None:
        if cot == "implicit":
            cot_prompt = "Think step-by-step, utilizing information from previous feedbacks, and state your reasoning clearly.\n"
        elif cot == "explicit":
            cot_prompt = cot_prompt + "Track the boxes where the tokens have been found before.\n"""
        else:
            raise ValueError("CoT must be None, or either of 'implicit' or 'explicit'.")
        question = f"Answer concisely. {cot_prompt}Which of the {n_boxes} boxes would you like to open?\nState your final answer by wrapping it with <ANS> and </ANS>"
    else:
        question = f"Answer concisely. Which of the {n_boxes} boxes would you like to open?\nState your final answer by wrapping it with <ANS> and </ANS>"

    # Initialize run statistics & variables
    legal_boxes = set(x for x in range(1, n_boxes+1))
    worst_case_n = sum(legal_boxes)
    n_guesses = []
    illegal_guesses = []
    invalid_guesses = []
    repeated_guesses = [] 

    # Start the test
    # Modified: Any chosen box will be empty unless it is the last legal box
    response = model.send_message(question)
    with tqdm(total=2*worst_case_n, desc="Total guesses") as pbar:
        for i in tqdm(range(n_boxes), total=n_boxes, desc="Rounds"):
            n_guesses.append(0)
            illegal_guesses.append(0)
            invalid_guesses.append(0)
            repeated_guesses.append(0)

            # Track opened boxes in this trial
            opened_boxes = []

            found = False
            while not found:
                n_guesses[-1] += 1
                print(n_guesses[-1])
                pbar.update(1)

                # Save to temp file
                with open("data/temp_history.json", "w") as f:
                    json.dump(model.history, f)

                if sum(n_guesses) > 2*worst_case_n:
                    break
                
                # Get and validate response
                if re.search(r"<ANS>(?s:.*)</ANS>", response) is not None:
                    chosen_box = re.search(r"<ANS>(?s:.*)</ANS>", response)[0]
                    chosen_box = re.sub(r"<ANS>|</ANS>", "", chosen_box).strip()
                    try:
                        chosen_box = int(chosen_box)
                    except ValueError:
                        response = model.send_message("Please answer with the specified format\nTokens found: {i}\n" + question)
                        invalid_guesses[-1] += 1
                        continue
                else:
                    response = model.send_message("Please answer with the specified format\nTokens found: {i}\n" + question)
                    invalid_guesses[-1] += 1
                    continue

                if chosen_box in opened_boxes:      
                    response = model.send_message(f"EMPTY\nBox {chosen_box} is empty.\nTokens found: {i}\n" + question)
                    repeated_guesses[-1] += 1
                else:              
                    opened_boxes.append(chosen_box)

                    if chosen_box not in legal_boxes:    
                        response = model.send_message(f"EMPTY\nBox {chosen_box} is empty.\nTokens found: {i}\n" + question)
                        illegal_guesses[-1] += 1
                    
                    elif len(legal_boxes.intersection(opened_boxes)) == len(legal_boxes):
                        response = model.send_message(f"TOKEN\nBox {chosen_box} contains a token.\nTokens found: {i+1}\n" + question)
                        found = True
                        legal_boxes.remove(chosen_box)
                
                
            # Save to temp file
            with open("data/temp_history.json", "w") as f:
                json.dump(model.history, f)
            
            if sum(n_guesses) > 2*worst_case_n:
                break
    
    run_stats = {
        "total_guesses": sum(n_guesses),
        "total_illegal": sum(illegal_guesses),
        "total_invalid": sum(invalid_guesses),
        "total_repeated": sum(repeated_guesses),
        "worst_case_guesses": worst_case_n,
        "illegal": illegal_guesses,
        "guesses": n_guesses,
        "invalid": invalid_guesses,
        "repeated": repeated_guesses
    }

    return run_stats

def score(run_stats):
    """
    Score the run statistics from the SWM test.
    Args:
        run_stats (dict): The run statistics.
    Returns:
        float: The score.
    """
    return run_stats["total_guesses"] / run_stats["worst_case_guesses"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SWM problem.")
    parser.add_argument("--model", type=str, default=None, help="The model to use.")
    parser.add_argument("--model_source", type=str, default="hf", help="The source of the model.")
    parser.add_argument("--n_boxes", type=int, default=6, help="The number of boxes in the test. More == harder.")
    parser.add_argument("--cot", type=str, default=None, help="The type of CoT to use.")
    parser.add_argument("--runs", type=int, default=1, help="The number of runs to perform.")
    parser.add_argument("--max_tokens", type=int, default=512, help="The maximum number of tokens to generate.")
    # parser.add_argument("--seed", type=int, default=42, help="The random seed.")
    parser.add_argument("--api_key", type=str, default=None, help="API key to use. If none, uses key stored in environment variable.")
    args = parser.parse_args()

    # Input validation
    if args.cot is not None and args.cot not in ["implicit", "explicit"]:
        raise ValueError("CoT must be None, or either of 'implicit' or 'explicit'")
    if args.model_source not in ["hf", "google", "litellm"]:
        raise ValueError("Model source must be either 'hf', 'google', or 'litellm'.")
    
    if args.model is None:
        if args.model_source == "hf":
            args.model = "unsloth/Meta-Llama-3.1-8B-Instruct"
        elif args.model_source == "google":
            args.model = "gemini-1.5-flash-8b"
        elif args.model_source == "litellm":
            args.model = "gpt-4o-mini-2024-07-18"
    
    if not os.path.isdir("./data"):
        os.mkdir("./data")

    file_header = f"data/{args.model_source}_{args.model.replace("/", "-")}{"_" + args.cot if args.cot is not None else ''}_{args.n_boxes}_"
    
    run_stats = {}
    run_history = {}
    for i in range(args.runs):
        model = None
        torch.cuda.empty_cache()
        
        model = ModelWrapper(args.model, args.model_source, api_key=args.api_key, max_new_tokens=args.max_tokens)

        print(f"Run {i+1}/{args.runs}")
        run_stats[f"run_{i+1}"] = run_swm(model, args.n_boxes, cot=args.cot)
        run_history[f"run_{i+1}"] = model.history

        with open(file_header + "run_stats.json", "w") as f:
            json.dump(run_stats, f)
        
        with open(file_header + "run_history.json", "w") as f:
            json.dump(run_history, f)


    avg_stats = {}
    for key in run_stats["run_1"].keys():
        if type(run_stats["run_1"][key]) == int:
            avg_stats[key] = np.mean([stats[key] for stats in run_stats.values()])
    
    for key, value in avg_stats.items():
        print(f"{key}: {value}")