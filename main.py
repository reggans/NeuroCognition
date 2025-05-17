from huggingface_hub import login
import transformers
import torch
import numpy as np
from tqdm.auto import tqdm

import random, json, re, argparse, os
import string

from model_wrapper import ModelWrapper

def run_swm(model, n_boxes, n_tokens=1, cot=None, think_budget=64, note_assist=False):
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
There are {n_tokens} types of tokens, hidden in any one of {n_boxes} boxes.
Your goal is to find the {n_tokens} types of tokens {n_boxes} times each, by repeatedly selecting a box to open.
If the box contains a token, you will be informed which token type it is.
If the box does not contain a token, you will be informed that it is empty.
Once the token is found, another token of the same type will be regenerated in another box.
The token will be generated in a box that has never contained a token of that type before in the trial.
The token may be generated in a box that has been opened and found empty before, as long as it never contained the token of that type previously.
Your final answer should be a number from 1-{n_boxes}, the index of the box you selected.
"""
    model.init_chat(task_prompt)

    # Configure the question presented each turn and CoT prompt
    if cot is not None:
        cot_prompt = f"Think step-by-step, utilizing information from previous feedbacks, and state your reasoning in maximum {think_budget} tokens, wrapped with <think> and </think>. Then, provide a really short summary of your reasoning after the closing </think> tag.\n"
        question = f"Answer concisely. {cot_prompt}Which of the {n_boxes} boxes would you like to open?\nYour final answer should be a box number, wrapped with <answer> and </answer>"
    else:
        question = f"Answer only with your final answer. Which of the {n_boxes} boxes would you like to open?\nYour final answer should be a box number, wrapped with <answer> and </answer>"

    # Initialize run statistics & variables
    tokens = [string.ascii_uppercase[x] for x in range(n_tokens)]
    legal_boxes = dict.fromkeys(tokens)
    for token in tokens:
        legal_boxes[token] = [x for x in range(1, n_boxes+1)]

    worst_case_n = n_boxes ** 2
    total_guess = 0
    illegal_guess = 0
    invalid_guess = 0
    repeated_guess = 0

    # Start the test
    response = model.send_message(question, cot=cot)
    with tqdm(total=worst_case_n, desc="Total guesses") as guess_bar:
        with tqdm(total=n_boxes * n_tokens, desc="Tokens") as token_bar:
            token_box = dict.fromkeys(tokens)
            for token in tokens:
                token_box[token] = random.choice(legal_boxes[token])
                # tqdm.write(f"Token {token} put in box {token_box[token]}")
            found_tokens = []

            while (True):
                for token in found_tokens:
                    if len(legal_boxes[token]) == 0:
                        continue
                    token_box[token] = random.choice(legal_boxes[token])
                    # tqdm.write(f"Token {token} put in box {token_box[token]}")

                # Save to temp file
                with open("data/temp_history.json", "w") as f:
                    json.dump(model.history, f, indent=4)
                
                # End test
                if all([len(legal) == 0 for legal in legal_boxes.values()]):
                    break
                if total_guess >= worst_case_n:
                    break
                
                opened_boxes = set()
                found_tokens = []
                found = False
                while not found:
                    total_guess += 1
                    guess_bar.update(1)

                    with open("data/temp_history.json", "w") as f:
                        json.dump(model.history, f, indent=4)
                    
                    if total_guess >= worst_case_n:
                        break
                    
                    # Note-taking assistance
                    if note_assist:
                        notes = ""
                        for token, legal in legal_boxes.items():
                            notes += f"Boxes that has contained token {token}: "
                            for box in range(1, n_boxes+1):
                                if box not in legal:
                                    notes += f"{box}, "
                            notes += "\n"
                        notes += f"Opened boxes: "
                        for box in opened_boxes:
                            notes += f"{box}, "
                        notes += "\n"
                    
                    msg = ""
                    for token in tokens:
                        msg += f"{token} tokens found: {n_boxes - len(legal_boxes[token])}\n"

                    # Get and validate response
                    if re.search(r"<answer>(?s:.*)</answer>", response) is not None:
                        chosen_box = re.search(r"<answer>(?s:.*)</answer>", response)[0]
                        chosen_box = re.sub(r"<answer>|</answer>", "", chosen_box).strip()
                        try:
                            chosen_box = int(chosen_box)
                        except ValueError:
                            response = model.send_message(f"Please answer with a box number (1-{n_boxes}).\n" + msg + notes + question, truncate_history=True, cot=cot)
                            invalid_guess += 1
                            continue
                    else:
                        response = model.send_message(f"Please answer with the specified format\n" + msg + notes + question, truncate_history=True, cot=cot)
                        invalid_guess += 1
                        continue
                    
                    legal = False
                    for legal in legal_boxes.values():
                        if chosen_box in legal:
                            legal = True
                            break
                    if not legal:
                        illegal_guess += 1
                    elif chosen_box in opened_boxes:
                        repeated_guess += 1

                    opened_boxes.add(chosen_box)

                    for token in tokens:
                        if chosen_box == token_box[token]:
                            found = True
                            token_box[token] = -1
                            token_bar.update(1)
                            legal_boxes[token].remove(chosen_box)
                            found_tokens.append(token)
                    
                    msg = ""
                    if found:
                        for token in found_tokens:
                            msg = f"Token {token} found in box {chosen_box}.\n" + msg
                    else:
                        msg += f"No tokens found in box {chosen_box}.\n" + msg

                    response = model.send_message(msg + notes + question, truncate_history=True, cot=cot)
                    model.history[-2]["content"] = msg      # Truncate user response length

    run_stats = {
        "worst_case_guesses": worst_case_n,
        "illegal": illegal_guess,
        "guesses": total_guess,
        "invalid": invalid_guess,
        "repeated": repeated_guess,
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
    return 1 - (run_stats['illegal'] + run_stats['repeated']) / (run_stats['guesses'] - run_stats['invalid'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SWM problem.")
    parser.add_argument("--model", type=str, default=None, help="The model to use.")
    parser.add_argument("--model_source", type=str, default="hf", help="The source of the model.")
    parser.add_argument("--n_boxes", type=int, default=6, help="The number of boxes in the test. More == harder.")
    parser.add_argument("--n_tokens", type=int, default=1, help="The number of different tokens present at the same time. More == harder")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--runs", type=int, default=1, help="The number of runs to perform.")
    parser.add_argument("--max_tokens", type=int, default=512, help="The maximum number of tokens to generate.")
    parser.add_argument("--think_budget", type=int, default=64, help="The budget tokens for reasoning.")
    parser.add_argument("--api_key", type=str, default=None, help="API key to use. If none, uses key stored in environment variable.")
    args = parser.parse_args()

    # Input validation
    if args.model_source not in ["hf", "google", "litellm", "vllm"]:
        raise ValueError("Model source must be either 'hf', 'google', 'litellm', or 'vllm'.")

    if args.model is None:
        if args.model_source == "hf":
            args.model = "unsloth/Meta-Llama-3.1-8B-Instruct"
        elif args.model_source == "google":
            args.model = "gemini-1.5-flash-8b"
        elif args.model_source == "litellm":
            args.model = "gpt-4o-mini-2024-07-18"
        elif args.model_source == "vllm":
            args.model = "meta-llama/Llama-2-7b-chat-hf"
    
    if not os.path.isdir("./data"):
        os.mkdir("./data")

    file_header = f"data/{args.model_source}_{args.model.replace('/', '-')}{'_cot' if args.cot else ''}_{args.n_boxes}_{args.n_tokens}_"
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
        exit(0)

    run_stats = {}
    run_history = {}
    run_reasoning = {}  # Add new dictionary for reasoning traces
    
    for i in range(args.runs):
        model = None
        torch.cuda.empty_cache()
        
        model = ModelWrapper(args.model, args.model_source, api_key=args.api_key, max_new_tokens=args.max_tokens)

        print(f"Run {i+1}/{args.runs}")
        run_stats[f"run_{i+1}"] = run_swm(model, args.n_boxes, n_tokens=args.n_tokens, cot=args.cot, think_budget=args.think_budget)
        run_history[f"run_{i+1}"] = model.history
        run_reasoning[f"run_{i+1}"] = model.reasoning_trace  # Save reasoning trace

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