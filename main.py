import google.generativeai as genai
from huggingface_hub import login
import transformers
import torch
import numpy as np
from tqdm.auto import tqdm

import random, json, re, argparse, os
import string

from model_wrapper import ModelWrapper

def run_swm(model, n_boxes, n_tokens=1, cot=None):
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
        if cot == "implicit":
            cot_prompt = "Think step-by-step, utilizing information from previous feedbacks, and state your reasoning clearly.\n"
        elif cot == "explicit":
            cot_prompt = "Think step-by-step, utilizing information from previous feedbacks, and state your reasoning clearly.\n"
            cot_prompt = cot_prompt + "Track the boxes where the tokens have been found before.\n"""
        else:
            raise ValueError("CoT must be None, or either of 'implicit' or 'explicit'.")
        question = f"Answer concisely. {cot_prompt}Which of the {n_boxes} boxes would you like to open?\nState your final answer by wrapping it with <ANS> and </ANS>"
    else:
        question = f"Answer concisely. Which of the {n_boxes} boxes would you like to open?\nState your final answer by wrapping it with <ANS> and </ANS>"

    # Initialize run statistics & variables
    tokens = [string.ascii_uppercase[x] for x in range(n_tokens)]
    legal_boxes = dict.fromkeys(tokens)
    for token in tokens:
        legal_boxes[token] = [x for x in range(n_boxes)]

    worst_case_n = sum(legal_boxes[token[0]]) * 2
    total_guess = 0
    illegal_guess = 0
    invalid_guess = 0
    repeated_guess = 0

    # Start the test
    response = model.send_message(question)
    with tqdm(total=2*worst_case_n, desc="Total guesses") as guess_bar:
        with tqdm(total=n_boxes * 2, desc="Tokens") as token_bar:
            token_box = dict.fromkeys(tokens)
            for token in tokens:
                token_box[token] = random.choice(legal_boxes[token])
            found_tokens = []
            
            while (True):
                for token in found_tokens:
                    token_box[token] = random.choice(legal_boxes[token])
                opened_boxes = set()

                found_tokens = []
                found = False
                while not found:
                    total_guess += 1
                    guess_bar.update(1)

                    with open("data/temp_history.json", "w") as f:
                        json.dump(model.history, f)
                    
                    if total_guess > 2*worst_case_n:
                        break

                    # Get and validate response
                    if re.search(r"<ANS>(?s:.*)</ANS>", response) is not None:
                        chosen_box = re.search(r"<ANS>(?s:.*)</ANS>", response)[0]
                        chosen_box = re.sub(r"<ANS>|</ANS>", "", chosen_box).strip()
                        try:
                            chosen_box = int(chosen_box)
                        except ValueError:
                            response = model.send_message(f"Please answer with the specified format\nTokens found: {i}\n" + question)
                            invalid_guess += 1
                            continue
                    else:
                        response = model.send_message(f"Please answer with the specified format\nTokens found: {i}\n" + question)
                        invalid_guess += 1
                        continue
                    
                    legal = False
                    for legal in legal_boxes.values():
                        if chosen_box in legal:
                            legal = True
                            break
                    if not legal:
                        illegal_guess += 1
                    if chosen_box in opened_boxes:
                        repeated_guess += 1

                    opened_boxes.add(chosen_box)

                    # After first guess, choose a box for the token excluding the chosen box
                    # if token_box is None and len(set(legal_boxes).intersection(opened_boxes)) == 1:
                    #     if len(legal_boxes) == 1:
                    #         token_box = legal_boxes[0]
                    #     else:
                    #         legal_boxes.remove(chosen_box)
                    #         token_box = random.choice(legal_boxes)
                    #         legal_boxes.append(chosen_box)
                    
                    for token in tokens:
                        if chosen_box == token_box[token]:
                            found = True
                            token_bar.update(1)
                            legal_boxes[tokens[i]].remove(chosen_box)
                            found_tokens.append(tokens[i])
                    
                    msg = ""
                    if found:
                        for token in found_tokens:
                            msg += f"Token {token} found in box {chosen_box}.\n"
                        
                        for token in tokens:
                            msg += f"{token} tokens found: {n_boxes - len(legal_boxes[token])}\n"
                    else:
                        msg += f"No tokens found in box {chosen_box}.\n"
                        for token in tokens:
                            msg += f"{token} tokens found: {n_boxes - len(legal_boxes[token])}\n"

                    response = model.send_message(msg + question)
                    
                # Save to temp file
                with open("data/temp_history.json", "w") as f:
                    json.dump(model.history, f)
                
                if all([len(legal) == 0 for legal in legal_boxes.values()]):
                    break

                if total_guess > 2*worst_case_n:
                    break
    
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
    return 1 - (run_stats['total_illegal'] + run_stats['total_repeated']) / run_stats['total_guesses']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SWM problem.")
    parser.add_argument("--model", type=str, default=None, help="The model to use.")
    parser.add_argument("--model_source", type=str, default="hf", help="The source of the model.")
    parser.add_argument("--n_boxes", type=int, default=6, help="The number of boxes in the test. More == harder.")
    parser.add_argument("--n_tokens", type=int, default=1, help="The number of different tokens present at the same time. More == harder")
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

    file_header = f"data/{args.model_source}_{args.model.replace('/', '-')}{'_' + args.cot if args.cot is not None else ''}_{args.n_boxes}_{args.n_tokens}_"
    print(f"Saving to: {file_header}")

    run_stats = {}
    run_history = {}
    for i in range(args.runs):
        model = None
        torch.cuda.empty_cache()
        
        model = ModelWrapper(args.model, args.model_source, api_key=args.api_key, max_new_tokens=args.max_tokens)

        print(f"Run {i+1}/{args.runs}")
        run_stats[f"run_{i+1}"] = run_swm(model, args.n_boxes, n_tokens=args.n_tokens, cot=args.cot)
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
    
    tot_score = 0
    for stats in run_stats.values():
        tot_score += score(stats)
    print(f'Score: {tot_score / len(run_stats.keys())}')