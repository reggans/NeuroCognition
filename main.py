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
    task_prompt = f"""You will be performing the Spatial Working Memory (SWM) test.
{n_boxes} boxes will be presented to you, one of which contains a token.
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
        question = f"{cot_prompt}Which of the {n_boxes} boxes would you like to open?\nState your final answer by wrapping it with <ANS> and </ANS>"
    else:
        question = f"Which of the {n_boxes} boxes would you like to open?\nState your final answer by wrapping it with <ANS> and </ANS>"

    # Initialize run statistics & variables
    legal_boxes = [x for x in range(1, n_boxes+1)]
    worst_case_n = sum(legal_boxes)
    n_guesses = []
    illegal_guesses = []
    invalid_guesses = []

    # Start the test
    response = model.send_message(question)
    with tqdm(total=3*worst_case_n, desc="Total guesses") as pbar:
        for i in tqdm(range(n_boxes), total=n_boxes, desc="Rounds"):
            n_guesses.append(0)
            illegal_guesses.append(0)
            invalid_guesses.append(0)

            # TODO - If answer is invalid, wait for a valid first answer before re-choosing token box
            # Ensure no lucky first guess
            if len(legal_boxes) > 1: 
                if re.search(r"<ANS>(?s:.*)</ANS>", response) is not None:
                    chosen_box = re.search(r"<ANS>(?s:.*)</ANS>", response)[0]
                    chosen_box = re.sub(r"<ANS>|</ANS>", "", chosen_box).strip()
                    try:
                        chosen_box = int(chosen_box)
                    except ValueError:
                        chosen_box = None
                        continue

                # Re-choose token box among non-selected boxes
                if chosen_box is None or not chosen_box in legal_boxes: # Invalid or wrong answer
                    token_box = random.choice(legal_boxes)
                    legal_boxes.remove(token_box)
                else:                                                   # Potentially correct answer, generate different token box
                    legal_boxes.remove(chosen_box)
                    token_box = random.choice(legal_boxes)
                    legal_boxes.remove(token_box)
                    legal_boxes.append(chosen_box)
            else:                                                       # No choice but to choose the last box
                token_box = random.choice(legal_boxes)
                legal_boxes.remove(token_box)

            # tqdm.write(f"Round {i+1}")
            # tqdm.write(f"Answer: Box {token_box}")
            # tqdm.write(f"Legal boxes: {legal_boxes}")

            found = False
            while not found:
                n_guesses[-1] += 1
                pbar.update(1)

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

                if chosen_box == token_box:
                    response = model.send_message(f"TOKEN\nBox {chosen_box} contains a token.\nTokens found: {i+1}\n" + question)
                    found = True
                else:
                    if chosen_box not in legal_boxes:
                        illegal_guesses[-1] += 1
                    response = model.send_message(f"EMPTY\nBox {chosen_box} is empty.\nTokens found: {i}\n" + question)

                if sum(n_guesses) > 3*worst_case_n:
                    break

            if sum(n_guesses) > 3*worst_case_n:
                break
    
    run_stats = {
        "total_guesses": sum(n_guesses),
        "total_illegal": sum(illegal_guesses),
        "total_invalid": sum(invalid_guesses),
        "worst_case_guesses": worst_case_n,
        "illegal": illegal_guesses,
        "guesses": n_guesses,
        "invalid": invalid_guesses
    }

    return run_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SWM toy problem.")
    parser.add_argument("--model", type=str, default="gemini", help="The model to use.")
    parser.add_argument("--n_boxes", type=int, default=6, help="The number of boxes in the test. More == harder.")
    parser.add_argument("--cot", type=str, default=None, help="The type of CoT to use.")
    parser.add_argument("--runs", type=int, default=1, help="The number of runs to perform.")
    # parser.add_argument("--seed", type=int, default=42, help="The random seed.")
    parser.add_argument("--api_key", type=str, default=None, help="API key to use. If none, uses key stored in environment variable.")
    args = parser.parse_args()
    
    run_stats = {}
    run_history = {}
    for i in range(args.runs):
        model = ModelWrapper(args.model, api_key=args.api_key)
        torch.cuda.empty_cache()

        print(f"Run {i+1}/{args.runs}")
        run_stats[f"run_{i+1}"] = run_swm(model, args.n_boxes, cot=args.cot)
        run_history[f"run_{i+1}"] = model.history


    avg_stats = {}
    for key in run_stats["run_1"].keys():
        avg_stats[key] = np.mean([stats[key] for stats in run_stats.values()])
    
    for key, value in avg_stats.items():
        print(f"{key}: {value}")
    
    with open("run_stats.json", "w") as f:
        json.dump(run_stats, f)
    
    with open("run_history.json", "w") as f:
        json.dump(model.history, f)