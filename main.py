import google.generativeai as genai
from huggingface_hub import login
import transformers
import torch
import numpy as np
from tqdm.auto import tqdm

import random, json, re, argparse, os

from model_wrapper import ModelWrapper

def run_swm(model, n_boxes, cot=None):
    if cot == "implicit":
        cot_prompt = "Think step-by-step, utilizing information from previous feedbacks, and state your reasoning clearly.\n"
    elif cot == "explicit":
        cot_prompt = cot_prompt + "Track the boxes where the tokens have been found before.\n"""
    else:
        raise ValueError("CoT must be None, or either of 'implicit' or 'explicit'.")
    
    if cot is not None:
        question = f"{cot_prompt}Which of the {n_boxes} boxes would you like to open?\nState your final answer by wrapping it with <ANS> and </ANS>"
    else:
        question = f"Which of the {n_boxes} boxes would you like to open?\nState your final answer by wrapping it with <ANS> and </ANS>"

    legal_boxes = [x for x in range(1, n_boxes+1)]
    worst_case_n = sum(legal_boxes)
    n_guesses = []
    illegal_guesses = []
    invalid_guesses = []


    response = model.send_message(question)
    with tqdm(total=3*worst_case_n, desc="Total guesses") as pbar:
        for i in tqdm(range(n_boxes), total=n_boxes, desc="Rounds"):
            n_guesses.append(0)
            illegal_guesses.append(0)
            invalid_guesses.append(0)

            chosen_box = None
            # TODO - If answer is invalid, wait for a valid first answer before re-choosing token box
            if len(legal_boxes) > 1:
                # Ensure no lucky first guess
                if re.search(r"<ANS>(?s:.*)</ANS>", response) is not None:
                    chosen_box = re.search(r"<ANS>(?s:.*)</ANS>", response)[0]
                    chosen_box = re.sub(r"<ANS>|</ANS>", chosen_box).strip()
                    try:
                        chosen_box = int(chosen_box)
                    except ValueError:
                        chosen_box = None
                        continue

                # Re-choose token box among non-selected boxes
                if chosen_box in legal_boxes:
                    legal_boxes.remove(chosen_box)
                    token_box = random.choice(legal_boxes)
                    legal_boxes.remove(token_box)
                    legal_boxes.append(chosen_box)
                else:
                    token_box = random.choice(legal_boxes)
                    legal_boxes.remove(token_box)

            if chosen_box is None:
                token_box = random.choice(legal_boxes)
                legal_boxes.remove(token_box)

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
    parser.add_argument("--seed", type=int, default=42, help="The random seed.")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = ModelWrapper(args.model)
    
    run
    
