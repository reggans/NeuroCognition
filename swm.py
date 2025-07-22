from huggingface_hub import login
import transformers
import torch
import numpy as np
from tqdm.auto import tqdm

import random, json, re, argparse, os
import string

from model_wrapper import ModelWrapper
from image import SWMImage

def image_swm(model, n_boxes, n_tokens=1, cot=None, think_budget=64, note_assist=False):
    if n_tokens > 1 or note_assist: 
        raise NotImplementedError
    
    # Initiate w/ task prompt
    task_prompt = f"""You will be performing the Spatial Working Memory task. 
You will be given an image containing 8 yellow boxes in a grid. 
One of the boxes contains a red token. 
Your goal is to find the token 8 times by repeatedly selecting a box to open. 
Once the token is found, another will be generated in another box. 
The token will be generated in a box that has never contained the token before in the trial. 
The token may be generated in a box that has been opened and found empty before, as long as it never contained that type of token previously. 
Your final answer should be a coordinate (x, y), the grid coordinate of the box you choose.
"""
    model.init_chat(task_prompt)

    # Configure the question presented each turn and CoT prompt
    if cot is not None:
        cot_prompt = f"Think step-by-step, utilizing information from previous feedbacks, and state your reasoning in maximum {think_budget} tokens, wrapped with <think> and </think>. Then, provide a really short summary of your reasoning after the closing </think> tag.\n"
        question = f"Answer concisely. {cot_prompt}Which of the {n_boxes} boxes would you like to open?\nYour final answer should be a grid coordinate (x, y), wrapped with <answer> and </answer>"
    else:
        question = f"Answer only with your final answer. Which of the {n_boxes} boxes would you like to open?\nYour final answer should be a grid coordinate (x, y), wrapped with <answer> and </answer>"

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

    os.makedirs("images", exist_ok=True)
    swm_gen = SWMImage("images", n_boxes)

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
                        chosen_coord = re.search(r"<answer>(?s:.*)</answer>", response)[0]
                        chosen_coord = re.sub(r"<answer>|</answer>", "", chosen_coord).strip()
                        try:
                            chosen_coord = re.findall(r"[0-9]*", chosen_coord)
                            chosen_coord = (int(chosen_coord[0]), int(chosen_coord[1])) 
                            chosen_box = swm_gen.get_box_id(chosen_coord)
                        except ValueError:
                            response = model.send_message(f"Please answer with a valid grid coordinate (x, y).\n" + msg + notes + question, truncate_history=True, cot=cot)
                            invalid_guess += 1
                            continue
                    else:
                        response = model.send_message(f"Please answer with the specified format\n" + msg + notes + question, truncate_history=True, cot=cot)
                        invalid_guess += 1
                        continue
                        
                    swm_gen.open_box(chosen_coord, token_box[tokens[0]])
                    
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