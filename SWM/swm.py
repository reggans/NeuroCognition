from huggingface_hub import login
import transformers
import torch
import numpy as np
from tqdm.auto import tqdm

import random, json, re, argparse, os
import string
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..shared.model_wrapper import ModelWrapper
except ImportError:
    from shared.model_wrapper import ModelWrapper
from .image import SWMImage


def image_swm(
    model,
    n_boxes,
    n_tokens=1,
    cot=None,
    think_budget=64,
    note_assist=False,
    image_only=False,
):
    if note_assist:
        raise NotImplementedError

    # Initiate w/ task prompt
    task_prompt = f"""You will be performing the Spatial Working Memory task. 
You will be given an image containing {n_boxes} yellow boxes in a grid. 
There are {n_tokens} types of tokens, hidden in any one of {n_boxes} boxes.
Each token type is represented by a distinct color.
Your goal is to find the {n_tokens} types of tokens {n_boxes} times each, by repeatedly selecting a box to open.
A box can contain multiple types of tokens, but only one token of each type.
If the box contains multiple tokens, a token with mixed colors corresponding to the tokens will be shown.
Once the token is found, another will be generated in another box. 
The token will be generated in a box that has never contained a token of that type before in the trial. 
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

    # Initialize image generator
    os.makedirs("SWM/images", exist_ok=True)
    swm_gen = SWMImage("SWM/images", n_boxes)

    # Initialize run statistics & variables
    tokens = [swm_gen.token_colors[i] for i in range(n_tokens)]
    legal_boxes = dict.fromkeys(tokens)
    for token in tokens:
        legal_boxes[token] = [x for x in range(1, n_boxes + 1)]

    worst_case_n = n_boxes**2
    total_guess = 0
    illegal_guess = 0
    invalid_guess = 0
    repeated_guess = 0
    nobox_guess = 0
    valid_guess = 0

    run_history = []

    # Start the test
    response = model.send_message(question, cot=cot)
    with tqdm(total=worst_case_n, desc="Total guesses") as guess_bar:
        with tqdm(total=n_boxes * n_tokens, desc="Tokens") as token_bar:
            token_box = dict.fromkeys(tokens)
            for token in tokens:
                token_box[token] = random.choice(legal_boxes[token])
                # tqdm.write(f"Token {token} put in box {token_box[token]}")
            found_tokens = []

            while True:
                for token in found_tokens:
                    if len(legal_boxes[token]) == 0:
                        token_box[token] = None
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
                    with open("data/temp_history.json", "w") as f:
                        json.dump(model.history, f, indent=4)

                    if total_guess >= worst_case_n:
                        break

                    # Note-taking assistance
                    notes = ""
                    if note_assist:
                        for token, legal in legal_boxes.items():
                            notes += f"Boxes that has contained token {token}: "
                            for box in range(1, n_boxes + 1):
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
                        chosen_coord = re.search(r"<answer>(?s:.*)</answer>", response)[
                            0
                        ]
                        chosen_coord = re.sub(
                            r"<answer>|</answer>", "", chosen_coord
                        ).strip()
                        try:
                            chosen_coord = re.findall(r"[0-9]+", chosen_coord)
                            chosen_coord = (int(chosen_coord[0]), int(chosen_coord[1]))
                        except IndexError:
                            run_history.append(
                                {
                                    "token_box": [
                                        swm_gen.get_box_coord(token_box[t])
                                        for t in tokens
                                    ],
                                    "chosen_coord": None,
                                    "found": False,
                                    "status": "invalid",
                                    "raw_response": response,
                                }
                            )

                            response = model.send_message(
                                f"Please answer with a valid grid coordinate (x, y).\n"
                                + msg
                                + notes
                                + question,
                                truncate_history=True,
                                cot=cot,
                            )
                            invalid_guess += 1
                            continue
                        try:
                            chosen_box = swm_gen.get_box_id(chosen_coord)
                        except ValueError:
                            run_history.append(
                                {
                                    "token_box": [
                                        swm_gen.get_box_coord(token_box[t])
                                        for t in tokens
                                    ],
                                    "chosen_coord": chosen_coord,
                                    "found": False,
                                    "status": "nobox",
                                    "raw_response": response,
                                }
                            )

                            response = model.send_message(
                                f"No box in grid coordinate (x, y).\n"
                                + msg
                                + notes
                                + question,
                                truncate_history=True,
                                cot=cot,
                            )

                            nobox_guess += 1
                            continue
                    else:
                        run_history.append(
                            {
                                "token_box": [
                                    swm_gen.get_box_coord(token_box[t]) for t in tokens
                                ],
                                "chosen_coord": None,
                                "found": False,
                                "status": "invalid",
                                "raw_response": response,
                            }
                        )

                        response = model.send_message(
                            f"Please answer with the specified format\n"
                            + msg
                            + notes
                            + question,
                            truncate_history=True,
                            cot=cot,
                        )
                        invalid_guess += 1
                        continue

                    swm_gen.open_box(chosen_coord, token_box)

                    legal = False
                    for legal_list in legal_boxes.values():
                        if chosen_box in legal_list:
                            legal = True
                            break
                    if not legal:
                        illegal_guess += 1
                    elif chosen_box in opened_boxes:
                        repeated_guess += 1
                    else:
                        valid_guess += 1

                    opened_boxes.add(chosen_box)

                    for token in tokens:
                        if token_box[token] is not None and chosen_box == token_box[token]:
                            found = True
                            token_bar.update(1)
                            legal_boxes[token].remove(chosen_box)
                            found_tokens.append(token)

                    msg = ""
                    if found:
                        for token in found_tokens:
                            msg = f"Token {token} found in box {chosen_coord}.\n" + msg
                    else:
                        msg += f"No tokens found in box {chosen_coord}.\n" + msg
                    run_history.append(
                        {
                            "token_box": [
                                swm_gen.get_box_coord(token_box[t]) for t in tokens
                            ],
                            "chosen_coord": chosen_coord,
                            "found": found,
                            "status": "box",
                            "raw_response": response,
                        }
                    )

                    response = model.send_message(
                        msg + notes + question,
                        truncate_history=True,
                        cot=cot,
                        image_only=image_only,
                    )

                    if not image_only:
                        model.history[-2]["content"][0][
                            "text"
                        ] = msg  # Truncate user response length
                    
                    total_guess += 1
                    guess_bar.update(1)

    run_stats = {
        "worst_case_guesses": worst_case_n,
        "illegal": illegal_guess,
        "guesses": total_guess,
        "invalid": invalid_guess,
        "repeated": repeated_guess,
        "nobox": nobox_guess,
        "valid": valid_guess,
    }

    return run_stats, run_history


def text_swm(model, n_boxes, n_tokens=1, cot=None, think_budget=64, note_assist=False):
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
        legal_boxes[token] = [x for x in range(1, n_boxes + 1)]

    worst_case_n = n_boxes**2
    total_guess = 0
    illegal_guess = 0
    invalid_guess = 0
    repeated_guess = 0
    nobox_guess = 0
    valid_guess = 0

    run_history = []

    # Start the test
    response = model.send_message(question, cot=cot)
    with tqdm(total=worst_case_n, desc="Total guesses") as guess_bar:
        with tqdm(total=n_boxes * n_tokens, desc="Tokens") as token_bar:
            token_box = dict.fromkeys(tokens)
            for token in tokens:
                token_box[token] = random.choice(legal_boxes[token])
                # tqdm.write(f"Token {token} put in box {token_box[token]}")
            found_tokens = []

            while True:
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
                    notes = ""
                    if note_assist:
                        for token, legal in legal_boxes.items():
                            notes += f"Boxes that has contained token {token}: "
                            for box in range(1, n_boxes + 1):
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
                        chosen_box = re.sub(
                            r"<answer>|</answer>", "", chosen_box
                        ).strip()
                        try:
                            chosen_box = int(chosen_box)
                        except ValueError:
                            run_history.append(
                                {
                                    "token_box": [token_box[t] for t in tokens],
                                    "chosen_box": None,
                                    "found": False,
                                    "status": "invalid",
                                    "raw_response": response,
                                }
                            )

                            response = model.send_message(
                                f"Please answer with a box number (1-{n_boxes}).\n"
                                + msg
                                + notes
                                + question,
                                truncate_history=True,
                                cot=cot,
                            )
                            invalid_guess += 1
                            continue
                    else:
                        run_history.append(
                            {
                                "token_box": [token_box[t] for t in tokens],
                                "chosen_box": None,
                                "found": False,
                                "status": "invalid",
                                "raw_response": response,
                            }
                        )

                        response = model.send_message(
                            f"Please answer with the specified format\n"
                            + msg
                            + notes
                            + question,
                            truncate_history=True,
                            cot=cot,
                        )
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
                    else:
                        valid_guess += 1

                    opened_boxes.add(chosen_box)

                    for token in tokens:
                        if chosen_box == token_box[token]:
                            found = True
                            token_bar.update(1)
                            legal_boxes[token].remove(chosen_box)
                            found_tokens.append(token)

                    msg = ""
                    if found:
                        for token in found_tokens:
                            msg = f"Token {token} found in box {chosen_box}.\n" + msg
                    else:
                        msg += f"No tokens found in box {chosen_box}.\n" + msg

                    run_history.append(
                        {
                            "token_box": [token_box[t] for t in tokens],
                            "chosen_box": chosen_box,
                            "found": found,
                            "status": "valid",
                            "raw_response": response,
                        }
                    )

                    response = model.send_message(
                        msg + notes + question, truncate_history=True, cot=cot
                    )
                    model.history[-2]["content"] = msg  # Truncate user response length

    run_stats = {
        "worst_case_guesses": worst_case_n,
        "illegal": illegal_guess,
        "guesses": total_guess,
        "invalid": invalid_guess,
        "repeated": repeated_guess,
        "nobox": nobox_guess,
        "valid": valid_guess,
    }

    return run_stats, run_history


def score(run_stats):
    """
    Score the run statistics from the SWM test.
    Args:
        run_stats (dict): The run statistics.
    Returns:
        float: The score.
    """
    return 1 - (run_stats["illegal"] + run_stats["repeated"] + run_stats["nobox"]) / (
        run_stats["guesses"] - run_stats["invalid"]
    )
