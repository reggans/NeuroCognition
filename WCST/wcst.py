import transformers
import google
import torch
from tqdm.auto import tqdm

import json, argparse, random, time, os, re
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils import (
    generate_few_shot,
    wcst_generator,
    string_generator,
    check_rule_ambiguity,
)

try:
    from ..shared.model_wrapper import ModelWrapper
except ImportError:
    from shared.model_wrapper import ModelWrapper
from .image import draw_five_cards

wcst_prompt = """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you have to figure out.
The cards will be described by the following attributes:
1. Number of symbols
2. Color of symbols
3. Shape of symbols
4. Background color of the card

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, you have to figure out the correct rule to match the cards.
If you are correct, you have to stick with the same attribute until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.

"""
wcst_random_prompt = """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you have to figure out.
The cards will be described by the following attributes in a random order:
1. Number of symbols
2. Color of symbols
3. Shape of symbols
4. Background color of the card

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, you have to figure out the correct rule to match the cards.
If you are correct, you have to stick with the same attribute until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.

"""
random_prompt = """You are performing a modified version of the Wisconsin Card Sorting Test (WCST).
You will be shown a given string, and you have to match it with one of four option strings according to a rule that you have to figure out.
The rule is one of the following:
1. Length of the string
2. The number of vowels in the string
3. The number of consonants in the string

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, you have to figure out the correct rule to match the strings.
If you are correct, you have to stick with the same rule until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the string you think is the correct match.

"""

empty_prompt = """You are performing a modified version of the Wisconsin Card Sorting Test (WCST).
One option among 1, 2, 3, and 4 is the correct answer.
You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the correct answer has changed, you have to figure out the correct answer.
If you are correct, you have to stick with the same answer until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the answer you think is correct.

"""

image_prompt = """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you have to figure out.
The cards will have the following attributes:
1. Number of symbols
2. Color of symbols
3. Shape of symbols
4. Background color of the card

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed.
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, you have to figure out the correct rule to match the cards.
If you are correct, you have to stick with the same attribute until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.
"""


def run_wcst(
    model="llama",
    variant="card",
    max_trials=64,
    num_correct=5,
    bg_color=False,
    repeats=1,
    ambiguous_mode="off",
    few_shot=False,
    cot=False,
    hint=False,
    model_source="hf",
    max_tokens=512,
    think_budget=64,
    api_key=None,
    verbose=15,
):
    """
    Run the Wisconsin Card Sorting Test (WCST).

    Args:
        model: The model to use
        variant: The variant of the test ("card", "card-random", "string", "empty")
        max_trials: Maximum number of trials
        num_correct: Number of correct answers required per category
        repeats: Number of runs to perform
        ambiguous_mode: Control ambiguity in card generation ("off", "first", "rest")
        few_shot: Whether to use few-shot prompting
        cot: Whether to use chain-of-thought reasoning
        hint: Whether to provide hints
        model_source: The source of the model ("hf", "google", "litellm", "vllm")
        max_tokens: Maximum number of tokens to generate
        think_budget: Budget tokens for reasoning
        api_key: API key to use
        verbose: Verbosity level
    """
    print(f"few_shot: {few_shot}")

    os.makedirs(os.path.join("WCST", "data", "text"), exist_ok=True)

    save_path = os.path.join(
        "WCST",
        "data",
        "text",
        f"{model_source}_{model.replace('/', '-')}_{variant}_{max_trials}-{num_correct}.json",
    )

    if few_shot and cot:
        save_path = save_path.replace(".json", "_few_shot_cot.json")

    elif few_shot:
        save_path = save_path.replace(".json", "_few_shot.json")

    elif cot:
        save_path = save_path.replace(".json", "_cot.json")

    print(f"Saving to: {save_path}")

    # Check if results already exist
    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}")
        with open(save_path, "r") as f:
            save = json.load(f)

        # Calculate and display statistics for each run
        for run_key in save:
            save_rep = save[run_key]
            total_trials = len(save_rep)
            total_correct = sum(
                1 for row in save_rep if "correct" in row and row["correct"]
            )
            completed_categories = len(
                set(
                    row["rule"]
                    for row in save_rep
                    if "correct" in row and row["correct"]
                )
            )

            print(f"\n{run_key.title()} Statistics:")
            print(f"Completed categories: {completed_categories}")
            print(f"Total number of trials: {total_trials}")
            print(f"Total accuracy: {total_correct/total_trials:.3f}")
        return

    if variant == "card":
        system_prompt = wcst_prompt
        rules = ["color", "shape", "number", "background"]
    elif variant == "card-random":
        system_prompt = wcst_random_prompt
        rules = ["color", "shape", "number", "background"]
    elif variant == "string":
        system_prompt = random_prompt
        rules = ["length", "vowels", "consonants"]
    elif variant == "empty":
        system_prompt = empty_prompt
        rules = [1, 2, 3, 4]
        if few_shot:
            raise NotImplementedError
    else:
        raise Exception("Variant not recognized")

    if few_shot:
        system_prompt += generate_few_shot(variant)

    if cot:
        system_prompt += f"Explain your thought process regarding the problem and the feedbacks you received in maximum {think_budget} tokens wrapped with <think> and </think>. Then, provide a really short summary of your reasoning after the closing </think> tag.\n"
    else:
        system_prompt += "Answer only with your final answer.\n"
    system_prompt += """State your final answer using the template: "<answer>your answer</answer>"\n"""

    save = {}
    run_history = {}
    run_reasoning = {}

    for rep in range(repeats):
        model_instance = None
        torch.cuda.empty_cache()
        save_rep = []

        model_instance = ModelWrapper(
            model, model_source, api_key=api_key, max_new_tokens=max_tokens
        )
        model_instance.init_chat(system_prompt)

        n_trials = 0
        completed_cat = 0
        total_correct = 0
        correct_prefix = ""
        force_ambig = False

        with tqdm(total=max_trials, desc="Total trials") as trial_bar:
            for _ in range(2):
                for rule in rules:
                    correct_cnt = 0
                    force_ambig = True if ambiguous_mode == "first" else False

                    with tqdm(
                        total=num_correct, desc=f"Correct answers for {rule}"
                    ) as correct_bar:
                        while correct_cnt < num_correct:
                            if n_trials >= max_trials:
                                break

                            if variant == "card":
                                if ambiguous_mode != "off":
                                    given, opt = wcst_generator(rule, randomize=False, bg_color=bg_color, ambiguous=force_ambig)
                                    if ambiguous_mode == "rest":  # After the first non-ambiguous trial, keep all subsequent trials ambiguous
                                        force_ambig = True
                                    else:  # Only the first trial is ambiguous
                                        force_ambig = False
                                else:
                                    given, opt = wcst_generator(rule, randomize=False, bg_color=bg_color,)
                            elif variant == "card-random":
                                if ambiguous_mode != "off":
                                    given, opt = wcst_generator(rule, True, ambiguous=force_ambig)
                                    if ambiguous_mode == "rest":  # After the first non-ambiguous trial, keep all subsequent trials ambiguous
                                        force_ambig = True
                                    else:  # Only the first trial is ambiguous
                                        force_ambig = False
                                else:
                                    given, opt = wcst_generator(rule, True)
                            elif variant == "string":
                                given, opt = string_generator(rule)

                            if variant == "empty":
                                chosen = rule
                                chosen_idx = rule

                                test_prompt = f"""Options:\n1.\n2.\n3.\n4."""
                            else:
                                chosen = opt[0]
                                random.shuffle(opt)
                                chosen_idx = opt.index(chosen) + 1

                                test_prompt = f"""Given: {given}\nOptions:\n1. {opt[0]}\n2. {opt[1]}\n3. {opt[2]}\n4. {opt[3]}"""

                            # Add hint
                            if hint:
                                test_prompt += f"\nRule: {rule}"

                            correct = False
                            while not correct:
                                if n_trials >= max_trials:
                                    break
                                trial_bar.update(1)

                                n_trials += 1
                                response = model_instance.send_message(
                                    correct_prefix + test_prompt,
                                    truncate_history=True,
                                    cot=cot,
                                )
                                ans = re.search(r"<answer>(?s:.*)</answer>", response)
                                if ans:
                                    ans = re.search(
                                        r"<answer>(?s:.*)</answer>", response
                                    )[0]
                                    ans = re.sub(r"<answer>|</answer>", "", ans).strip()
                                    if ans == str(chosen_idx):
                                        correct_prefix = "Correct!\n"
                                        correct = True
                                        correct_cnt += 1
                                        total_correct += 1
                                        correct_bar.update(1)
                                    else:
                                        correct_prefix = (
                                            "Incorrect. Please try again.\n"
                                        )
                                        correct_cnt = 0
                                        correct_bar.n = 0
                                        correct_bar.last_print_n = 0
                                        correct_bar.refresh()
                                else:
                                    correct_prefix = """Answer not found. Please state your final answer using the template: \"<answer>your answer</answer>\""""
                                    correct_cnt = 0
                                    correct_bar.n = 0
                                    correct_bar.last_print_n = 0
                                    correct_bar.refresh()

                                if n_trials % 50 == 0:
                                    tqdm.write(f"Rule: {rule}")
                                    tqdm.write(test_prompt)
                                    tqdm.write(response)

                                save_row = {
                                    "rule": rule,
                                    "correct": correct,
                                    "correct_prefix": correct_prefix,
                                    "question": test_prompt,
                                    "response": response,
                                    "model_ans": ans,
                                    "true_ans": chosen_idx,
                                }
                                
                                # Add ambiguity information for card variants
                                if variant in ["card", "card-random"] and ambiguous_mode != "off":
                                    try:
                                        save_row["ambiguous"] = check_rule_ambiguity(
                                            given, opt[chosen_idx - 1], bg_color=bg_color
                                        )
                                    except:
                                        save_row["ambiguous"] = None
                                        
                                save_rep.append(save_row)

                    if correct_cnt == num_correct:
                        completed_cat += 1

        print(f"Completed categories: {completed_cat}")
        print(f"Total number of trials: {n_trials}")
        print(f"Total accuracy: {total_correct/n_trials}")

        save[f"run_{rep+1}"] = save_rep
        run_history[f"run_{rep+1}"] = model_instance.history
        run_reasoning[f"run_{rep+1}"] = model_instance.reasoning_trace

        with open(save_path, "w") as f:
            json.dump(save, f, indent=4)

        with open(save_path.replace(".json", "_history.json"), "w") as f:
            json.dump(run_history, f, indent=4)

        with open(save_path.replace(".json", "_reasoning.json"), "w") as f:
            json.dump(run_reasoning, f, indent=4)


def run_wcst_image(
    model="llama",
    max_trials=64,
    num_correct=5,
    repeats=1,
    bg_color=False,
    ambiguous_mode="off",
    few_shot=False,
    cot=False,
    hint=False,
    model_source="hf",
    max_tokens=512,
    think_budget=64,
    api_key=None,
    verbose=15,
):
    """
    Run the Wisconsin Card Sorting Test (WCST) with visual card images.

    Args:
        model: The model to use
        max_trials: Maximum number of trials
        num_correct: Number of correct answers required per category
        repeats: Number of runs to perform
        few_shot: Whether to use few-shot prompting
        cot: Whether to use chain-of-thought reasoning
        hint: Whether to provide hints
        model_source: The source of the model ("hf", "google", "litellm", "vllm")
        max_tokens: Maximum number of tokens to generate
        think_budget: Budget tokens for reasoning
        api_key: API key to use
        verbose: Verbosity level
    """
    print(f"Running image WCST with model: {model}")
    print(f"few_shot: {few_shot}")

    os.makedirs(os.path.join("WCST", "data", "image"), exist_ok=True)
    os.makedirs(os.path.join("WCST", "images"), exist_ok=True)

    save_path = os.path.join(
        "WCST",
        "data",
        "image",
        f"{model_source}_{model.replace('/', '-')}_image_{max_trials}-{num_correct}{'-bg' if bg_color else ''}_{ambiguous_mode}.json",
    )

    if few_shot and cot:
        save_path = save_path.replace(".json", "_few_shot_cot.json")
    elif few_shot:
        save_path = save_path.replace(".json", "_few_shot.json")
    elif cot:
        save_path = save_path.replace(".json", "_cot.json")

    print(f"Saving to: {save_path}")

    # Check if results already exist
    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}")
        with open(save_path, "r") as f:
            save = json.load(f)

        # Calculate and display statistics for each run
        for run_key in save:
            save_rep = save[run_key]
            total_trials = len(save_rep)
            total_correct = sum(
                1 for row in save_rep if "correct" in row and row["correct"]
            )
            completed_categories = len(
                set(
                    row["rule"]
                    for row in save_rep
                    if "correct" in row and row["correct"]
                )
            )

            print(f"\n{run_key.title()} Statistics:")
            print(f"Completed categories: {completed_categories}")
            print(f"Total number of trials: {total_trials}")
            print(f"Total accuracy: {total_correct/total_trials:.3f}")
        return

    system_prompt = image_prompt
    if not bg_color:
        system_prompt = system_prompt.replace("4. Background color of the card", "")

    if few_shot:
        system_prompt += generate_few_shot("card")

    if cot:
        system_prompt += f"Explain your thought process regarding the problem and the feedbacks you received in maximum {think_budget} tokens wrapped with <think> and </think>. Then, provide a really short summary of your reasoning after the closing </think> tag.\n"
    else:
        system_prompt += "Answer only with your final answer.\n"
    system_prompt += """State your final answer using the template: "<answer>your answer</answer>"\n"""

    rules = ["color", "shape", "number"]
    if bg_color:
        rules.append("background")

    save = {}
    run_history = {}
    run_reasoning = {}

    for rep in range(repeats):
        model_instance = None
        torch.cuda.empty_cache()
        save_rep = []

        # Initialize model with image input capability
        model_instance = ModelWrapper(
            model,
            model_source,
            api_key=api_key,
            max_new_tokens=max_tokens,
            image_input=True,
            image_path="WCST/images/",
        )
        model_instance.init_chat(system_prompt)

        n_trials = 0
        completed_cat = 0
        total_correct = 0
        correct_prefix = ""
        force_ambig = False

        with tqdm(total=max_trials, desc="Total trials") as trial_bar:
            for _ in range(2):  # Twice per rule
                for rule in rules:
                    correct_cnt = 0
                    force_ambig = True if ambiguous_mode == "first" else False

                    with tqdm(
                        total=num_correct, desc=f"Correct answers for {rule}"
                    ) as correct_bar:
                        while correct_cnt < num_correct:
                            if n_trials >= max_trials:
                                break

                            # Generate card attributes for the given card
                            if ambiguous_mode != "off":
                                given_attrs, option_cards = wcst_generator(
                                    rule,
                                    False,
                                    bg_color=bg_color,
                                    ambiguous=force_ambig,
                                )
                                if (
                                    ambiguous_mode == "rest"
                                ):  # After the first non-ambiguous trial, keep all subsequent trials ambiguous
                                    force_ambig = True
                                else:  # Only the first trial is ambiguous
                                    force_ambig = False
                            else:
                                given_attrs, option_cards = wcst_generator(
                                    rule, False, bg_color=bg_color
                                )  # Ambiguity not regulated

                            # Convert text representation to image attributes
                            given_card_attrs = parse_card_attributes(
                                given_attrs, bg_color=bg_color
                            )

                            # Generate the 5-card image
                            draw_five_cards(given_card_attrs, bg_color=bg_color)

                            # Find the correct answer (which of cards 1-4 matches the rule)
                            correct_card_idx = find_matching_card(
                                given_card_attrs, rule
                            )

                            test_prompt = "Look at the image showing 5 cards. Match the 'Given' card to one of cards 1-4 based on the rule you need to figure out."

                            # Add hint if enabled
                            if hint:
                                test_prompt += f"\nRule: {rule}"

                            correct = False
                            while not correct:
                                if n_trials >= max_trials:
                                    break
                                trial_bar.update(1)

                                n_trials += 1
                                response = model_instance.send_message(
                                    correct_prefix + test_prompt,
                                    truncate_history=True,
                                    cot=cot,
                                )
                                ans = re.search(r"<answer>(?s:.*)</answer>", response)
                                if ans:
                                    ans = re.search(
                                        r"<answer>(?s:.*)</answer>", response
                                    )[0]
                                    ans = re.sub(r"<answer>|</answer>", "", ans).strip()
                                    if ans == str(correct_card_idx):
                                        correct_prefix = "Correct!\n"
                                        correct = True
                                        correct_cnt += 1
                                        total_correct += 1
                                        correct_bar.update(1)
                                    else:
                                        correct_prefix = (
                                            "Incorrect. Please try again.\n"
                                        )
                                        correct_cnt = 0
                                        correct_bar.n = 0
                                        correct_bar.last_print_n = 0
                                        correct_bar.refresh()
                                else:
                                    correct_prefix = """Answer not found. Please state your final answer using the template: \"<answer>your answer</answer>\""""
                                    correct_cnt = 0
                                    correct_bar.n = 0
                                    correct_bar.last_print_n = 0
                                    correct_bar.refresh()

                                if n_trials % 50 == 0:
                                    tqdm.write(f"Rule: {rule}")
                                    tqdm.write(f"Given card: {given_card_attrs}")
                                    tqdm.write(f"Correct answer: {correct_card_idx}")
                                    tqdm.write(response)

                                save_row = {
                                    "rule": rule,
                                    "correct": correct,
                                    "correct_prefix": correct_prefix,
                                    "given_card": given_card_attrs,
                                    "correct_card": correct_card_idx,
                                    "ambiguous": check_rule_ambiguity(
                                        given_attrs,
                                        option_cards[correct_card_idx - 1],
                                        bg_color=bg_color,
                                    ),
                                    "response": response,
                                    "model_ans": ans,
                                    "true_ans": correct_card_idx,
                                }
                                save_rep.append(save_row)

                    if correct_cnt == num_correct:
                        completed_cat += 1

        print(f"Completed categories: {completed_cat}")
        print(f"Total number of trials: {n_trials}")
        print(f"Total accuracy: {total_correct/n_trials}")

        save[f"run_{rep+1}"] = save_rep
        run_history[f"run_{rep+1}"] = model_instance.history
        run_reasoning[f"run_{rep+1}"] = model_instance.reasoning_trace

        with open(save_path, "w") as f:
            json.dump(save, f, indent=4)

        with open(save_path.replace(".json", "_history.json"), "w") as f:
            json.dump(run_history, f, indent=4)

        with open(save_path.replace(".json", "_reasoning.json"), "w") as f:
            json.dump(run_reasoning, f, indent=4)


def parse_card_attributes(card_description, bg_color=False):
    """
    Parse a card description string into attributes dict for image generation.

    Args:
        card_description: String like "2 green triangles" or "four green triangles red" (with bg color)
        bg_color: Whether to expect background color in the description

    Returns:
        dict: {'shape': 'triangle', 'color': 'green', 'count': 2, 'background': 'red'} (if bg_color=True)
    """

    # Helper function to convert word numbers to integers
    def word_to_int(word):
        word_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        return word_map.get(word.lower(), None)

    parts = card_description.split()

    # Try to parse the first part as an integer, if that fails try word form
    try:
        count = int(parts[0])
    except ValueError:
        count = word_to_int(parts[0])
        if count is None:
            raise ValueError(
                f"Could not parse count from '{parts[0]}' in card description: {card_description}"
            )

    color = parts[1]
    shape = parts[2].rstrip("s")  # Remove plural 's'

    # Map shape names to image.py format
    shape_map = {
        "triangle": "triangle",
        "circle": "circle",
        "star": "star",
        "cross": "square",  # Map cross to square since we use square in image.py
    }

    result = {"shape": shape_map.get(shape, shape), "color": color, "count": count}

    # Add background color if expected
    if bg_color and len(parts) > 3:
        result["background"] = parts[3]
    elif bg_color:
        # Default background if not specified
        result["background"] = "white"

    return result


def find_matching_card(given_attrs, rule):
    """
    Find which of the 4 reference cards matches the given card based on the rule.

    Args:
        given_attrs: dict with given card attributes
        rule: str, the matching rule ('color', 'shape', 'number', 'background')

    Returns:
        int: The card number (1-4) that matches the rule
    """
    # Define the 4 reference cards (same as in image.py)
    reference_cards = [
        {"shape": "circle", "color": "red", "count": 1, "background": "red"},  # Card 1
        {
            "shape": "triangle",
            "color": "green",
            "count": 2,
            "background": "green",
        },  # Card 2
        {"shape": "star", "color": "blue", "count": 3, "background": "blue"},  # Card 3
        {
            "shape": "square",
            "color": "yellow",
            "count": 4,
            "background": "yellow",
        },  # Card 4
    ]

    # Map rule names to attribute names
    rule_map = {
        "color": "color",
        "shape": "shape",
        "number": "count",
        "background": "background",
    }

    attribute = rule_map.get(rule, rule)
    given_value = given_attrs.get(attribute)

    if given_value is None:
        raise ValueError(
            f"Given card does not have attribute '{attribute}' for rule '{rule}'"
        )

    for i, ref_card in enumerate(reference_cards):
        if ref_card[attribute] == given_value:
            return i + 1  # Return 1-based index

    # Should not happen if cards are generated correctly
    raise ValueError(
        f"No matching card found for rule '{rule}' with value '{given_value}'. Given card: {given_attrs}"
    )
