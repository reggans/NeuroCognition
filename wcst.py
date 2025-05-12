import transformers
import google.generativeai as genai
import google
import torch
from tqdm.auto import tqdm

import json, argparse, random, time, os

from utils import generate_few_shot, wcst_generator, string_generator
from model_wrapper import ModelWrapper

wcst_prompt = """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you have to figure out.
The cards will be described by the following attributes:
1. Number of symbols
2. Color of symbols
3. Shape of symbols

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, you have to figure out the correct rule to match the cards.
If you are correct, you have to stick with the same attribute until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
You have to answer within 300 tokens.
Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.
State your final answer using the template: "<answer><index></answer>"

"""
wcst_random_prompt = """You are performing the Wisconsin Card Sorting Test (WCST).
You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you have to figure out.
The cards will be described by the following attributes in a random order:
1. Number of symbols
2. Color of symbols
3. Shape of symbols

You will be told "Correct!" if you are correct and "Incorrect. Please try again." if you are incorrect.
If you are incorrect, you either made a mistake or the rule has changed. 
If you believe you have made a mistake, correct it and try again.
If you believe the rule has changed, you have to figure out the correct rule to match the cards.
If you are correct, you have to stick with the same attribute until you are incorrect.
There is always a true answer in the task, and you have to keep performing the task until the end of the test.
You have to answer within 300 tokens.
Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.
State your final answer using the template: "<answer><index></answer>"

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
You have to answer within 300 tokens.
Your final answer should be a number between 1-4 corresponding to the index of the string you think is the correct match.
State your final answer using the template: "<answer><index></answer>"

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--variant", type=str, default="card")
    parser.add_argument("--max_trials", type=int, default=64)
    parser.add_argument("--num_correct", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--model_source", type=str, default="hf", help="The source of the model.")
    parser.add_argument("--api_key", type=str, default=None, help="API key to use. If none, uses key stored in environment variable.")
    parser.add_argument("--verbose", type=int, default=15)
    
    args = parser.parse_args()
    variant = args.variant
    max_trials = args.max_trials
    num_correct = args.num_correct
    few_shot = args.few_shot
    cot = args.cot

    if not os.path.isdir("./wcst_data"):
        os.mkdir("./wcst_data")

    save_path = f"wcst_data/{args.model_source}_{args.model.replace('/', '-')}{'_cot' if args.cot is not None else ''}_{variant}_{max_trials}-{num_correct}"
    print(f"Saving to: {save_path}")

    if variant == "card":
        system_prompt = wcst_prompt
        rules = ["color", "shape", "number"]
    elif variant == "card-random":
        system_prompt = wcst_random_prompt
        rules = ["color", "shape", "number"]
    elif variant == "string":
        system_prompt = random_prompt
        rules = ["length", "vowels", "consonants"]
    else:
        raise Exception("Variant not recognized")

    if few_shot:
        system_prompt += generate_few_shot(variant)

    if cot:
        system_prompt += "Explain your thought process regarding the problem and the feedbacks you received.\n"

    if few_shot:
        save_path = save_path.replace(".json", "_few_shot.json")
    if cot:
        save_path = save_path.replace(".json", "_cot.json")

    save = []

    for rep in range(args.repeats):
        model = None
        torch.cuda.empty_cache()
        save_rep = []

        model = ModelWrapper(args.model, args.model_source, api_key=args.api_key)
        model.init_chat(system_prompt)

        n_trials = 0
        completed_cat = 0
        total_correct = 0
        correct_prefix = ""

        with tqdm(total=max_trials, desc="Total trials") as trial_bar:
            for _ in range(2):      
                for rule in rules:
                    correct_cnt = 0
                    
                    with tqdm(total=num_correct, desc=f"Correct answers for {rule}") as correct_bar:
                        while correct_cnt < num_correct:
                            if n_trials >= max_trials:
                                break
                            
                            if variant == "card":
                                given, opt = wcst_generator(rule, False)
                            elif variant == "card-random":
                                given, opt = wcst_generator(rule, True)
                            else:
                                given, opt = string_generator(rule)

                            chosen = opt[0]
                            random.shuffle(opt)
                            chosen_idx = opt.index(chosen) + 1
            
                            test_prompt = f"""Given: {given}\nOptions:\n1. {opt[0]}\n2. {opt[1]}\n3. {opt[2]}\n4. {opt[3]}"""
            
                            correct = False
                            while not correct:
                                if n_trials >= max_trials:
                                    break
                                trial_bar.update(1)
            
                                n_trials += 1
                                response = model.send_message(correct_prefix + test_prompt)
                                # ans = response.split("ANSWER: ")[-1].strip()

                                if len(response) != 1:
                                    correct_prefix = """Answer not found. Please state your final answer using the template: \"<answer><index></answer>\""""
                                    correct_cnt = 0
                                    correct_bar.n = 0
                                    correct_bar.last_print_n = 0
                                    correct_bar.refresh()

                                elif response == str(chosen_idx):
                                    correct_prefix = "Correct!\n"
                                    correct = True
                                    correct_cnt += 1
                                    total_correct += 1
                                    correct_bar.update(1)
                                else:
                                    correct_prefix = "Incorrect. Please try again.\n"
                                    correct_cnt = 0
                                    correct_bar.n = 0
                                    correct_bar.last_print_n = 0
                                    correct_bar.refresh()

                                # if n_trials % 15 == 0:
                                #     tqdm.write(f"Rule: {rule}")
                                #     tqdm.write(test_prompt)
                                    # tqdm.write(response)

                                save_row = {"rule": rule,
                                            "correct": correct,
                                            "question": test_prompt,
                                            "response": response,
                                            "true_ans": chosen_idx}
                                save_rep.append(save_row)
                    
                    if correct_cnt == num_correct:
                        completed_cat += 1

        print(f"Completed categories: {completed_cat}")
        print(f"Total number of trials: {n_trials}")
        print(f"Total accuracy: {total_correct/n_trials}")
    
        save.append(save_rep)
        with open(save_path, "w") as f:
            json.dump(save, f, indent=4)