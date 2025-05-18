from tqdm.auto import tqdm

import json, argparse, random, time, os, re

from utils import generate_few_shot, wcst_generator, string_generator
from model_wrapper import ModelWrapper

matching_card_prompt = """You will be shown a given card with a symbol on it, and you will have to match it to one of four option cards according to an attribute that you will be told.
The cards will be described by the following attributes:
1. Number of symbols
2. Color of symbols
3. Shape of symbols

Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.

"""

matching_string_prompt = """You will be shown a given string, and you have to match it with one of four option strings according to a rule that you have to figure out.
The rule is one of the following:
1. Length of the string
2. The number of vowels in the string
3. The number of consonants in the string

Your final answer should be a number between 1-4 corresponding to the index of the card you think is the correct match.

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--variant", type=str, default="card")
    # parser.add_argument("--max_trials", type=int, default=64)
    # parser.add_argument("--num_correct", type=int, default=5)
    # parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--model_source", type=str, default="hf", help="The source of the model.")
    parser.add_argument("--max_tokens", type=int, default=512, help="The maximum number of tokens to generate.")
    parser.add_argument("--think_budget", type=int, default=64, help="The budget tokens for reasoning.")
    parser.add_argument("--api_key", type=str, default=None, help="API key to use. If none, uses key stored in environment variable.")
    parser.add_argument("--verbose", type=int, default=15)
    
    args = parser.parse_args()
    variant = args.variant
    # max_trials = args.max_trials
    # num_correct = args.num_correct
    few_shot = args.few_shot
    cot = args.cot

    print(f"few_shot: {few_shot}")

    if not os.path.isdir("wcst_data"):
        os.mkdir("wcst_data")
    if not os.path.isdir("wcst_data/baselines"):
        os.mkdir("wcst_data/baselines")

    save_path = f"wcst_data/baselines/{args.model_source}_{args.model.replace('/', '-')}_{variant}.json"

    if few_shot:
        raise NotImplementedError

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
        with open(save_path, 'r') as f:
            save = json.load(f)
            
        # Calculate and display statistics for each run
        for run_key in save:
            save_rep = save[run_key]
            total_trials = len(save_rep)
            total_correct = sum(1 for row in save_rep if "correct" in row and row["correct"])
            completed_categories = len(set(row["rule"] for row in save_rep if "correct" in row and row["correct"]))
            
            print(f"\n{run_key.title()} Statistics:")
            print(f"Completed categories: {completed_categories}")
            print(f"Total number of trials: {total_trials}")
            print(f"Total accuracy: {total_correct/total_trials:.3f}")
        exit(0)
    
    if variant == "card":
        system_prompt = matching_card_prompt
        rules = ["color", "shape", "number"]
    elif variant == "card-random":
        raise NotImplementedError
    elif variant == "string":
        system_prompt = matching_string_prompt
        rules = ["length", "vowels", "consonants"]
    elif variant == "empty":
        raise NotImplementedError
    else:
        raise Exception("Variant not recognized")
    
    if cot:
        system_prompt += f"Explain your thought process regarding the problem and the feedbacks you received in maximum {args.think_budget} tokens wrapped with <think> and </think>. Then, provide a really short summary of your reasoning after the closing </think> tag.\n"
    else:
        system_prompt += "Answer only with your final answer.\n"
    system_prompt += """State your final answer using the template: "<answer>your answer</answer>"\n"""

    save = {r: [] for r in rules}
    run_history = []

    model = ModelWrapper(args.model, args.model_source, api_key=args.api_key, max_new_tokens=args.max_tokens)

    for rule in rules:
        for q in tqdm(range(args.queries), total=args.queries, desc=f"Total queries for {rule}"):
            model.init_chat(system_prompt)

            if variant == "card":
                given, opt = wcst_generator(rule, False)
            elif variant == "string":
                given, opt = string_generator(rule)
            
            chosen = opt[0]
            random.shuffle(opt)
            chosen_idx = opt.index(chosen) + 1

            test_prompt = f"""Given: {given}\nOptions:\n1. {opt[0]}\n2. {opt[1]}\n3. {opt[2]}\n4. {opt[3]}\nRule: {rule}"""
            run_history.append(
                {"role": "user", "content": test_prompt}
            )

            response = model.send_message(test_prompt, truncate_history=True, cot=cot)
            run_history.append(
                {"role": "model", "content": response}
            )
            ans = re.search(r"<answer>(?s:.*)</answer>", response)
            if ans:
                ans = re.search(r"<answer>(?s:.*)</answer>", response)[0]
                ans = re.sub(r"<answer>|</answer>", "", ans).strip()

                save[rule].append(ans == str(chosen_idx))
            else:
                save[rule].append(False)
    
    for r in rules:
        print(f"Acc for {rule}: {np.mean(save[r])}")
    print(f"Overall acc: {np.mean([x for x in save.values()])}")

    with open(save_path, "w") as f:
            json.dump(save, f, indent=4)
        
    with open(save_path.replace(".json", "_history.json"), "w") as f:
        json.dump(run_history, f, indent=4)