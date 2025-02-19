import random, argparse

parser = argparse.ArgumentParser(description="Run the SWM toy problem.")
parser.add_argument("--num_boxes", type=int, default=6, help="The number of boxes in the test. More == harder.")
args = parser.parse_args()

question = "Which box would you like to open?\n"

legal_boxes = [x for x in range(1, args.num_boxes + 1)]
worst_case_n = sum(legal_boxes)
n_guesses = []
for i in range(1, args.num_boxes + 1):
  n_guesses.append(0)
  token_box = random.choice(legal_boxes)
  legal_boxes.remove(token_box)
  
  found = False
  while not found:
    print(f"Found tokens: {i}")
    response = input(question).strip()
    chosen_box = int(response.split("ANSWER: ")[-1].strip())
    if chosen_box == token_box:
      print("TOKEN")
      found = True
    else:
      print("EMPTY")
    n_guesses[-1] += 1
print(f"Worst case guesses: {worst_case_n}")
print(f"Total number of guesses: {sum(n_guesses)}")