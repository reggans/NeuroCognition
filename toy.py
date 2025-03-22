import random, argparse

parser = argparse.ArgumentParser(description="Run the SWM toy problem.")
parser.add_argument("--num_boxes", type=int, default=6, help="The number of boxes in the test. More == harder.")
args = parser.parse_args()

question = "Which box would you like to open?\n"

legal_boxes = [x for x in range(1, args.num_boxes+1)]
worst_case_n = sum(legal_boxes)
n_guesses = []
for i in range(args.num_boxes):
  n_guesses.append(0)
  token_box = -1

  opened_boxes = []
  
  found = False
  while not found:
    print(f"Found tokens: {i}")
    response = input(question).strip()
    response = int(response)
    
    # Track opened boxes in this round
    opened_boxes.append(response)

    # If chosen box is the last legal box, it contains a token
    # if len(legal_boxes.intersection(opened_boxes)) == len(legal_boxes):
    #   found = True
    #   legal_boxes.remove(response)
    #   print("FOUND!")
    # else:
    #   print("EMPTY!")

    if len(opened_boxes) == 1 and token_box == -1:
      if len(legal_boxes) == 1:
        token_box = legal_boxes[0]
      else:
        legal_boxes.remove(response)
        token_box = random.choice(legal_boxes)
        legal_boxes.append(response)

    if response == token_box:
      print("FOUND!")
      legal_boxes.remove(token_box)
      found = True
    else:
      print("EMPTY!")
    
    n_guesses[-1] += 1
print(f"Worst case guesses: {worst_case_n}")
print(f"Total number of guesses: {sum(n_guesses)}")