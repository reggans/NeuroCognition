import random, string

def wcst_generator(rule, randomize=False, bg_color=False, ambiguous=None):
    rules = ["number", "color", "shape"]
    if bg_color:
        rules.append("background")

    if rule not in rules:
        raise Exception("Rule not recognized")
    
    if bg_color:
        options = [{"number": "one", "color": "red", "shape": "circle", "background": "red"},
                {"number": "two", "color": "green", "shape": "triangle", "background": "green"},
                {"number": "three", "color": "blue", "shape": "star", "background": "blue"},
                {"number": "four", "color": "yellow", "shape": "square", "background": "yellow"},
                ]
    else:
        options = [{"number": "one", "color": "red", "shape": "circle"},
                {"number": "two", "color": "green", "shape": "triangle"},
                {"number": "three", "color": "blue", "shape": "star"},
                {"number": "four", "color": "yellow", "shape": "square"},
                ]
    
    ans = random.choice(options)

    match_attr = ans[rule]
    addtnl_attr = None
    if ambiguous:
        addtnl_attr = random.choice([attr for attr in ans.items() if attr[0] != rule])  # Additional rule-value for ambiguity

    card_set = [card for card in options if card[rule] == match_attr] # Correct choice

    options.remove(ans)
    random.shuffle(options)
    card_set.extend(options)

    given_card = {}
    for r in rules:
        if r == rule:
            given_card[r] = match_attr
        elif addtnl_attr and r == addtnl_attr[0]:
            given_card[r] = addtnl_attr[1]
        elif not ambiguous:
            given_card[r] = random.choice([card[r] for card in options if card[r] != match_attr])
        else:
            given_card[r] = random.choice([card[r] for card in options])
    
    if randomize:
        card_set = [random.sample(list(card.values()), k=len(card.values())) for card in card_set]
    else:
        card_set = [list(card.values()) for card in card_set]

    card_set = [' '.join(card) for card in card_set]
    given_card = ' '.join(list(given_card.values()))

    return given_card, card_set

def string_generator(rule, max_given_length=10):
    letters = string.ascii_letters
    vowels = "aeiouAEIOU"
    consonants = ''.join([char for char in letters if char not in vowels])

    # odd numbers only to avoid equal number of vowels and consonants
    given_length = random.choice(range(3,  max_given_length, 2)) 
    given_string = ''.join(random.choices(letters, k=given_length))

    string_set = []
    if rule == "length":
        length = len(given_string)
        
        chosen = ''
        while chosen == given_string or chosen == '':
            chosen = ''.join(random.choices(letters, k=length))
        string_set.append(chosen)

        for _ in range(3):
            len_range = list(range(1,  int(given_length*1.5)))
            len_range.remove(length)
            lure_length = random.choice(len_range)

            lure = ''
            while (lure == given_string or
                  lure in string_set or
                  lure == ''):
                lure = ''.join(random.choices(letters, k=lure_length))
            string_set.append(lure)
            
    elif rule == "vowels":
        cnt = count_vowels(given_string)

        # odd numbers only to avoid equal number of vowels and consonants
        min_length = max(cnt, int(given_length*0.5))
        if min_length % 2 == 0:
            min_length += 1
        length = random.choice(range(min_length, int(given_length*1.5), 2))
        
        chosen = ''
        while chosen == given_string or chosen == '':

            chosen = ''.join(random.choices(vowels, k=cnt))
            chosen += ''.join(random.choices(consonants, k=length-cnt))
            char_list = list(chosen)
            random.shuffle(char_list)
            chosen = ''.join(char_list)
        string_set.append(chosen)

        for _ in range(3):
            length = random.choice(range(min_length, int(given_length*1.5), 2)) 
            
            cnt_range = list(range(0,  length+1))
            if cnt in cnt_range:
                cnt_range.remove(cnt)
            lure_cnt = random.choice(cnt_range)
            
            lure = ''
            while (lure == given_string or
                  lure in string_set or
                  lure == ''):
                lure = ''.join(random.choices(vowels, k=lure_cnt))
                lure += ''.join(random.choices(consonants, k=length-lure_cnt))
                char_list = list(lure)
                random.shuffle(char_list)
                lure = ''.join(char_list)
            string_set.append(lure)

    elif rule == "consonants":
        cnt = count_vowels(given_string)
        
        # odd numbers only to avoid equal number of vowels and consonants
        min_length = max(given_length - cnt, int(given_length*0.5))
        if min_length % 2 == 0:
            min_length += 1
        length = random.choice(range(min_length, int(given_length*1.5), 2))

        cnt = len(given_string) - count_vowels(given_string)
        
        chosen = ''
        while chosen == given_string or chosen == '':
            chosen = ''.join(random.choices(consonants, k=cnt))
            chosen += ''.join(random.choices(vowels, k=length-cnt))
            char_list = list(chosen)
            random.shuffle(char_list)
            chosen = ''.join(char_list)
        string_set.append(chosen)

        for _ in range(3):
            length = random.choice(range(min_length,  int(given_length*1.5), 2)) 
            
            cnt_range = list(range(0,  length+1))
            if cnt in cnt_range:
                cnt_range.remove(cnt)
            lure_cnt = random.choice(cnt_range)
            
            lure = ''
            while (lure == given_string or
                  lure in string_set or
                  lure == ''):                
                lure = ''.join(random.choices(consonants, k=lure_cnt))
                lure += ''.join(random.choices(vowels, k=length-lure_cnt))
                char_list = list(lure)
                random.shuffle(char_list)
                lure = ''.join(char_list)
            string_set.append(lure)
    else:
        raise Exception("Rule not recognized")

    return given_string, string_set

def count_vowels(str):
    vowels = "aeiouAEIOU"
    return sum(1 for char in str if char in vowels)

def generate_few_shot(variant):
    prompt = "Example of a short session:\n"
    if variant == "card":
        rules = ["number", "color", "shape"]

        for rule in rules:
            if rules.index(rule) != 0:
                prompt += "Correct!\n"
                
            given_card, card_set = wcst_generator(rule)
            true_ans = card_set[0]
            ans = card_set[1] # Arbitrary wrong choice
            random.shuffle(card_set)
            ans = card_set.index(ans) + 1
            true_ans = card_set.index(true_ans) + 1

            prompt += f"Given: {given_card}\nOptions:\n1. {card_set[0]}\n2. {card_set[1]}\n3. {card_set[2]}\n4. {card_set[3]}\n\n"
            prompt += f"ANSWER: {ans}\n\n"
            prompt += "Incorrect. Please try again.\n"
            prompt += f"Given: {given_card}\nOptions:\n1. {card_set[0]}\n2. {card_set[1]}\n3. {card_set[2]}\n4. {card_set[3]}\n\n"
            prompt += f"ANSWER: {true_ans}\n\n"

            for _ in range(3):
                prompt += "Correct!\n"

                given_card, card_set = wcst_generator(rule)
                true_ans = card_set[0]
                random.shuffle(card_set)
                true_ans = card_set.index(true_ans) + 1

                prompt += f"Given: {given_card}\nOptions:\n1. {card_set[0]}\n2. {card_set[1]}\n3. {card_set[2]}\n4. {card_set[3]}\n\n"
                prompt += f"ANSWER: {true_ans}\n\n"

    elif variant == "card-random":
        rules = ["number", "color", "shape"]

        for rule in rules:
            if rules.index(rule) != 0:
                prompt += "Correct!\n"

            given_card, card_set = wcst_generator(rule, randomize=True)
            true_ans = card_set[0]
            ans = card_set[1] # Arbitrary wrong choice
            random.shuffle(card_set)
            ans = card_set.index(ans) + 1
            true_ans = card_set.index(true_ans) + 1

            prompt += f"Given: {given_card}\nOptions:\n1. {card_set[0]}\n2. {card_set[1]}\n3. {card_set[2]}\n4. {card_set[3]}\n\n"
            prompt += f"ANSWER: {ans}\n\n"
            prompt += "Incorrect. Please try again.\n"
            prompt += f"Given: {given_card}\nOptions:\n1. {card_set[0]}\n2. {card_set[1]}\n3. {card_set[2]}\n4. {card_set[3]}\n\n"
            prompt += f"ANSWER: {true_ans}\n\n"

            for _ in range(3):
                prompt += "Correct!\n"

                given_card, card_set = wcst_generator(rule, randomize=True)
                true_ans = card_set[0]
                random.shuffle(card_set)
                true_ans = card_set.index(true_ans) + 1

                prompt += f"Given: {given_card}\nOptions:\n1. {card_set[0]}\n2. {card_set[1]}\n3. {card_set[2]}\n4. {card_set[3]}\n\n"
                prompt += f"ANSWER: {true_ans}\n\n"

    elif variant == "string":
        rules = ["length", "vowels", "consonants"]

        for rule in rules:
            if rules.index(rule) != 0:
                prompt += "Correct!\n"

            given_string, string_set = string_generator(rule)
            true_ans = string_set[0]
            ans = string_set[1] # Arbitrary wrong choice
            random.shuffle(string_set)
            ans = string_set.index(ans) + 1
            true_ans = string_set.index(true_ans) + 1

            prompt += f"Given: {given_string}\nOptions:\n1. {string_set[0]}\n2. {string_set[1]}\n3. {string_set[2]}\n4. {string_set[3]}\n\n"
            prompt += f"ANSWER: {ans}\n\n"
            prompt += "Incorrect. Please try again.\n"
            prompt += f"Given: {given_string}\nOptions:\n1. {string_set[0]}\n2. {string_set[1]}\n3. {string_set[2]}\n4. {string_set[3]}\n\n"
            prompt += f"ANSWER: {true_ans}\n\n"

            for _ in range(3):
                prompt += "Correct!\n"

                given_string, string_set = string_generator(rule)
                true_ans = string_set[0]
                random.shuffle(string_set)
                true_ans = string_set.index(true_ans) + 1

                prompt += f"Given: {given_string}\nOptions:\n1. {string_set[0]}\n2. {string_set[1]}\n3. {string_set[2]}\n4. {string_set[3]}\n\n"
                prompt += f"ANSWER: {true_ans}\n\n"

    else:
        raise Exception("Variant not recognized")
    
    return prompt

def check_rule_ambiguity(given_card_str, true_answer_str, bg_color=False):
    """
    Check if multiple rules could lead to the same correct answer for a given card.
    
    Args:
        given_card_str: String representation of the given card (e.g., "two green triangles")
        true_answer_str: String representation of the correct answer card
        bg_color: Whether to include background color in the analysis
    
    Returns:
        bool: True if multiple rules could match, False otherwise
    """
    # Parse given card attributes
    given_parts = given_card_str.split()
    true_parts = true_answer_str.split()
    
    if bg_color and len(given_parts) >= 4 and len(true_parts) >= 4:
        given_attrs = {
            'number': given_parts[0],
            'color': given_parts[1], 
            'shape': given_parts[2].rstrip('s'),  # Remove plural
            'background': given_parts[3] if len(given_parts) > 3 else 'white'
        }
        true_attrs = {
            'number': true_parts[0],
            'color': true_parts[1],
            'shape': true_parts[2].rstrip('s'),
            'background': true_parts[3] if len(true_parts) > 3 else 'white'
        }
        rules_to_check = ['number', 'color', 'shape', 'background']
    else:
        given_attrs = {
            'number': given_parts[0],
            'color': given_parts[1],
            'shape': given_parts[2].rstrip('s')  # Remove plural
        }
        true_attrs = {
            'number': true_parts[0],
            'color': true_parts[1],
            'shape': true_parts[2].rstrip('s')
        }
        rules_to_check = ['number', 'color', 'shape']
    
    # Count how many rules match between given card and true answer
    matching_rules = 0
    for rule in rules_to_check:
        if given_attrs[rule] == true_attrs[rule]:
            matching_rules += 1
    
    # Ambiguous if more than one rule matches
    return matching_rules > 1