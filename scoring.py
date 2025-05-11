import numpy as np
import json

def score(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    scores = []
    for trial in data:
        current_rule = ""
        completion_lengths = []
        num_correct = 0

        for query in trial:
            if query['rule'] != current_rule:
                current_rule = query['rule']
                completion_lengths.append(0)
                num_correct = 0
            else:
                completion_lengths[-1] += 1
            
            if query['correct']:
                num_correct += 1
        
        if num_correct < 5:
            completion_lengths.pop(-1)
        # score = (len(completion_lengths) - 1) / len(completion_lengths) * np.sum(completion_lengths)
        score = np.sum([1/l for l in completion_lengths]) * 5/6
        scores.append(score)
    return np.mean(scores)