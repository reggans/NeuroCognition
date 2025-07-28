import numpy as np
import json

# def score(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
    
#     scores = []
#     for trial in data:
#         current_rule = ""
#         completion_lengths = []
#         num_correct = 0

#         for query in trial:
#             if query['rule'] != current_rule:
#                 current_rule = query['rule']
#                 completion_lengths.append(0)
#                 num_correct = 0
#             else:
#                 completion_lengths[-1] += 1
            
#             if query['correct']:
#                 num_correct += 1
        
#         if num_correct < 5:
#             completion_lengths.pop(-1)
#         # score = (len(completion_lengths) - 1) / len(completion_lengths) * np.sum(completion_lengths)
#         score = np.sum([1/l for l in completion_lengths]) * 5/6
#         scores.append(score)
#     return np.mean(scores)

def score(file_path):
    # TODO: Customize experiment parameters
    # - consecutive correct ans
    # - conceptual response
    with open(file_path, 'r') as file:
        data = json.load(file)

    scores = {
        "accuracy": [],
        "perserverative_error": [],
        "cat_complete": [],
        "first_cat_trials": [],
        "failure_set": []
    }

    for trial in data:
        correct_rule = ""
        perserverated_response = -1
        first_complete = False
        correct_run = 0
        conceptual_response = False

        total_complete = 0
        total_perseverated = 0
        total_fms = 0
        total_correct = 0

        for i, query in enumerate(trial):
            if correct_rule != query["rule"] and correct_rule != "":
                perserverated_response = -1
                correct_run = 0
                conceptual_response = False

                if not first_complete:
                    first_complete = True
                    scores["first_cat_trials"].append(i+1)

            if query["correct"]:
                total_correct += 1
                correct_run += 1

                if correct_run >= 3:
                    conceptual_response = True
                if correct_run >= 5:
                    total_complete += 1
            else:
                correct_run = 0

                if conceptual_response:
                    total_fms += 1
                
                try:
                    ans = int(query["model_ans"])

                    if ans == perserverated_response:
                        total_perseverated += 1
                    perserverated_response = ans
                except:
                    continue
        
        n_query = len(trial)
        scores["accuracy"].append(total_correct / n_query)
        scores["perserverative_error"].append(total_perseverated / n_query)
        scores["cat_complete"].append(total_complete)
        scores["failure_set"].append(total_fms / n_query)
    
    for metric in scores:
        scores[metric] = np.mean(scores[metric])
    
    return scores