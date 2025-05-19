import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_run_stats(stats_file):
    """Load WCST statistics from a JSON file"""
    with open(stats_file, 'r') as f:
        return json.load(f)

def calculate_score(runs):
    """Calculate score for WCST runs"""
    scores = []
    # Each run is a list of trials
    for run_data in runs:
        if isinstance(run_data, list):  # Make sure it's a list of trials
            trial_scores = []
            current_rule = ""
            completion_lengths = []
            num_correct = 0
            
            for query in run_data:
                if isinstance(query, dict) and 'rule' in query and 'correct' in query:
                    if query['rule'] != current_rule:
                        if current_rule != "" and completion_lengths:  # Only append if we have a previous rule
                            if num_correct >= 5:  # Only count if we reached criterion
                                trial_scores.append(completion_lengths[-1])
                        current_rule = query['rule']
                        completion_lengths.append(0)
                        num_correct = 0
                    else:
                        completion_lengths[-1] += 1
                    
                    if query['correct']:
                        num_correct += 1
            
            # Handle the last rule if it met criterion
            if completion_lengths and num_correct >= 5:
                trial_scores.append(completion_lengths[-1])
            
            if trial_scores:
                # Calculate score as average of 1/length for each completed rule
                score = np.mean([1/l for l in trial_scores])
                scores.append(score)
    
    return np.mean(scores) if scores else 0

def get_setup_type(filename):
    """Get the setup type from filename"""
    if 'card-random' in filename:
        return 'card-random'
    elif 'card' in filename:
        return 'card'
    elif 'string' in filename:
        return 'string'
    return None

def analyze_results():
    data_dir = Path('./wcst_data')
    if not data_dir.exists():
        raise FileNotFoundError("WCST data directory not found")

    # Dictionary to store results by setup type and category
    results = {
        'card': {
            'non_cot': {},
            'cot': {},
            'few_shot': {},
            'few_shot_cot': {}
        },
        'card-random': {
            'non_cot': {},
            'cot': {},
            'few_shot': {},
            'few_shot_cot': {}
        },
        'string': {
            'non_cot': {},
            'cot': {},
            'few_shot': {},
            'few_shot_cot': {}
        }
    }
    
    # Process files
    for model_dir in data_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == 'old':
            continue
        
        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        for stats_file in model_dir.glob('*.json'):
            if 'old' in str(stats_file) or 'history' in str(stats_file) or 'reasoning' in str(stats_file):
                continue
                
            setup_type = get_setup_type(stats_file.stem)
            if not setup_type:
                continue
                
            is_cot = '_cot' in stats_file.stem
            is_few_shot = '_few_shot' in stats_file.stem
            
            if is_few_shot and is_cot:
                category = 'few_shot_cot'
            elif is_few_shot:
                category = 'few_shot'
            elif is_cot:
                category = 'cot'
            else:
                category = 'non_cot'
                
            print(f"  Processing {setup_type} file: {stats_file.name}")
                
            try:
                stats = load_run_stats(stats_file)
                if not isinstance(stats, dict):
                    continue
                
                scores = []
                for run_id, run_data in stats.items():
                    if not isinstance(run_data, list):
                        continue
                    try:
                        score = calculate_score([run_data])
                        if score > 0:
                            scores.append(score)
                    except Exception as e:
                        print(f"  Error processing run {run_id}: {e}")
                        continue
                
                if scores:
                    if model_name not in results[setup_type][category]:
                        results[setup_type][category][model_name] = {
                            'scores': [],
                            'n_files': 0
                        }
                    results[setup_type][category][model_name]['scores'].extend(scores)
                    results[setup_type][category][model_name]['n_files'] += 1
                    
            except Exception as e:
                print(f"  Error processing {stats_file.name}: {e}")
                continue

    # Calculate statistics and generate plots for each setup type
    for setup_type in results:
        # Calculate final statistics
        for category in results[setup_type]:
            for model_name in list(results[setup_type][category].keys()):
                model_data = results[setup_type][category][model_name]
                scores = model_data['scores']
                if scores:
                    results[setup_type][category][model_name] = {
                        'avg_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'max_score': np.max(scores),
                        'min_score': np.min(scores),
                        'n_runs': len(scores),
                        'n_files': model_data['n_files']
                    }
                else:
                    del results[setup_type][category][model_name]

        # Generate plot for this setup type
        plt.figure(figsize=(12, 6))
        
        all_models = set()
        for category in results[setup_type].values():
            all_models.update(category.keys())
        all_models = sorted(list(all_models))
        
        x = np.arange(len(all_models))
        width = 0.2
        
        for i, category in enumerate(['non_cot', 'cot', 'few_shot', 'few_shot_cot']):
            scores = []
            min_scores = []
            max_scores = []
            for model in all_models:
                if model in results[setup_type][category]:
                    model_data = results[setup_type][category][model]
                    scores.append(model_data['avg_score'])
                    min_scores.append(model_data['min_score'])
                    max_scores.append(model_data['max_score'])
                else:
                    scores.append(0)
                    min_scores.append(0)
                    max_scores.append(0)
            
            yerr = np.array([
                [s - min_s for s, min_s in zip(scores, min_scores)],
                [max_s - s for s, max_s in zip(scores, max_scores)]
            ])
            
            bars = plt.bar(x + (i-1.5)*width, scores, width, yerr=yerr,
                          label=category.replace('_', ' ').title())
            
            for idx, rect in enumerate(bars):
                height = rect.get_height()
                if height > 0:
                    plt.text(rect.get_x() + rect.get_width()/2., height,
                            f'{scores[idx]:.2f}',
                            ha='center', va='bottom')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title(f'WCST {setup_type.title()} - Average Scores by Model and Category')
        plt.xticks(x, all_models, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plots_dir = Path('./wcst_data/plots')
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(f'wcst_data/plots/wcst_analysis_{setup_type}.png', bbox_inches='tight')
        plt.close()
    
    # Save summary statistics
    with open('wcst_data/wcst_summary.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == '__main__':
    results = analyze_results()
    # Print summary
    print("\nWCST Analysis Results:")
    for setup_type in ['card', 'card-random', 'string']:
        print(f"\n{setup_type.upper()}:")
        for category in ['non_cot', 'cot', 'few_shot', 'few_shot_cot']:
            print(f"\n{category.upper()}:")
            for model, stats in results[setup_type][category].items():
                print(f"\n{model}:")
                for metric, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")