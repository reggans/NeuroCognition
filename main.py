#!/usr/bin/env python3
"""
Main script to run cognitive evaluation tests.
This script can run either the Wisconsin Card Sorting Test (WCST) or the Spatial Working Memory (SWM) test.
"""

import argparse
import sys
import os

# Add the current directory to Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(
        description="Run cognitive evaluation tests (WCST or SWM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run WCST with default settings
  python main.py wcst
  
  # Run SWM with chain-of-thought reasoning
  python main.py swm --cot
  
  # Run WCST with specific model and few-shot prompting
  python main.py wcst --model gpt-4 --model_source litellm --few_shot
  
  # Run SWM with image mode and multiple runs
  python main.py swm --image --runs 5 --n_boxes 8
        """)
    
    # Add subparsers for different tests
    subparsers = parser.add_subparsers(dest='test', help='Which test to run')
    subparsers.required = True
    
    # WCST parser
    wcst_parser = subparsers.add_parser('wcst', help='Run Wisconsin Card Sorting Test')
    wcst_parser.add_argument("--model", type=str, default="llama", help="The model to use")
    wcst_parser.add_argument("--variant", type=str, default="card", choices=["card", "card-random", "string", "empty"],
                           help="The variant of the test")
    wcst_parser.add_argument("--max_trials", type=int, default=64, help="Maximum number of trials")
    wcst_parser.add_argument("--num_correct", type=int, default=5, help="Number of correct answers required per category")
    wcst_parser.add_argument("--repeats", type=int, default=1, help="Number of runs to perform")
    wcst_parser.add_argument("--few_shot", action="store_true", help="Use few-shot prompting")
    wcst_parser.add_argument("--cot", action="store_true", help="Use chain-of-thought reasoning")
    wcst_parser.add_argument("--hint", action="store_true", help="Provide hints")
    wcst_parser.add_argument("--image", action="store_true", help="Use image mode")
    wcst_parser.add_argument("--model_source", type=str, default="vllm", choices=["vllm", "openai", "openrouter"],
                           help="The source of the model")
    wcst_parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    wcst_parser.add_argument("--think_budget", type=int, default=64, help="Budget tokens for reasoning")
    wcst_parser.add_argument("--api_key", type=str, default=None, help="API key to use")
    wcst_parser.add_argument("--verbose", type=int, default=15, help="Verbosity level")
    
    # SWM parser
    swm_parser = subparsers.add_parser('swm', help='Run Spatial Working Memory test')
    swm_parser.add_argument("--model", type=str, default=None, help="The model to use")
    swm_parser.add_argument("--model_source", type=str, default="vllm", choices=["vllm", "openai", "openrouter"],
                          help="The source of the model")
    swm_parser.add_argument("--n_boxes", type=int, default=6, help="Number of boxes in the test (more = harder)")
    swm_parser.add_argument("--n_tokens", type=int, default=1, 
                          help="Number of different tokens present at the same time (more = harder)")
    swm_parser.add_argument("--cot", action="store_true", help="Use chain-of-thought reasoning")
    swm_parser.add_argument("--runs", type=int, default=1, help="Number of runs to perform")
    swm_parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    swm_parser.add_argument("--think_budget", type=int, default=64, help="Budget tokens for reasoning")
    swm_parser.add_argument("--notes", action="store_true", help="Use note-taking assistance")
    swm_parser.add_argument("--image", action="store_true", help="Use image mode")
    swm_parser.add_argument("--api_key", type=str, default=None, help="API key to use")
    
    args = parser.parse_args()
    
    # Import and run the appropriate test
    if args.test == 'wcst':
        if args.image:
            try:
                from WCST.wcst import run_wcst_image
            except ImportError:
                # Try absolute import if relative fails
                import sys
                import os
                sys.path.append(os.path.dirname(__file__))
                from WCST.wcst import run_wcst_image
            
            print("Running Wisconsin Card Sorting Test (WCST) - Image Mode")
            print(f"Model: {args.model} (source: {args.model_source})")
            print(f"Settings: max_trials={args.max_trials}, num_correct={args.num_correct}, repeats={args.repeats}")
            print(f"Options: few_shot={args.few_shot}, cot={args.cot}, hint={args.hint}, image={args.image}")
            print("-" * 50)
            
            run_wcst_image(
                model=args.model,
                max_trials=args.max_trials,
                num_correct=args.num_correct,
                repeats=args.repeats,
                few_shot=args.few_shot,
                cot=args.cot,
                hint=args.hint,
                model_source=args.model_source,
                max_tokens=args.max_tokens,
                think_budget=args.think_budget,
                api_key=args.api_key,
                verbose=args.verbose
            )
        else:
            try:
                from WCST.wcst import run_wcst
            except ImportError:
                # Try absolute import if relative fails
                import sys
                import os
                sys.path.append(os.path.dirname(__file__))
                from WCST.wcst import run_wcst
            
            print("Running Wisconsin Card Sorting Test (WCST)")
            print(f"Model: {args.model} (source: {args.model_source})")
            print(f"Variant: {args.variant}")
            print(f"Settings: max_trials={args.max_trials}, num_correct={args.num_correct}, repeats={args.repeats}")
            print(f"Options: few_shot={args.few_shot}, cot={args.cot}, hint={args.hint}")
            print("-" * 50)
            
            run_wcst(
                model=args.model,
                variant=args.variant,
                max_trials=args.max_trials,
                num_correct=args.num_correct,
                repeats=args.repeats,
                few_shot=args.few_shot,
                cot=args.cot,
                hint=args.hint,
                model_source=args.model_source,
                max_tokens=args.max_tokens,
                think_budget=args.think_budget,
                api_key=args.api_key,
                verbose=args.verbose
            )
        
    elif args.test == 'swm':
        try:
            from SWM.main import swm_main
        except ImportError:
            # Try absolute import if relative fails
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from SWM.main import swm_main
        
        print("Running Spatial Working Memory (SWM) test")
        print(f"Model: {args.model} (source: {args.model_source})")
        print(f"Settings: n_boxes={args.n_boxes}, n_tokens={args.n_tokens}, runs={args.runs}")
        print(f"Options: cot={args.cot}, notes={args.notes}, image={args.image}")
        print("-" * 50)
        
        swm_main(
            model=args.model,
            model_source=args.model_source,
            n_boxes=args.n_boxes,
            n_tokens=args.n_tokens,
            cot=args.cot,
            runs=args.runs,
            max_tokens=args.max_tokens,
            think_budget=args.think_budget,
            notes=args.notes,
            image=args.image,
            api_key=args.api_key
        )


if __name__ == "__main__":
    main()
