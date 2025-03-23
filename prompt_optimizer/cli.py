#!/usr/bin/env python
import argparse
import sys
import os
import pandas as pd
from pathlib import Path
import asyncio
from prompt_optimizer.model import BaseModel, GPTModel
from prompt_optimizer.prompt_optimizer import run_optimizer
from prompt_optimizer.config import OPTIMIZER_CONFIG


def read_prompt_from_file(file_path):
    """Read a prompt from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading prompt file: {e}", file=sys.stderr)
        sys.exit(1)


async def main():
    """Main entry point for the prompt optimizer CLI."""
    parser = argparse.ArgumentParser(
        description="Prompt Optimizer - Improve your system prompts automatically",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-csv", "-i",
        required=True,
        help="Path to CSV file with input and ground_truth columns"
    )
    
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt", "-p",
        help="Initial system prompt text"
    )
    prompt_group.add_argument(
        "--prompt-file", "-f",
        help="Path to file containing initial system prompt"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file to save the optimized prompt (default: print to stdout)"
    )
    
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=OPTIMIZER_CONFIG.max_iterations,
        help="Maximum number of optimization iterations"
    )
    
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=OPTIMIZER_CONFIG.chunk_size,
        help="Number of examples to process in each chunk"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output during optimization"
    )
    
    parser.add_argument(
        "--llm-client", "-l",
        default="gpt",
        help="LLM client to use for optimization"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)
    
    # Get the initial prompt
    if args.prompt_file:
        initial_prompt = read_prompt_from_file(args.prompt_file)
    else:
        initial_prompt = args.prompt
    
    # Configure optimizer
    OPTIMIZER_CONFIG.max_iterations = args.iterations
    OPTIMIZER_CONFIG.chunk_size = args.chunk_size
    
    # Initialize and run optimizer
    try:
        if args.llm_client == "gpt":
            model = GPTModel()
        else:
            raise ValueError(f"Invalid LLM client: {args.llm_client}")
        
        if args.verbose:
            print(f"Starting prompt optimization with {args.iterations} iterations")
            print(f"Initial prompt: {initial_prompt[:100]}...")
        optimized_prompt = await run_optimizer(
            llm_client=model,
            input_ground_truth_csv=args.input_csv,
            initial_prompt=initial_prompt
        )
        
        # Output the result
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(optimized_prompt)
                
            if args.verbose:
                print(f"Optimized prompt saved to: {args.output}")
        else:
            print("\nOptimized Prompt:\n" + "="*50)
            print(optimized_prompt)
            print("="*50)
        
    except Exception as e:
        print(f"Error during optimization: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())