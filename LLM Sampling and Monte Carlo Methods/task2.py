#!/usr/bin/env python3
"""
Task 2: Sequential Monte Carlo (SMC) — CLI.

Input test row (JSONL):
  {
    "prompt_id": int,
    "prefix": str,
    "instruction": str,          # optional; default "Continue the text."
    "max_output_tokens": int
  }

Output row (JSONL), one per prompt (minimal fields used by eval):
  {
    "prompt_id": int,
    "prefix": str,
    "continuations": [{
      "method": "SMC[topk=K; beta=β; N=B]",
      "samples": [                 # only fields needed by eval/plots
        {"text": str, "weight": float}, ...
      ],
      "normalized_weights": [float, ...]
    }]
  }

Usage:
    python task2.py --hf-token <token> --counts-dir /home/daksh/AML/assignment_3/llama_test_output 
                    --A <num_prompts> --B <num_particles> [--beta <beta>] [--out <file>]
    
Note: Uses fixed model meta-llama/Meta-Llama-3-8B-Instruct

"""
import argparse
from utils import set_seed, load_jsonl, save_jsonl, ensure_dir
from generate_task2_smc import load_model, load_counts_and_reward, smc_for_prompt


def parse_args():
    """Parse command line arguments for Task 2 Sequential Monte Carlo execution.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            Model Configuration:
                - hf_token (str): HuggingFace authentication token
                - device (str): CUDA device specification (default: 'cuda:0')
                Note: Uses fixed model meta-llama/Meta-Llama-3-8B-Instruct
                
            Reward Function Parameters:
                - counts_dir (str): Path to n-gram count files (llama_test_output/)
                - epsilon (float): Smoothing parameter for n-gram probabilities (default: 1e-9)
                
            SMC Algorithm Parameters:
                - beta (float): Terminal reward scaling factor β in exp(β×R)
                - k (int): Top-k sampling parameter
                
            Execution Configuration:
                - test_file (str): Path to JSONL file with test prompts
                - A (int): Number of prompts to process
                - B (int): Number of particles per prompt (N=B)
                - seed (int): Random seed for reproducibility (default: 123)
                
            Output Configuration:
                - out (str): Output JSONL file path (default: 'data/outputs_task2_SMC.jsonl')
    """
    p = argparse.ArgumentParser(
        description="Task 2: Algorithm 2 Sequential Monte Carlo (SMC) for Controlled Text Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run SMC with 16 particles on 10 prompts
    python task2.py --hf-token <token> \\
                    --counts-dir /home/daksh/AML/assignment_3/llama_test_output --A 10 --B 16
    
    # High-temperature sampling with frequent resampling
    python task2.py --hf-token <token> --counts-dir /home/daksh/AML/assignment_3/llama_test_output \\
                    --A 5 --B 32 --beta 10.0
        """
    )
    
    # === MODEL CONFIGURATION ===
    model_group = p.add_argument_group("Model Configuration")
    model_group.add_argument("--hf-token", type=str, required=True,
                           help="HuggingFace authentication token for model access")
    model_group.add_argument("--device", type=str, default="cuda:0",
                           help="CUDA device specification (default: cuda:0)")

    # === REWARD FUNCTION CONFIGURATION ===
    reward_group = p.add_argument_group("Reward Function Parameters")
    reward_group.add_argument("--counts-dir", type=str, required=True)
    reward_group.add_argument("--epsilon", type=float, default=1e-9,
                            help="Smoothing parameter for n-gram probabilities (default: 1e-9)")

    # === SMC ALGORITHM PARAMETERS ===
    smc_group = p.add_argument_group("SMC Algorithm Parameters (Algorithm 2)")
    smc_group.add_argument("--beta", type=float, default=5.0,
                         help="Terminal reward scaling β in exp(β×R) (default: 5.0)")

    # === PROPOSAL DISTRIBUTION ===
    proposal_group = p.add_argument_group("Proposal Distribution ")
    proposal_group.add_argument("--k", type=int, default=10,
                              help="Top-k sampling parameter")

    # === EXECUTION CONFIGURATION ===
    exec_group = p.add_argument_group("Execution Configuration")
    exec_group.add_argument("--test-file", type=str, default="data/test_prompts.jsonl",
                          help="Path to JSONL file containing test prompts")
    exec_group.add_argument("--A", type=int, required=True,
                          help="Number of prompts to process from test file")
    exec_group.add_argument("--B", type=int, required=True,
                          help="Number of particles per prompt (N=B in Algorithm 2)")
    exec_group.add_argument("--seed", type=int, default=123,
                          help="Random seed for reproducible results (default: 123)")

    # === OUTPUT CONFIGURATION ===
    output_group = p.add_argument_group("Output Configuration")
    output_group.add_argument("--out", type=str, default="data/outputs_task2_SMC.jsonl",
                            help="Output JSONL file path (default: data/outputs_task2_SMC.jsonl)")
    return p.parse_args()


def _method_tag(args, N: int) -> str:
    """Generate standardized method identification tag for Algorithm 2 SMC results.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments containing:
            - k (int): Top-k sampling parameter (fixed at 10 for Algorithm 2)
            - beta (float): Terminal reward scaling factor
        N (int): Number of particles used in SMC sampling
        
    Returns:
        str: Formatted method tag for JSON output, e.g., "SMC[topk=10; beta=5.0; N=32]"
    """
    return f"SMC[topk={args.k}; beta={args.beta}; N={N}]"


def main():
    
    # === ARGUMENT PARSING AND INITIALIZATION ===
    args = parse_args()
    set_seed(args.seed)  # Ensure reproducible results across runs
    ensure_dir("data")   # Create output directory if it doesn't exist

    # === DATASET LOADING ===
    # Load test prompts and limit to requested number A
    rows = load_jsonl(args.test_file)[: args.A]
    print(f"[Task 2] Loaded {len(rows)} prompts from {args.test_file}")

    # === MODEL AND REWARD INITIALIZATION ===
    # Load HuggingFace model, tokenizer, and special token IDs
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tok, model, eos_id = load_model(model_name, args.hf_token, args.device)
    print(f"[Task 2] Loaded model: {model_name}")
    
    # Initialize reward function with n-gram count statistics
    reward_calc = load_counts_and_reward(args.counts_dir, epsilon=args.epsilon)
    print(f"[Task 2] Loaded reward function from: {args.counts_dir}")

    # === PROMPT PROCESSING LOOP ===
    out_rows = []
    print(f"[Task 2] Starting SMC sampling with N={args.B} particles per prompt")
    
    for row in rows:
        # === PROMPT PREPARATION ===
        prompt_id = row["prompt_id"]
        prefix = row["prefix"]
        instruction = row.get("instruction", "Continue the text.")
        
        # Build complete prompt by combining instruction and prefix
        full_prompt = f"{instruction} {prefix}"
        max_new = int(row.get("max_output_tokens", 50))
        
        print(f"[Task 2] Processing prompt {prompt_id}: '{prefix[:50]}{'...' if len(prefix) > 50 else ''}'")

        result = smc_for_prompt(
            tokenizer=tok,
            model=model,
            reward_calc=reward_calc,
            prefix=full_prompt,
            N=args.B,                    # Number of particles
            max_new_tokens=max_new,      # Maximum sequence length
            eos_id=eos_id,              # End-of-sequence token ID
            beta=args.beta,              # Terminal reward scaling β
            k=args.k,                    # Top-k parameter (fixed at 10)
        )

        # === RESULT COMPILATION ===
        out_rows.append({
            "prompt_id": prompt_id,
            "prefix": prefix,
            "continuations": [
                {
                    "method": _method_tag(args, args.B),
                    "samples": result["samples"],
                    "normalized_weights": result["normalized_weights"],
                }
            ]
        })
        print(f"[Task 2] Completed prompt {prompt_id}")
    # === OUTPUT GENERATION ===
    # Save all results to JSONL file (one JSON object per line)
    save_jsonl(args.out, out_rows)
    
    # === COMPLETION SUMMARY ===
    total_particles = len(out_rows) * args.B
    print(f"\n[Task 2 Complete] Successfully processed {len(out_rows)} prompts")
    print(f"[Task 2 Complete] Generated {total_particles} total particles")
    print(f"[Task 2 Complete] Results saved to: {args.out}")
    
    # Display algorithm configuration for verification
    print(f"\n[Algorithm 2 Configuration]")
    print(f"  - Particles per prompt (N): {args.B}")
    print(f"  - Top-k parameter: {args.k}")
    print(f"  - Terminal reward scaling (β): {args.beta}")

    print(f"  - Model: {model_name}")
    print(f"  - Reward function: {args.counts_dir}")


if __name__ == "__main__":
    main()
