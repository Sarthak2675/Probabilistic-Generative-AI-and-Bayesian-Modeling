# task1.py
"""Task 1: Sequential Importance Sampling (SIS) — CLI.

Uses fixed model: meta-llama/Meta-Llama-3-8B-Instruct

Input test row (JSONL):
  {
    "prompt_id": int,
    "prefix": str,
    "instruction": str,          # optional; default "Continue the text."
    "max_output_tokens": int
  }

Output row (JSONL), one per prompt:
  {
    "prompt_id": int,
    "prefix": str,
    "continuations": [{
      "method": "IS[topk=K; beta=β]",
      "samples": [                 # only fields needed by eval/plots
        {"text": str, "weight": float}, ...
      ],
      "normalized_weights": [float, ...]
    }]
  }
"""
import argparse
from utils import set_seed, load_jsonl, save_jsonl, ensure_dir
from generate_task1_is import (
    load_model,
    load_counts_and_reward,
    importance_sampling_for_prompt,
)


def parse_args():
    """Parse command line arguments for Sequential Importance Sampling execution.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - hf_token: Hugging Face authentication token
            - device: Target device for model placement
            Note: Uses fixed model meta-llama/Meta-Llama-3-8B-Instruct
            - counts_dir: Directory containing trigram probability data
            - epsilon: Smoothing parameter for unseen trigrams
            - beta: Reward scaling factor (controls emphasis on rare trigrams)
            - k: Top-k parameter for proposal sampling (fixed at 10 per spec)
            - test_file: Path to input prompts in JSONL format
            - A: Number of prompts to process from test file
            - B: Number of samples to generate per prompt
            - seed: Random seed for reproducible results
            - out: Output file path for results in JSONL format
            
    Note:
        The top-k parameter k is fixed at 10
    """
    p = argparse.ArgumentParser(
        description="Task 1: Sequential Importance Sampling for trigram-biased text generation"
    )
    
    # === MODEL CONFIGURATION ===
    p.add_argument("--hf-token", type=str, required=True, 
                   help="Hugging Face token for model access")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device for model inference (cuda:0, cpu, etc.)")

    # === REWARD COMPUTATION ===
    p.add_argument("--counts-dir", type=str, required=True,
                   help="Directory containing trigram probability data")
    p.add_argument("--epsilon", type=float, default=1e-9,
                   help="Smoothing parameter for unseen trigrams")

    # === IMPORTANCE SAMPLING PARAMETERS ===
    p.add_argument("--beta", type=float, default=5.0,
                   help="Reward scaling factor β in exp(β×R(x))")

    # === SAMPLING CONFIGURATION ===
    p.add_argument("--k", type=int, default=10,
                   help="Top-k parameter for proposal sampling (Algorithm 1 uses k=10)")

    # === DATASET AND EXECUTION ===
    p.add_argument("--test-file", type=str, default="data/test_prompts.jsonl",
                   help="Path to input prompts file")
    p.add_argument("--A", type=int, required=True,
                   help="Number of prompts to process from test file")
    p.add_argument("--B", type=int, required=True,
                   help="Number of samples to generate per prompt")
    p.add_argument("--seed", type=int, default=123,
                   help="Random seed for reproducible results")

    # === OUTPUT ===
    p.add_argument("--out", type=str, default="data/outputs_task1_IS.jsonl",
                   help="Output file path for generated samples")
    
    return p.parse_args()


def _method_tag(args) -> str:
    """Generate method identification string for output tracking and analysis.
    
    Args:
        args: Parsed command line arguments containing hyperparameters
        
    Returns:
        str: Method identifier string in format "IS[topk=K; beta=β]"
    """
    return f"IS[topk={args.k}; beta={args.beta}]"


def main():

    # === INITIALIZATION ===
    args = parse_args()
    set_seed(args.seed)      # Ensure reproducible results
    ensure_dir("data")       # Create output directory if needed

    # === LOAD COMPONENTS ===
    # Load test prompts (first A prompts only)
    rows = load_jsonl(args.test_file)[: args.A]
    
    # Initialize model components for text generation
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tok, model, eos_id = load_model(model_name, args.hf_token, args.device)
    
    # Initialize reward calculator for trigram-based scoring
    reward_calc = load_counts_and_reward(args.counts_dir, epsilon=args.epsilon)

    # === SEQUENTIAL IMPORTANCE SAMPLING EXECUTION ===
    out_rows = []
    for row in rows:
        # Extract prompt information
        prompt_id = row["prompt_id"]
        prefix = row["prefix"]
        instruction = row.get("instruction", "Continue the text.")
        print(f"[task1:IS] Processing prompt ='{prefix[:30]}...'")
        # Combine instruction and prefix for complete context
        full_prompt = f"{instruction} {prefix}"
        max_new = int(row.get("max_output_tokens", 50))

        # Execute Algorithm 1: Sequential Importance Sampling
        result = importance_sampling_for_prompt(
            tokenizer=tok,
            model=model,
            reward_calc=reward_calc,
            prefix=full_prompt,
            K=args.B,                    # Number of samples per prompt
            max_new_tokens=max_new,
            eos_id=eos_id,
            beta=args.beta,              # Reward scaling parameter
            k=args.k,                    # Top-k for proposal sampling
        )

        # Package results in standardized format
        out_rows.append({
            "prompt_id": prompt_id,
            "prefix": prefix,
            "continuations": [
                {
                    "method": _method_tag(args),                # Algorithm identifier
                    "samples": result["samples"],               # Detailed sample data
                    "normalized_weights": result["normalized_weights"],  # Weight array
                }
            ]
        })

    # === OUTPUT GENERATION ===
    save_jsonl(args.out, out_rows)
    print(f"[task1:IS] Generated {len(out_rows)} prompts × {args.B} samples → {args.out}")


if __name__ == "__main__":
    main()
