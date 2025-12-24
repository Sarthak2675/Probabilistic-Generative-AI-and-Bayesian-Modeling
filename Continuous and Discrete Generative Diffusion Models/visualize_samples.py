import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

def visualize_samples(samples_path, save_path=None):
    """
    Visualize generated samples from DDPM
    """
    # Load samples
    samples = torch.load(samples_path, map_location='cpu')
    
    # Convert to numpy
    samples = samples.detach().cpu().numpy()
    
    # Create figure
    n_samples = min(samples.shape[0], 24)
    cols = 4
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        row = i // cols
        col = i % cols
        # print(i, row, col)
        # Display image
        axes[row, col].imshow(samples[i, 0], cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Sample {i+1}')
    
    # Hide empty subplots
    for i in range(n_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Model Template")
    parser.add_argument("--model", type=str, default="ddpm", choices=["ddpm", "ddpm_cond", "d3pm","d3pm_cond"], help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample","eval_fid"], help="Mode: train or sample")
    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine"], help="Noise scheduler type")
    parser.add_argument("--class_label", type=int, default=0, help="Class label for conditional models")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    samples_path = ""
    run_name = f"exps_{args.model}/{args.epochs}ep_{args.learning_rate}lr_{args.num_steps}_ns_{args.scheduler}_scheduler/"
    if args.model == 'ddpm':
        samples_path = f"{run_name}{args.num_samples}samples_{args.num_steps}steps.pt"
    elif args.model == 'ddpm_cond':
        samples_path = f"{run_name}{args.class_label}class_{args.num_samples}samples_{args.num_steps}steps.pt"
    elif args.model == 'd3pm':
        samples_path = f"{run_name}{args.num_samples}samples_{args.num_steps}steps"
    elif args.model == 'd3pm_cond':
        samples_path = f"{run_name}{args.class_label}class_{args.num_samples}samples_{args.num_steps}steps.pt"
    
    save_path = ""
    if args.model == 'ddpm' or args.model == 'd3pm':
        save_path = f"{run_name}{args.num_samples}samples_{args.num_steps}steps.png"
    else:
        save_path = f"{run_name}{args.class_label}class_{args.num_samples}samples_{args.num_steps}steps.png"
    
    visualize_samples(samples_path, save_path)
