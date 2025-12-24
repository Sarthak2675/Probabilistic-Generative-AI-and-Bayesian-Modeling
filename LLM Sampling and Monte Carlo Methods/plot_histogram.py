## Students need to implement this file.
import matplotlib.pyplot as plt
import argparse
import json
from utils import load_jsonl, ensure_dir

def parse_args():
    p = argparse.ArgumentParser(description="Histogram plotter")
    p.add_argument("--inputs", type=str, required=True, nargs="+", help="Input data file (text file with one number per line)")
    p.add_argument("--output", type=str, default="images/", help="Output image file path for the histogram")
        
    return p.parse_args()

def main():
    args = parse_args()
    for path in args.inputs:
        print(f"Evaluating {path}...")
        rows = load_jsonl(path)        
        all_weights = []
        methods_seen = set()
        
        for r in rows:
            for block in r.get("continuations", []):
                methods_seen.add(block.get("method", "Unknown"))
                samples = block.get("samples", [])
                
                # Handle normalized_weights if available at block level
                if "normalized_weights" in block:
                    norm_weights = block["normalized_weights"]
                    if len(norm_weights) == len(samples):
                        all_weights.extend(norm_weights)
                        continue
                
                # Otherwise, extract weights from individual samples
                weights = []
                for sample in samples:
                    # Check if sample is a dict (TSMC format) or string (Greedy format)
                    if isinstance(sample, dict):
                        weight = sample.get("weight", 0.0)
                        weights.append(weight)
                    elif isinstance(sample, str):
                        # Greedy samples are just strings - assign uniform weight
                        weights.append(1.0)
                    else:
                        weights.append(0.0)
                
                # Normalize weights
                if weights:
                    total_weight = sum(weights)
                    if total_weight > 0:
                        norm_weights = [w / total_weight for w in weights]
                        all_weights.extend(norm_weights)
        
        # Create histogram with labels and title
        # 10 bins with intervals [0,0.1), [0.1,0.2), ..., [0.9,1.0]
        bins = [i * 0.1 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
        plt.hist(all_weights, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Normalized Weight')
        plt.ylabel('Frequency')
        plt.title(f'Weight Distribution - {path.split("/")[-1].replace(".jsonl", "")}')
        plt.xlim(0, 1.0)  # Set x-axis range from 0 to 1
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = f"{args.output}/histogram_{path.split('/')[-1].replace('.jsonl','')}.png"
        plt.savefig(output_path, dpi=150)
        plt.clf()
        print(f"Saved histogram to {output_path}")

if __name__ == "__main__":
    ensure_dir("images/")
    main()