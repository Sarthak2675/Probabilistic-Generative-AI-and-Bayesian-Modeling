#!/usr/bin/env python3
"""
Simple script to calculate test accuracy with hardcoded hyperparameters.
Edit the hyperparameters list in the script to test different configurations.
"""

import torch
from torchvision import datasets, transforms
from train_test import train_and_test_NN, train_and_test_CNN
from utils import seed_everything
import math


def convert_to_internal_format(hyperparams_list):
    """
    Convert hyperparameters from user-friendly format to internal format.
    
    Input: [layer_size, epochs, lr, batch_size, dropout_rate, weight_decay]
    Output: tensor with [layer_size, epochs, log_lr, log_batch_size, dropout_rate, log_weight_decay]
    """
    layer_size = float(hyperparams_list[0])
    epochs = float(hyperparams_list[1])
    lr = float(hyperparams_list[2])
    batch_size = float(hyperparams_list[3])
    dropout_rate = float(hyperparams_list[4])
    weight_decay = float(hyperparams_list[5])
    
    # Convert to log scale
    log_lr = math.log10(lr)
    log_batch_size = math.log2(batch_size)
    log_weight_decay = math.log10(weight_decay)
    
    return torch.tensor([layer_size, epochs, log_lr, log_batch_size, dropout_rate, log_weight_decay])


if __name__ == '__main__':
    # ============ CONFIGURE THESE PARAMETERS ============
    
    # Hyperparameters: [layer_size, epochs, lr, batch_size, dropout_rate, weight_decay]
    hyperparameters = [200, 9, 0.0012, 256, 0.0403, 1.09e-6]
    
    model_type = 'nn'  # Options: 'nn' or 'cnn'
    seed = 100
    
    # ===================================================
    
    seed_everything(seed)
    
    print("=" * 70)
    print("CALCULATING TEST ACCURACY WITH SPECIFIED HYPERPARAMETERS")
    print("=" * 70)
    print(f"\nModel Type: {model_type.upper()}")
    print(f"Random Seed: {seed}")
    print(f"\nHyperparameters:")
    print(f"  Layer Size:    {int(hyperparameters[0])}")
    print(f"  Epochs:        {int(hyperparameters[1])}")
    print(f"  Learning Rate: {hyperparameters[2]:.6f}")
    print(f"  Batch Size:    {int(hyperparameters[3])}")
    print(f"  Dropout Rate:  {hyperparameters[4]:.4f}")
    print(f"  Weight Decay:  {hyperparameters[5]:.6e}")
    print("=" * 70)
    
    # Convert to internal format
    hyperparams_tensor = convert_to_internal_format(hyperparameters)
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size:  {len(test_dataset)}")
    
    # Select model
    if model_type == 'nn':
        train_func = train_and_test_NN
        print("\nUsing SimpleNN model")
    else:
        train_func = train_and_test_CNN
        print("\nUsing CNN model")
    
    # Train and evaluate
    print("\n" + "-" * 70)
    print("Starting training and testing...")
    print("-" * 70 + "\n")
    
    test_accuracy = train_func((train_dataset, test_dataset), hyperparams_tensor, seed=seed)
    
    # Display results
    print("\n" + "=" * 70)
    print(f"{'FINAL RESULTS':^70}")
    print("=" * 70)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("=" * 70)
