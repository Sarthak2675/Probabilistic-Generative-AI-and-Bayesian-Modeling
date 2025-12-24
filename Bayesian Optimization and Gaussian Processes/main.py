### Add necessary imports ###
import os
from acquisition_functions import expected_improvement, probability_of_improvement
from kernels import rbf_kernel, matern_kernel, rational_quadratic_kernel
from train_test import train_and_test_NN, train_and_test_CNN
from utils import gaussian_process_predict, optimize_hyperparameters, seed_everything, renormalize
import argparse
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
import time

def get_rand_samples(num_samples, lower_bound, upper_bound):
    samples = torch.rand(num_samples, lower_bound.shape[0])
    return lower_bound + samples * (upper_bound - lower_bound)

def normalize(x, lower_bound, upper_bound):
    return (x - lower_bound) / (upper_bound - lower_bound)

def parse_args():
    parser = argparse.ArgumentParser(description='Train and Test Models with Hyperparameters')
    # Add arguments as per requirements
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--model_type', type=str, choices=['nn', 'cnn'], default='nn', help='Type of model to use')
    parser.add_argument('--acquisition_function', type=str, choices=['ei', 'pi'], default='ei', help='Acquisition function to use')
    parser.add_argument('--kernel', type=str, choices=['rbf', 'matern', 'rational_quadratic'], default='rbf', help='Kernel function to use')
    parser.add_argument('--max_budget', type=int, default=25, help='Maximum budget for hyperparameter optimization')
    parser.add_argument('--init_points', type=int, default=10, help='Number of initial random points for hyperparameter optimization')
    return parser.parse_args()

def bo_loop(train_val_datasets,lower_bounds, upper_bounds, kernel_func, acquisition_func, train_func, max_budget, init_points):
    accrs = []
    x_train = torch.zeros(init_points, 6)

    # Initial random sampling
    x_train = get_rand_samples(init_points, lower_bounds, upper_bounds)
    for i in range(init_points):
        acc = train_func(train_val_datasets, x_train[i])
        accrs.append(acc)
        print(f'Initial Point {i+1}/{init_points}, Hyperparameters: {x_train[i].tolist()}, Accuracy: {acc:.4f}')
    
    y_train = torch.tensor(accrs).view(-1,1)
    bo_accrs = []
    for step in range(max_budget - init_points):

        start_time = time.time()
        # Sample candidate points from hyperparameter space
        num_candidates = 1000
        candidates = get_rand_samples(num_candidates, lower_bounds, upper_bounds)
        normalized_candidates = normalize(candidates, lower_bounds, upper_bounds)
        normalized_x_train = normalize(x_train, lower_bounds, upper_bounds)
        
        length_scale, sigma_f = optimize_hyperparameters(normalized_x_train, y_train, kernel_func)
        mu, sigma = gaussian_process_predict(normalized_x_train, y_train, normalized_candidates, kernel_func,length_scale=length_scale, sigma_f=sigma_f)
        
        f_best = max(accrs)
        acquisition_vals = acquisition_func(mu, sigma, f_best)
        next_idx = torch.argmax(acquisition_vals)
        next_point = candidates[next_idx]
        middle_time = time.time()
        acc = train_func(train_val_datasets, next_point)
        accrs.append(acc)
        bo_accrs.append(acc)
        y_train = torch.cat([y_train, torch.tensor([[acc]])], dim=0)
        x_train = torch.cat([x_train, next_point.unsqueeze(0)], dim=0)
        final_time = time.time()
        print(f'BO Step {step+1}/{max_budget-init_points}, Hyperparameters: {next_point.tolist()}, Accuracy: {acc:.4f}')
        print(f'Time taken for GP prediction and acquisition function: {middle_time - start_time:.2f} seconds')
        print(f'Time taken for training and testing: {final_time - middle_time:.2f} seconds')
        
    best_idx = torch.argmax(torch.tensor(accrs))
    best_hyperparams = x_train[best_idx]
    best_accuracy = accrs[best_idx]

    return best_hyperparams, best_accuracy , bo_accrs  


if __name__ == '__main__':
    st = time.time()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    args = parse_args()
    seed_everything(args.seed)

    assert args.max_budget >= args.init_points, "max_budget should be greater than init_points"

    if args.kernel == 'rbf':
        kernel_func = rbf_kernel
    elif args.kernel == 'matern':
        kernel_func = matern_kernel
    elif args.kernel == 'rational_quadratic':
        kernel_func = rational_quadratic_kernel

    if args.acquisition_function == 'ei':
        acquisition_func = expected_improvement
    elif args.acquisition_function == 'pi':
        acquisition_func = probability_of_improvement

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_validation_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_size = int(0.8 * len(train_validation_dataset))
    validation_size = len(train_validation_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_validation_dataset, [train_size, validation_size])
    train_val_datasets = (train_dataset, validation_dataset) # Give this as input to train_and_test_NN or train_and_test_CNN functions
    hyperparam_space = {
        'layer_size' : [100,200,300,400,500],
        'epochs' : torch.arange(1,11, 1),
        'log_lr': torch.arange(-5,-1, 0.5),
        'batch_size': [16,32,64,128,256],
        'dropout_rate': torch.arange(0,0.51,0.05),
        'log_weight_decay' : torch.arange(-6,-1.5,0.5)
    }
    lower_bounds = torch.tensor([100, 1, -5.0,4, 0.0, -6.0])
    upper_bounds = torch.tensor([500, 10, -1.0,8, 0.5, -1.5])
    
    if args.model_type == 'nn':
        train_func = train_and_test_NN
        print("Using SimpleNN")
    else:  # cnn
        train_func = train_and_test_CNN
        print("Using CNN")
    
    best_hyperparams, best_accuracy, accrs = bo_loop(
        train_val_datasets, lower_bounds, upper_bounds, kernel_func, acquisition_func,
        train_func, args.max_budget, args.init_points
    )
    
    print("Testing on the test dataset with the best hyperparameters...")
    print(f"Best Hyperparameters : {best_hyperparams}")
    test_accuracy = train_func((train_validation_dataset, test_dataset), best_hyperparams)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    plt.plot(range(1, len(accrs)+1), accrs)
    plt.xlabel('Iteration')
    plt.ylim(0, 1)
    plt.ylabel('Validation Accuracy')
    plt.title(f"BO Progress: {args.model_type.upper()} with {args.kernel} kernel and {args.acquisition_function.upper()} acquisition")
    plt.grid()
    plt.savefig(f"./results/{args.model_type}_{args.max_budget}_{args.model_type}_{args.kernel}_{args.acquisition_function}_bo_progress.png")
    
    with open(f'results/{args.model_type}{args.max_budget}_{args.model_type}_{args.kernel}_{args.acquisition_function}_results.txt', 'w') as f:
        f.write(f'Best Hyperparameters: {renormalize(best_hyperparams)}\n')
        f.write(f'Best Validation Accuracy: {best_accuracy:.4f}\n')
        f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
    
    print(f"Best Hyperparameters: {best_hyperparams}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    et = time.time()
    print(f"Total Execution Time: {(et - st)/60:.2f} minutes")