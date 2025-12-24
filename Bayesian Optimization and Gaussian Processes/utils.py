### Add necessary imports ###
import random
import numpy as np
import torch
import math


def renormalize(hyperparams):
    hyperparams[2] = torch.pow(10,hyperparams[2])
    hyperparams[5] = torch.pow(10,hyperparams[5])
    hyperparams = [p.item() for p in hyperparams]
    hyperparams[0] = int(hyperparams[0])
    hyperparams[1] = int(hyperparams[1])
    hyperparams[0] = int(math.ceil( hyperparams[0]/100.0 )*100)
    hyperparams[3] = int(math.pow(2,int(math.ceil(hyperparams[3]))))
    return hyperparams
    
def seed_everything(seed):
    """Set random seed for all libraries to ensure reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
    """
    Compute the log-marginal likelihood.
    x_train: Training inputs (num_train_samples, num_hyperparameters)
    y_train: Training targets (num_train_samples, 1)

    Returns:
    log_likelihood: Scalar log-marginal likelihood value
    """
    n = x_train.shape[0]
    # kernel matrix computation 
    K = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(x_train[i], x_train[j], length_scale, sigma_f)
    # iid noise addition 
    K += noise*torch.eye(n)
    # inverse computation using pseudo inverse 
    K_inv = torch.linalg.pinv(K)

    eigenvals = torch.linalg.eigvals(K).real
    nonzero_eigenvals = eigenvals[eigenvals > 1e-10]

    # adding all three terms of log_likelihood
    log_likelihood = (-0.5 * y_train.t() @ K_inv @ y_train) \
        + (-0.5 * torch.sum(torch.log(nonzero_eigenvals))) \
        + (-(n / 2) * torch.log(torch.tensor(2 * torch.pi)))
    
    return log_likelihood.item()


def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """
    Optimize hyperparameters using grid search.
    x_train: Training inputs (num_train_samples, num_hyperparameters)
    y_train: Training targets (num_train_samples, 1)

    Returns:
    best_length_scale: Optimized length scale
    best_sigma_f: Optimized signal variance
    """
    best_logL = -float('inf')
    best_len_scale = 1.0 
    best_sigma_f = 1.0 

    length_scales = [0.1, 0.5, 1.0, 2.0,5.0]
    sigma_fs = [0.1, 0.5, 1.0, 2.0,5.0]

    for l in length_scales:
        for sf in sigma_fs:
            log_likeliood = log_marginal_likelihood(x_train, y_train, kernel_func, l, sf, noise)
            if log_likeliood > best_logL:
                best_logL = log_likeliood
                best_len_scale = l
                best_sigma_f = sf

    return best_len_scale, best_sigma_f

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    """
    Perform GP prediction. Return mean and standard deviation of predictions.
    x_train: Training inputs (num_train_samples, num_hyperparameters)
    y_train: Training targets (num_train_samples, 1)
    x_test: Test inputs (num_test_samples, num_hyperparameters)
    kernel_func: Kernel function to use

    Returns:
    mu_s: Predicted means (num_test_samples, 1)
    sigma_s: Predicted standard deviations (num_test_samples, 1)
    """
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    K = torch.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(n_train):
            K[i, j] = kernel_func(x_train[i], x_train[j], length_scale, sigma_f)

    K += noise * torch.eye(n_train)

    K_s = torch.zeros((n_train, n_test))
    for i in range(n_train):
        for j in range(n_test):
            K_s[i, j] = kernel_func(x_train[i], x_test[j], length_scale, sigma_f)
    
    K_ss_diag = torch.zeros(n_test)
    for i in range(n_test):
        K_ss_diag[i] = kernel_func(x_test[i], x_test[i], length_scale, sigma_f)

    K_inv = torch.linalg.pinv(K)
    mu_s = K_s.t() @ K_inv @ y_train
    cov_diag = K_ss_diag - torch.sum((K_s.t() @ K_inv) * K_s.t(), dim=1)
    # non-negative variance for numerical stability 
    cov_diag = torch.clamp(cov_diag, min=0)
    sigma_s= torch.sqrt(cov_diag.reshape(-1, 1) + noise)

    return mu_s, sigma_s
