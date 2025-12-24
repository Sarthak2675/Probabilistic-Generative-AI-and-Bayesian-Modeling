### Add necessary imports ###
import torch
from torch.distributions import Normal
# Implement acquisition functions here

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Compute the Expected Improvement acquisition function.
    mu: Predicted means (num_test_samples, 1)
    sigma: Predicted standard deviations (num_test_samples, 1)
    f_best: Best observed function value
    
    Returns:
    ei: Expected Improvement values (num_test_samples, 1)
    """
    z = (mu - f_best - xi)/(sigma+ 1e-9)
    std_normal = Normal(loc=0.0,scale=1.0)
    std_cdf = std_normal.cdf(z)
    std_prob = torch.exp(std_normal.log_prob(z))
    
    return (mu - f_best -xi)*std_cdf + sigma*std_prob

def probability_of_improvement(mu, sigma, f_best, xi=0.01):
    """
    Compute the Probability of Improvement acquisition function.
    mu: Predicted means (num_test_samples, 1)
    sigma: Predicted standard deviations (num_test_samples, 1)
    f_best: Best observed function value
    xi: Exploration-exploitation trade-off parameter

    Returns:
    pi: Probability of Improvement values (num_test_samples, 1)
    """
    z = (mu - f_best - xi)/(sigma + 1e-9)
    std_normal = Normal(loc=0.0,scale=1.0)
    std_cdf = std_normal.cdf(z)
    return std_cdf