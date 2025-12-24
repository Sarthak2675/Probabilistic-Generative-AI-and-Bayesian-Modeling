### Add necessary imports ###
import torch
import math
### Implement various kernel functions here ###

def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Compute the RBF (Gaussian) kernel."""
    return (sigma_f**2)*torch.exp(-torch.sum(torch.square(x1-x2))/(2*(length_scale**2)))

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Compute the Matern kernel."""
    return (sigma_f**2)*(1 + (math.sqrt(2*nu)*torch.sum(torch.abs(x1-x2))/length_scale))*torch.exp(- math.sqrt(2*nu)*torch.sum(torch.abs(x1-x2))/length_scale)

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Compute the Rational Quadratic kernel."""
    return (sigma_f**2)*torch.pow(1 + (torch.sum(torch.square(x1-x2))/(2*alpha*(length_scale**2))),-alpha)