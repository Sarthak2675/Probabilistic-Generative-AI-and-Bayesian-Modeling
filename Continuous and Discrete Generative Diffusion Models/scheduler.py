import torch
import torch.nn.functional as F
import math

class NoiseSchedulerDDPM():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", beta_start=0.0001, beta_end=0.02, s = 0.008):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(beta_start=beta_start, beta_end=beta_end)
        elif type == "cosine":
            self.init_cosine_schedule(s=s)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") 

    def init_cosine_schedule(self, s=0.008):
        '''cosine scheduler initialization'''
        steps = self.num_timesteps + 1
        t = torch.linspace(0, self.num_timesteps, steps, dtype=torch.float32)
        
        f_t = torch.cos(((t / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        
        alphas_cumprod_t = f_t / f_t[0]        
        betas = 1. - (alphas_cumprod_t[1:] / alphas_cumprod_t[:-1])        
        self.betas = torch.clamp(betas, 0.0, 0.999)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def init_linear_schedule(self, beta_start=0.0001, beta_end=0.02):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_start, noise, timesteps):
        """
        Add noise to clean images according to noise schedule.
        This implements the forward diffusion process q(x_t | x_0).
        
        Args:
            x_start: Clean images [batch_size, channels, height, width]
            noise: Gaussian noise [batch_size, channels, height, width]
            timesteps: Timestep for each image in batch [batch_size]
        
        Returns:
            Noisy images at specified timesteps
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size, device):
        """
        Sample random timesteps for training.
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
    
    def get_posterior_mean(self, x_start, x_t, t):
        """
        Compute the posterior mean for p(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1_t = self.posterior_mean_coef1[t][:, None, None, None]
        posterior_mean_coef2_t = self.posterior_mean_coef2[t][:, None, None, None]
        
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        return posterior_mean
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def __len__(self):
        return self.num_timesteps
    
class MaskSchedulerD3PM():
    """
    Mask scheduler for Discrete Diffusion (D3PM) models.

    Args:
        num_timesteps: int, number of timesteps in the diffusion process
        mask_type: str, type of mask scheduling ("uniform", "linear", etc.)
        **kwargs: additional arguments for mask scheduling

    This object sets up the mask schedule for each timestep.
    """        
    def __init__(self, num_timesteps=1000, mask_type="cosine", device='cpu', **kwargs):
        self.num_timesteps = num_timesteps
        self.mask_type = mask_type
        self.device = device

        if mask_type == "linear":
            self.betas = self.init_linear_schedule()
        elif mask_type == "cosine":
            self.betas = self.init_cosine_schedule()
        else:
            raise NotImplementedError(f"{mask_type} mask scheduler is not implemented")

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # Pad with 1.0 at the beginning for t=0, which is alphas_cumprod[-1] effectively
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
    
    def init_linear_schedule(self):
        """
        linear schedule for the absorbing state with the probability of masking at step t being beta_t = 1 / (T - t + 1)
        which leads to a roughly linear increase
        in the number of masked tokens over time
        """
        timesteps = self.num_timesteps
        t = torch.arange(1, timesteps + 1, dtype=torch.float32, device=self.device)
        betas = 1 / (timesteps - t + 1)
        return betas

    def init_cosine_schedule(self, s=0.008):
        """
        cosine schedule as proposed in the reference paper (also theoretically justified for masked discrete diffusion)
        """
        steps = self.num_timesteps + 1
        t = torch.linspace(0, self.num_timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((t / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # Calculate betas from the cumulative alphas
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def __len__(self):
        return self.num_timesteps