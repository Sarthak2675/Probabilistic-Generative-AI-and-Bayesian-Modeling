import torch
from ddpm import sample as ddpm_sample
from ddpm_cond import sample as ddpm_cond_sample
from d3pm import sample as d3pm_sample
from d3pm_cond import sample as d3pm_cond_sample


if __name__ == "__main__":
    
    ## DDPM 
    ddpm_model = torch.load("ddpm.pth")
    ddpm_samples = ddpm_sample(ddpm_model, device="cuda", num_samples=64, num_steps=1000, scheduler_type="linear")
    torch.save(ddpm_samples, "samples_ddpm.pth")
    
    ## DDPM-Cond
    ddpm_cond_model = torch.load("ddpm_cond.pth")
    for digit in range(10):
        ddpm_cond_samples = ddpm_cond_sample(ddpm_cond_model, device="cuda", num_samples=64, num_steps=1000, scheduler_type="linear", class_label=digit)
        torch.save(ddpm_cond_samples, f"samples_ddpm_cond_{digit}.pth")
    
    ## D3PM
    d3pm_model = torch.load("d3pm.pth")
    d3pm_samples = d3pm_sample(d3pm_model, device="cuda", num_samples=64, num_steps=1000, scheduler_type="cosine")
    torch.save(d3pm_samples, "samples_d3pm.pth")
    
    ## D3PM-Cond
    d3pm_cond_model = torch.load("d3pm_cond.pth")
    for digit in range(10):
        d3pm_cond_samples = d3pm_cond_sample(d3pm_cond_model, device="cuda", num_samples=64, num_steps=1000, scheduler_type="cosine", class_label=digit)
        torch.save(d3pm_cond_samples, f"samples_d3pm_cond_{digit}.pth")

    
    
    