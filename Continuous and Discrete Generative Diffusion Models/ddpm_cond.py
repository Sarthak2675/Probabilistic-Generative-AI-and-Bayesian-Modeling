from models import ConditionalDDPM
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from utils import seed_everything, compute_fid
from scheduler import NoiseSchedulerDDPM
import os
import time
import matplotlib.pyplot as plt

# Add any extra imports you want here

def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device, num_timesteps=1000, scheduler_type="linear"):
    
    train_start_time = time.time()
    model.train()
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_timesteps, type=scheduler_type, beta_start=0.0001, beta_end=0.02, s=0.008)
    for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev', 
                      'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                      'posterior_variance', 'posterior_log_variance_clipped',
                      'posterior_mean_coef1', 'posterior_mean_coef2']:
        if hasattr(scheduler, attr_name):
            setattr(scheduler, attr_name, getattr(scheduler, attr_name).to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss()
    
    losses = []
    fids = []
    fids_x = []
    for epoch in range(epochs):
        total_loss = 0
        time_start = time.time()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size_current = images.shape[0]
            
            timesteps = scheduler.sample_timesteps(batch_size_current, device)
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, timesteps)            
            predicted_noise = model(noisy_images, timesteps, labels)
            loss = mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        time_end = time.time()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs},-------------------- Average Loss: {avg_loss:.6f},----- Time: {time_end - time_start:.2f}s')
        losses.append(avg_loss)
        
        # if(epoch) % 5 == 0 or epoch == epochs - 1:
        #     fids_x.append(epoch)
        #     fid = get_score(model, device, test_loader, num_classes=10, num_samples=64, iter=5, num_steps=num_timesteps, scheduler_type=scheduler_type)
        #     fids.append(fid)
        #     if epochs%10 ==0 :
        #         torch.save(model.state_dict(), f"{run_name}/ddpm_cond_epoch{epoch}.pth")            

    torch.save(model.state_dict(), f"{run_name}/ddpm_cond.pth")
    print(f"Model saved to {run_name}/ddpm_cond.pth")
    torch.save(losses, f"{run_name}/losses.pt")
    torch.save(fids, f"{run_name}/fids.pt")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(fids_x, fids, label='FID Score', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.title('FID Score over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_name}/training_plots.png")
    # plt.show()
    
    train_end_time = time.time()
    print(f"Training completed in {(train_end_time - train_start_time)/60:.2f} minutes")


def sample(model, class_label, device, num_samples=16, num_steps=1000, scheduler_type="linear"):
    '''
    Returns:
        torch.Tensor, shape (num_samples, 1, 28, 28)
    ''' 
    model.eval()
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, type=scheduler_type, beta_start=0.0001, beta_end=0.02, s=0.008)
    for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                        'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                        'posterior_variance', 'posterior_log_variance_clipped',
                        'posterior_mean_coef1', 'posterior_mean_coef2']:
            if hasattr(scheduler, attr_name):
                setattr(scheduler, attr_name, getattr(scheduler, attr_name).to(device))
    
    x = torch.randn(num_samples, 1, 28, 28).to(device)
    class_labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
    
    with torch.no_grad():
        time_start = time.time()
        for i in range(num_steps-1, -1 ,-1):
            t = torch.full((num_samples,),i,device=device,dtype=torch.long)
            predicted_noise = model(x, t, class_labels)
            x_0_pred = scheduler.predict_start_from_noise(x, t, predicted_noise)
            
            if i>0:
                posterior_mean = scheduler.get_posterior_mean(x_0_pred, x, t)
                scheduler_variance = scheduler.posterior_variance[i]
                noise = torch.randn_like(x) if i > 0 else 0
                x = posterior_mean + torch.sqrt(scheduler_variance) * noise
            else:
                x = x_0_pred
        time_end = time.time()
    x = torch.clamp(x, 0,1)
    print(f"Sampling of {num_samples} samples completed in {time_end - time_start:.2f} seconds")
    
    return x

def get_score(model,device,test_loader,num_classes,num_samples=64, iter=5,num_steps=1000, scheduler_type="linear"):
    
    avg_fid=0.0
    for class_label in range(num_classes):    
        all_real = test_loader.dataset.data[test_loader.dataset.targets == class_label].unsqueeze(1).float() / 255.0
        all_generated = sample(model, class_label, device, num_samples=num_samples, num_steps=num_steps, scheduler_type=scheduler_type).cpu()
        avg_fid_c = 0.0
        for i in range(iter):
            rand_indices = torch.randperm(all_real.shape[0])[:num_samples]
            real_images = all_real[rand_indices]
            rand_indices = torch.randperm(all_generated.shape[0])[:num_samples]
            fake_images = all_generated[rand_indices]
            time_fid_start = time.time()
            fid = compute_fid(real_images, fake_images)
            time_fid_end = time.time()
            print(f"FID for class {class_label}, iteration {i+1}/{iter}: {fid.item():.2f}, Time: {time_fid_end - time_fid_start:.2f}s")
            avg_fid_c += fid.item()
        
        avg_fid_c /= iter
        avg_fid += avg_fid_c
    avg_fid /= num_classes    
    print(f"Average FID : {avg_fid}")
    
    return avg_fid
     


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Conditional Model Template")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample","eval_fid"], help="Mode: train or sample")
    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine"], help="Noise scheduler type")
    # Add any other arguments you want here
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ### Data Preprocessing Start ### (Do not edit this)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    ### Data Preprocessing End ### (Do not edit this)

    model = ConditionalDDPM(num_classes=10)
    model.to(device)

    run_name = f"exps_ddpm_cond/{args.epochs}ep_{args.learning_rate}lr_{args.num_steps}_ns_{args.scheduler}_scheduler" # Change run name based on your experiments
    os.makedirs(run_name, exist_ok=True)

    if args.mode == "train":
        model.train()
        train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device)

        fid = get_score(model, device, test_loader, num_classes=10, num_samples=64, iter=20, num_steps=args.num_steps, scheduler_type=args.scheduler)
        
    
    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{run_name}/ddpm_cond.pth"))
        model.eval()
        for class_num in range(10):
            samples = sample(model, class_num, device, args.num_samples, args.num_steps)
            torch.save(samples, f"{run_name}/{class_num}class_{args.num_samples}samples_{args.num_steps}steps.pt")
    elif args.mode == "eval_fid":
        model.load_state_dict(torch.load(f"{run_name}/ddpm_cond.pth"))
        model.eval()
        fid = get_score(model, device, test_loader, num_classes=10, num_samples=128, iter=20, num_steps=args.num_steps, scheduler_type=args.scheduler)
