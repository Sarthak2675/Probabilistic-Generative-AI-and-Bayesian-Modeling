from models import D3PM
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from utils import seed_everything, compute_fid
from scheduler import MaskSchedulerD3PM
import os


import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
# Add any extra imports you want here
from torchvision.utils import save_image

# -----------------------------------
# helper function to extract correct values from schedule for a batch of timesteps
def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    # Move t to the correct device before gathering
    out = vals.gather(-1, t.to(vals.device)) # gather requires index on same device
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
# ----------------------------------------

# Forward Process (scheduled noise addition in a given number of steps -> p(x_T) ~ N(0, 1))
def q_sample(x_start, t, scheduler, mask_token_id=128):
    """
    Forward process for absorbing state D3PM. Corrupts the image by masking tokens.
    x_start: The initial clean image (batch, 1, 28, 28), with pixel values 0-255.
    t: Timestep for each image in the batch.
    """
    alpha_cumprod_t = get_index_from_list(scheduler.alphas_cumprod, t, x_start.shape) # cumulative product of alphas 
    # probabilistically mask a certain number of pixels 
    noise = torch.rand_like(x_start.float()) 
    # pixels are to be kept with probability alpha_cumprod_t
    mask = (noise > alpha_cumprod_t).long()
    x_t = x_start * (1 - mask) + mask_token_id * mask
    return x_t



# Reverse Process and training objective
def q_posterior_logits(x_start, x_t, t, scheduler, mask_token_id=128, num_classes=256):
    """
    Calculates the logits of the posterior distribution q(x_{t-1} | x_t, x_0).
    Fixed scatter_ usage: expands src tensors to match index shape [B,1,H,W].
    """
    # schedule constants (shapes: [B,1,1,1] after get_index_from_list)
    beta_t = get_index_from_list(scheduler.betas, t, x_t.shape)
    alpha_cumprod_prev = get_index_from_list(scheduler.alphas_cumprod_prev, t, x_t.shape)

    # shapes and device
    B, _, H, W = x_start.shape
    device = x_start.device
    dtype = torch.float32

    # ensure indices and masks are long and on same device
    x_start = x_start.to(device=device).long()          # [B,1,H,W]
    x_t = x_t.to(device=device).long()

    # when x_t is the mask token 
    # The posterior becomes a distribution between x_start or x_1 and the mask token (mentioned in class slides)H, W]
    logits_masked = torch.full((B, num_classes, H, W), -1e9, device=device, dtype=dtype)

    # compute log probs
    log_prob_xtm1_is_x_start = torch.log(beta_t * alpha_cumprod_prev + 1e-8)          # [B,1,1,1]
    log_prob_xtm1_is_mask = torch.log((1 - beta_t) * (1 - alpha_cumprod_prev) + 1e-8)  # [B,1,1,1]

    # expand src to match spatial dimensions
    log_prob_xtm1_is_x_start = log_prob_xtm1_is_x_start.expand(B, 1, H, W)
    log_prob_xtm1_is_mask = log_prob_xtm1_is_mask.expand(B, 1, H, W)

    # scatter for x_start position
    logits_masked.scatter_(1, x_start, log_prob_xtm1_is_x_start)

    # scatter for mask position
    mask_index = torch.full_like(x_start, mask_token_id)
    logits_masked.scatter_(1, mask_index, log_prob_xtm1_is_mask)

    # when x_t is not the mask token 
    # x_{t-1} should have been the same as x_t (and x_start)
    # the posterior is a delta function (deterministic)
    logits_not_masked = torch.full((B, num_classes, H, W), -1e9, device=device, dtype=dtype)
    high_val = torch.tensor(1e9, device=device, dtype=dtype)
    high_val = high_val.expand(B, 1, H, W)
    logits_not_masked.scatter_(1, x_start, high_val)

    # combine the two cases based on whether x_t is mask token using torch.where 
    is_masked = (x_t == mask_token_id)   # [B,1,H,W], bool
    # need is_masked to broadcast to [B, C, H, W]; expand along channel dim
    is_masked_expand = is_masked.to(device=device, dtype=torch.bool).expand(-1, num_classes, -1, -1)

    # torch.where picks from logits_masked where is_masked True, else from logits_not_masked
    final_logits = torch.where(is_masked_expand, logits_masked, logits_not_masked)

    return final_logits



def p_losses(model, x_start, labels, t, scheduler, mask_token_id=128, num_classes=256):
    """
    Calculates the D3PM loss.
    This is simply the cross-entropy loss of predicting x0, which is an effective
    and stable objective mentioned in the D3PM paper (as the auxiliary loss).
    """
    x_t = q_sample(x_start, t, scheduler, mask_token_id)
    # normalized input is fed to the model 
    model_input = (x_t.float() / 127.5) - 1.0
    predicted_x0_logits = model(model_input, t)
    # cross-entropy between the model's prediction and the true x_0
    loss = F.cross_entropy(predicted_x0_logits, x_start.squeeze(1))
    return loss

    # raise NotImplementedError("Training loop is not implemented.")
def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device, resume_checkpoint=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MaskSchedulerD3PM(args.num_steps, mask_type=args.schedule, device=device)
    mask_token_id = 128
    num_pixel_classes = 256

    # If resume checkpoint provided, load model + optimizer and set start epoch
    start_epoch = 0
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint from {resume_checkpoint} ...")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming training from epoch {start_epoch}")

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (images, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            # conversion images from [0, 1] float to [0, 255] long
            images = (images * 255).long().to(device)
            labels = labels.to(device)
            t = torch.randint(0, scheduler.num_timesteps, (images.shape[0],), device=device).long()
            loss = p_losses(model, images, labels, t, scheduler, mask_token_id, num_pixel_classes)
            loss.backward()
            optimizer.step()
            # tqdm progress bar
            progress_bar.set_postfix(loss=loss.item())

        # Save checkpoint at end of each epoch (model + optimizer + epoch)
        # ckpt_path = f"{run_name}/checkpoint_epoch_{epoch+1}.pth"
        # torch.save({
        #     "epoch": epoch,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict()
        # }, ckpt_path)
        # also save legacy model-only file like before
        # torch.save(model.state_dict(), f"{run_name}/model_epoch_{epoch+1}.pth")

        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}: Evaluating model and calculating FID...")
            model.eval()
            class_num = 0  # or any class, or random, but to match traceback
            generated_samples = sample(model, class_num, device, num_samples=128, num_steps=args.num_steps, schedule_type=args.schedule)
            fid_score = get_score(test_loader,generated_samples)
            print(f"Epoch {epoch+1}: FID Score = {fid_score:.4f}")

    print("Training finished.")
    # final save
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, f"{run_name}/checkpoint_final.pth")
    torch.save(model.state_dict(), f"{run_name}/model.pth")


@torch.no_grad()
def p_sample(model, x_t, t, class_label, scheduler, mask_token_id=128, num_classes=256):
    """
    Single step of the reverse denoising process.
    """
    t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
    model_input = (x_t.float() / 127.5) - 1.0
    # predicting logits of the original image x0
    predicted_x0_logits = model(model_input, t_tensor)
    # final prediction for x0 determined by taking the argmax
    predicted_x0_dist = torch.distributions.Categorical(logits=predicted_x0_logits.permute(0, 2, 3, 1))
    # 2. Sample the predicted x0 image
    predicted_x0 = predicted_x0_dist.sample().unsqueeze(1) 
    posterior_logits = q_posterior_logits(predicted_x0, x_t, t_tensor, scheduler, mask_token_id, num_classes)
    # sampling from this posterior distribution to get x_{t-1}
    dist = torch.distributions.Categorical(logits=posterior_logits.permute(0, 2, 3, 1))
    x_prev = dist.sample().unsqueeze(1)
    
    return x_prev

def get_score(test_loader, all_generated, num_samples=64, iter=5):
    """calculates FID score between generated samples and test dataset"""
    all_real = test_loader.dataset.data.unsqueeze(1).float() / 255.0
    avg_fid = 0.0
    for i in range(iter):
        print("Computing FID iteration:", i+1)
        real_batch = all_real[torch.randperm(all_real.size(0))[:num_samples]]
        generated_batch = all_generated[torch.randperm(all_generated.size(0))[:num_samples]]
        fid = compute_fid(real_batch, generated_batch)
        avg_fid += fid.item()
    
    avg_fid /= iter
    print(f"Average FID : {avg_fid}")
    
    return avg_fid

def sample(model, class_num, device, num_samples=16, num_steps=1000, schedule_type='cosine'):
    '''
    Returns:
        torch.Tensor, shape (num_samples, 1, 28, 28)
    '''
    print(f"Starting sampling for {num_samples} images with {num_steps} steps...")
    scheduler = MaskSchedulerD3PM(num_steps, mask_type=schedule_type, device=device)
    mask_token_id = 128
    # pure noise as a starting point 
    img = torch.full((num_samples, 1, 28, 28), mask_token_id, dtype=torch.long, device=device)
    class_label = torch.full((num_samples,), class_num, dtype=torch.long, device=device)
    # Iteratively denoise from timestep T-1 down to t=0
    for t in tqdm(reversed(range(0, num_steps)), desc="Sampling", total=num_steps):
        img = p_sample(model, img, t, class_label, scheduler, mask_token_id)
    print("Sampling finished.")
    # images retuned in the standard 0-1 float format for FID calculation
    return img.float() / 255.0

def parse_args():
    parser = argparse.ArgumentParser(description="D3PM Model Template")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample","eval_fid"], help="Mode: train or sample")
    # Add any other arguments you want here
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"], help="Noise schedule to use")
    parser.add_argument("--run_name", type=str, default=None, help="A name for the experiment run")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
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

    model = D3PM()
    model.to(device)

    if args.run_name is None:
        run_name = f"exps_d3pm/steps{args.num_steps}_sched{args.schedule}_{args.epochs}ep_{args.learning_rate}lr"
    else:
        run_name = f"exps_d3pm/{args.run_name}"
    os.makedirs(run_name, exist_ok=True)
    print(f"Results will be saved in: {run_name}")

    if args.mode == "train":
        model.train()
        train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device, resume_checkpoint=args.resume_checkpoint)
    
    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{run_name}/model.pth", map_location=device))
        model.eval()
        class_num = 0  # choose a class for sampling
        samples = sample(model, class_num, device, args.num_samples, args.num_steps)
        torch.save(samples, f"{run_name}/{args.num_samples}samples_{args.num_steps}steps.pt")
        # also saving the generated samples for reference 
        save_path = f"{run_name}/{args.num_samples}samples_{args.num_steps}steps_{args.schedule}sched.png"
        save_image(samples, save_path, nrow=int(args.num_samples**0.5))
        print(f"Generated samples saved to {save_path}")
    
    elif args.mode == "eval_fid":
        model.load_state_dict(torch.load(f"{run_name}/model.pth", map_location=device))
        model.eval()
        samples = sample(model,0,device,num_samples = 128, num_steps = args.num_steps)
        fid = get_score(test_loader,samples,iter=25)