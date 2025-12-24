from models import ConditionalD3PM
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from utils import seed_everything, compute_fid
from scheduler import MaskSchedulerD3PM
import os


import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
# Add any extra imports you want here

# -----------------------------------
# helper function to extract correct values from schedule for a batch of timesteps - same as in d3pm.py file 
def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    # Move t to the correct device before gathering
    out = vals.gather(-1, t.to(vals.device)) 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
# -------------------------------------

# Forward Process
def q_sample(x_start, t, scheduler, mask_token_id=128):
    """
    Forward process for absorbing state D3PM. Corrupts the image by masking tokens.
    """
    alpha_cumprod_t = get_index_from_list(scheduler.alphas_cumprod, t, x_start.shape)
    noise = torch.rand_like(x_start.float())
    mask = (noise > alpha_cumprod_t).long()
    x_t = x_start * (1 - mask) + mask_token_id * mask
    return x_t

# Reverse Process and training objective
def q_posterior_logits(x_start, x_t, t, scheduler, mask_token_id=128, num_classes=256):
    """
    Calculates the logits of the posterior distribution q(x_{t-1} | x_t, x_0).
    This version is compatible with older PyTorch versions without the 'mask' kwarg.
    """
    # schedule constants
    beta_t = get_index_from_list(scheduler.betas, t, x_t.shape)
    alpha_cumprod_prev = get_index_from_list(scheduler.alphas_cumprod_prev, t, x_t.shape)

    # ensure tensors on correct device/dtype and get shapes
    device = x_start.device
    dtype = torch.float32
    B, _, H, W = x_start.shape

    # make sure indices are long and on device
    x_start = x_start.to(device=device).long()   # [B,1,H,W]
    x_t = x_t.to(device=device).long()           # [B,1,H,W]

    # ----- masked-case logits (x_t == mask) -----
    # initialize with very low logits
    logits_masked = torch.full((B, num_classes, H, W), -1e9, device=device, dtype=dtype)

    # scalars shaped [B,1,1,1]
    log_prob_xtm1_is_x_start = torch.log(beta_t * alpha_cumprod_prev + 1e-8)
    log_prob_xtm1_is_mask = torch.log((1 - beta_t) * (1 - alpha_cumprod_prev) + 1e-8)

    # build one-hot maps: [B, C, H, W]
    x_start_oh = F.one_hot(x_start.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    mask_tensor = torch.full_like(x_start.squeeze(1), mask_token_id)
    mask_oh = F.one_hot(mask_tensor, num_classes=num_classes).permute(0, 3, 1, 2).to(device=device, dtype=dtype)

    # expand scalar log-probs to [B,1,H,W] so broadcasting works
    src_xstart = log_prob_xtm1_is_x_start.expand(B, 1, H, W).to(device=device, dtype=dtype)
    src_mask = log_prob_xtm1_is_mask.expand(B, 1, H, W).to(device=device, dtype=dtype)

    # place values into class-channels via elementwise multiplication (avoids scatter_)
    logits_masked = logits_masked + (x_start_oh * src_xstart) + (mask_oh * src_mask)

    # ----- not-masked-case logits (deterministic posterior -> delta on x_start) -----
    logits_not_masked = torch.full((B, num_classes, H, W), -1e9, device=device, dtype=dtype)
    high_val = torch.tensor(1e9, device=device, dtype=dtype)
    logits_not_masked = logits_not_masked + (x_start_oh * high_val)

    # ----- combine according to whether x_t is mask token -----
    is_masked = (x_t == mask_token_id)  # [B,1,H,W] bool
    # expand mask to channel dim: [B,C,H,W]
    is_masked_expand = is_masked.expand(-1, num_classes, -1, -1)

    # pick masked vs not-masked logits per spatial location
    final_logits = torch.where(is_masked_expand.to(device=device), logits_masked, logits_not_masked)

    return final_logits



def p_losses(model, x_start, t, labels, scheduler, mask_token_id=128, num_classes=256):
    """
    Calculates the conditional D3PM loss using the cross-entropy objective (auxiliary loss mentioned in the paper).
    """
    x_t = q_sample(x_start, t, scheduler, mask_token_id)
    model_input = (x_t.float() / 127.5) - 1.0
    # labels as input for conditioning are now provided
    predicted_x0_logits = model(model_input, t, labels)
    loss = F.cross_entropy(predicted_x0_logits, x_start.squeeze(1))
    return loss

# Sampling Loop
@torch.no_grad()
def p_sample(model, x_t, t, labels, scheduler, mask_token_id=128, num_classes=256):
    """
    Single step of the reverse denoising process.
    """
    batch_size = x_t.shape[0]
    t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
    
    model_input = (x_t.float() / 127.5) - 1.0
    predicted_x0_logits = model(model_input, t_tensor, labels)
    predicted_x0 = torch.argmax(predicted_x0_logits, 1, keepdim=True)
    
    posterior_logits = q_posterior_logits(predicted_x0, x_t, t_tensor, scheduler, mask_token_id, num_classes)
    dist = torch.distributions.Categorical(logits=posterior_logits.permute(0, 2, 3, 1))
    x_prev = dist.sample().unsqueeze(1)
    return x_prev


def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device, resume_checkpoint=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MaskSchedulerD3PM(args.num_steps, mask_type=args.schedule, device=device)
    mask_token_id = 128
    num_pixel_classes = 256

    # resume logic
    start_epoch = 0
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint from {resume_checkpoint} ...")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming training from epoch {start_epoch}")

    print(f"Starting conditional training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (images, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            images = (images * 255).long().to(device)
            labels = labels.to(device)
            t = torch.randint(0, scheduler.num_timesteps, (images.shape[0],), device=device).long()
            loss = p_losses(model, images, t, labels, scheduler, mask_token_id, num_pixel_classes)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        # save checkpoint each epoch
        ckpt_path = f"{run_name}/checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, ckpt_path)
        torch.save(model.state_dict(), f"{run_name}/model_epoch_{epoch+1}.pth")

        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}: Evaluating model and calculating average FID...")
            model.eval()
            fid_score = get_score(model, device, test_loader, num_classes=10, num_samples=64, iter=5, num_steps=args.num_steps, scheduler_type=args.schedule)
            print(f"Epoch {epoch+1}: FID Score = {fid_score.item():.4f}")
            # torch.save(model.state_dict(), f"{run_name}/model_epoch_{epoch+1}.pth")
    print("Training finished.")
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, f"{run_name}/checkpoint_final.pth")
    torch.save(model.state_dict(), f"{run_name}/model.pth")


def get_score(model,device,test_loader,num_classes,num_samples=64, iter=5,num_steps=1000, scheduler_type="linear"):
    
    avg_fid=0.0
    for class_label in range(num_classes):    
        all_real = test_loader.dataset.data[test_loader.dataset.targets == class_label].unsqueeze(1).float() / 255.0
        # pass scheduler_type through to sample (sample now accepts scheduler_type alias)
        all_generated = sample(model,device, class_label, num_samples=num_samples, num_steps=num_steps, scheduler_type=scheduler_type).cpu()
        avg_fid_c = 0.0
        for i in range(iter):
            rand_indices = torch.randperm(all_real.shape[0])[:num_samples]
            real_images = all_real[rand_indices]
            rand_indices = torch.randperm(all_generated.shape[0])[:num_samples]
            fake_images = all_generated[rand_indices]
            fid = compute_fid(real_images, fake_images)
            avg_fid_c += fid.item()
        
        avg_fid_c /= iter
        avg_fid += avg_fid_c
    avg_fid /= num_classes    
    print(f"Average FID : {avg_fid}")
    
    return avg_fid


def sample(model,device, class_label = None, num_samples=16, num_steps=1000, schedule_type='cosine', scheduler_type=None):
    '''
    use class_label = None to generate random samples.
    Returns:
        torch.Tensor, shape (num_samples, 1, 28, 28)
    '''
    # Respect alias: if scheduler_type provided, use it
    if scheduler_type is not None:
        schedule_type = scheduler_type

    scheduler = MaskSchedulerD3PM(num_steps, mask_type=schedule_type, device=device)
    mask_token_id = 128
    img = torch.full((num_samples, 1, 28, 28), mask_token_id, dtype=torch.long, device=device)
    labels = None
    if class_label is not None:
        labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
    else:
        labels = torch.randint(0, 10, (num_samples,), dtype=torch.long, device=device)
    for t in tqdm(reversed(range(0, num_steps)), desc="Sampling", total=num_steps, leave=False):
        # call p_sample with correct argument order (model, x_t, t, labels, scheduler, ...)
        img = p_sample(model, img, t, labels, scheduler, mask_token_id)
    return img.float() / 255.0
    # raise NotImplementedError("Sampling function is not implemented.")

def parse_args():
    parser = argparse.ArgumentParser(description="D3PM Conditional Model Template")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample", "eval_fid"], help="Mode: train or sample")
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

    model = ConditionalD3PM(num_classes=10, num_pixel_vals=256)
    model.to(device)

    if args.run_name is None:
        run_name = f"exps_d3pm_cond/steps{args.num_steps}_sched{args.schedule}_{args.epochs}ep_{args.learning_rate}lr"
    else:
        run_name = f"exps_d3pm_cond/{args.run_name}"
    os.makedirs(run_name, exist_ok=True)
    print(f"Results will be saved in: {run_name}")

    if args.mode == "train":
        model.train()
        train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device, resume_checkpoint=args.resume_checkpoint)
        fid = get_score(model, device, test_loader, num_classes=10, num_samples=128, iter=25, num_steps=args.num_steps, scheduler_type=args.schedule)
    elif args.mode == "sample":
        # It's good practice to specify a checkpoint file if one exists
        checkpoint_path = f"{run_name}/checkpoint_final.pth"
        if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"Final model checkpoint not found in {run_name}. Please train the model first.")
        
        print(f"Loading model from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(f"Generating samples...")
        # Note: passing schedule_type from args to the sample function
        samples = sample(model, device, class_label=None, num_samples=args.num_samples, num_steps=args.num_steps, schedule_type=args.schedule)
        
        save_path_class = f"{run_name}/{args.num_samples}samples_{args.num_steps}steps_{args.schedule}sched.png"
        save_image(samples, save_path_class, nrow=int(args.num_samples**0.5))
        print(f"  Saved to {save_path_class}")
    elif args.mode == "eval_fid":
        checkpoint_path = f"{run_name}/checkpoint_final.pth"
        if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"Final model checkpoint not found in {run_name}. Please train the model first.")
        
        print(f"Loading model from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        fid = get_score(model, device, test_loader, num_classes=10, num_samples=128, iter=25, num_steps=args.num_steps, scheduler_type=args.schedule)
        print(f"FID Score = {fid.item():.4f}")
