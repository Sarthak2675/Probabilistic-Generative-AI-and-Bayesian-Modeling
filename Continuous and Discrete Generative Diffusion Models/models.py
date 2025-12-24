import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.permute(0, 2, 3, 1).view(B, H * W, C)
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(C), dim=-1)
        out = attn @ v
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        
        return self.proj(out) + x

class UNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=128, out_channels=1, dropout=0.1, num_classes=None):
        super().__init__()
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)    
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        self.down1 = ResBlock(model_channels, model_channels * 2, time_embed_dim, dropout)
        self.down2 = ResBlock(model_channels * 2, model_channels * 4, time_embed_dim, dropout)
        
        self.downsample1 = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(model_channels * 4, model_channels * 4, 3, stride=2, padding=1)
        
        # Middle
        self.mid_block1 = ResBlock(model_channels * 4, model_channels * 4, time_embed_dim, dropout)
        self.attention = AttentionBlock(model_channels * 4)
        self.mid_block2 = ResBlock(model_channels * 4, model_channels * 4, time_embed_dim, dropout)
        
        # Decoder
        self.up1 = ResBlock(model_channels * 8, model_channels * 2, time_embed_dim, dropout)
        self.up2 = ResBlock(model_channels * 4, model_channels, time_embed_dim, dropout)
        
        self.upsample1 = nn.ConvTranspose2d(model_channels * 4, model_channels * 4, 4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, class_labels=None):

        time_emb = self.time_embed(timesteps)
        if self.num_classes is not None:
            if class_labels is None:
                raise ValueError("class_labels must be provided for conditional model")
            else:
                class_emb = self.class_embed(class_labels)
                time_emb = time_emb + class_emb
                        
        # Encoder
        h1 = self.conv_in(x)
        h2 = self.down1(h1, time_emb)
        h2_down = self.downsample1(h2)
        h3 = self.down2(h2_down, time_emb)
        h3_down = self.downsample2(h3)
        
        # Middle
        h = self.mid_block1(h3_down, time_emb)
        h = self.attention(h)
        h = self.mid_block2(h, time_emb)
        
        # Decoder
        h = self.upsample1(h)
        h = torch.cat([h, h3], dim=1)
        h = self.up1(h, time_emb)
        h = self.upsample2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up2(h, time_emb)
        
        return self.conv_out(h)

# ----------------------
# U-Net model has been implemented for image generation using diffusion models 
# Helper functions and blocks are as defined below 
class positional_emb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embs = math.log(10 ** 4) / (half_dim - 1)
        embs = torch.exp(torch.arange(half_dim, device=device) * (-1 * embs))
        embs = time[:, None] * embs[None, :]
        embs = torch.cat((embs.sin(), embs.cos()), dim = -1)
        return embs


class nn_block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False, down=True, up_output_padding=0):
        """
        Minimal changes:
        - added `down` flag: down=False => preserve spatial dimensions (Identity transform)
        - added `up_output_padding` to let ConvTranspose2d produce the correct spatial size
          for specific up blocks (fixes mismatches for odd-sized feature maps).
        """
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            # upsampling block expects concatenated skip connection (2 * in_channels)
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)
            # allow specifying output_padding per up block to match spatial sizes exactly
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=up_output_padding)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            if down:
                # downsampling conv (same behavior as original code)
                self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            else:
                # preserve spatial size (used for middle blocks)
                self.transform = nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.SiLU()

    def forward(self, x, t):
        h = self.bn1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[..., None, None]
        h = h + time_emb
        h = self.bn2(self.relu(self.conv2(h)))
        return self.transform(h)
# -------------------------      



# -----------------------------------------------------

class D3PM(nn.Module):
    def __init__(self, num_classes=256, dim=64): # Add any required parameters
        super().__init__()
        img_channels = 1
        down_channels = (dim, dim * 2, dim * 4, dim * 8)
        up_channels = (dim * 8, dim * 4, dim * 2, dim)
        out_dim = img_channels
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(positional_emb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU())
        self.conv0 = nn.Conv2d(img_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([nn_block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)])
        # choose output_padding per up-block to match the skip sizes (minimal change to fix mismatches)
        up_output_paddings = [1 if i == 0 else 0 for i in range(len(up_channels) - 1)]
        self.ups = nn.ModuleList([nn_block(up_channels[i], up_channels[i+1], time_emb_dim, up=True, up_output_padding=up_output_paddings[i]) for i in range(len(up_channels) - 1)])

        # Critical: mid blocks should NOT downsample further. set down=False.
        self.mid_conv1 = nn_block(down_channels[-1], down_channels[-1], time_emb_dim, up=False, down=False)
        self.mid_conv2 = nn_block(down_channels[-1], down_channels[-1], time_emb_dim, up=False, down=False)

        self.out = nn.Conv2d(up_channels[-1], num_classes, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        res_in = []
        for down in self.downs:
            x = down(x, t)
            res_in.append(x)
        
        x = self.mid_conv1(x, t)
        x = self.mid_conv2(x, t)

        for up in self.ups:
            res_x = res_in.pop()
            # res_x and x must have same spatial size here because of carefully chosen up_output_padding per up-block
            x = torch.cat((x, res_x), dim = 1)
            x = up(x, t)
        return self.out(x)
        
    

        # Define your model architecture here
        pass

class ConditionalD3PM(nn.Module):
    def __init__(self, num_classes, dim = 64, num_pixel_vals=256): # Add any required parameters
        super().__init__()
        self.num_classes = num_classes
        # Define your conditional model architecture here
        img_channels = 1
        down_channels = (dim, dim*2, dim*4, dim*8)
        up_channels = (dim*8, dim*4, dim*2, dim)
        time_emb_dim = 32
        out_dim  = img_channels

        self.time_mlp = nn.Sequential(positional_emb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU())
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        self.conv0 = nn.Conv2d(img_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([nn_block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])
        up_output_paddings = [1 if i == 0 else 0 for i in range(len(up_channels) - 1)]
        self.ups = nn.ModuleList([nn_block(up_channels[i], up_channels[i+1], time_emb_dim, up=True, up_output_padding=up_output_paddings[i]) for i in range(len(up_channels)-1)])

        # Also ensure mid convs here do not downsample.
        self.mid_conv1 = nn_block(down_channels[-1], down_channels[-1], time_emb_dim, up=False, down=False)
        self.mid_conv2 = nn_block(down_channels[-1], down_channels[-1], time_emb_dim, up=False, down=False)

        self.out = nn.Conv2d(up_channels[-1], num_pixel_vals, 1)

    def forward(self, x, timestep, labels):
        t = self.time_mlp(timestep)
        labels_emb = self.label_emb(labels)
        t = t + labels_emb
        x = self.conv0(x)
        res_in = []
        for down in self.downs:
            x = down(x, t)
            res_in.append(x)

        x = self.mid_conv1(x, t)
        x = self.mid_conv2(x, t)

        for up in self.ups:
            res_x = res_in.pop()
            x = torch.cat((x, res_x), dim=1)
            x = up(x, t)

        return self.out(x)

class DDPM(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, model_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.unet = UNet(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            dropout=0.1
        )

    def forward(self, x, timesteps):
        return self.unet(x, timesteps)

class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.unet = UNet(
            in_channels =1,
            model_channels=64,
            out_channels=1,
            dropout=0.1,
            num_classes=num_classes
        )
    def forward(self, x, timesteps, class_labels):
        return self.unet(x, timesteps, class_labels)