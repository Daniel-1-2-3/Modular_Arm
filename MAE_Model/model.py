from torch import nn, Tensor
import torch
import einops
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from MAE_Model.encoder import ViTMaskedEncoder
from MAE_Model.decoder import ViTMaskedDecoder

class MAEModel(nn.Module):
    def __init__(self, 
            patch_size: int = 8,
            encoder_embed_dim: int = 768,
            decoder_embed_dim: int = 512,
            encoder_heads: int = 16,
            decoder_heads: int = 16,
            in_channels: int = 3,
            img_h_size: int = 64,
            img_w_size: int = 64, 
            masking_ratio: float = 0.75,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.masking_ratio = masking_ratio
        self.in_channels = in_channels
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size

        self.num_patches = int((self.img_h_size * self.img_w_size) // (patch_size ** 2))
        
        self.encoder = ViTMaskedEncoder(
            patch_size=self.patch_size,
            embed_dim=self.encoder_embed_dim,
            in_channels=self.in_channels,
            img_h_size=self.img_h_size,
            img_w_size=self.img_w_size,
            heads=self.encoder_heads,
            masking_ratio=self.masking_ratio,
            depth=8
        )
        self.decoder = ViTMaskedDecoder(
            patch_size=self.patch_size,
            encoder_embed_dim=self.encoder_embed_dim,
            img_h_size=self.img_h_size,
            img_w_size=self.img_w_size,
            decoder_embed_dim=self.decoder_embed_dim,
            in_channels=self.in_channels,
            heads=self.decoder_heads,
            depth=4
        )
        self.out_proj = nn.Linear(decoder_embed_dim, self.patch_size ** 2 * in_channels)
    
    def forward(self, x: Tensor, mask_x: bool = True):
        """
        Args:
            x: single image, shape (B, H, W, C) or (B, C, H, W)
        Returns:
            out:  (B, num_patches, patch_size^2 * C)
            mask: (B, num_patches) with 0 unmasked, 1 masked
            z:    encoder tokens (your pipeline uses this)
        """
        x = self._to_bhwc(x)
        z, mask = self.encoder(x, mask_x)
        out = self.decoder(z, mask)
        out = self.out_proj(out)
        return out, mask, z
    
    def compute_loss(self, out: Tensor, truth: Tensor, mask: Tensor):
        mask = mask.to(out.device).float()
        truth_patchified = self.patchify(truth).to(out.device)

        loss_per_patch = F.mse_loss(out, truth_patchified, reduction='none').mean(dim=-1)
        den = mask.sum()
        loss = (loss_per_patch * mask).sum() / (den if den > 0 else loss_per_patch.numel())
        return loss
    
    def render_reconstruction(self, x: Tensor):
        x = self.unpatchify(x)[0]
        mean = torch.tensor([0.51905, 0.47986, 0.48809], device=x.device).view(3, 1, 1)
        std = torch.tensor([0.17454, 0.20183, 0.19598], device=x.device).view(3, 1, 1)
        x = x * std + mean
        x = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

        plt.imshow(x)
        plt.axis("off")
        plt.show()
        
    def patchify(self, x: Tensor):
        """
        Args:
            x: single image (B, H, W, C) or (B, C, H, W)
        Returns:
            (B, num_patches, patch_size^2 * C)
        """
        x = self._to_bhwc(x)
        # x: (B, H, W, C)
        return einops.rearrange(
            x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)',
            p1=self.patch_size, p2=self.patch_size
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, patch_size^2 * C)
        Returns:
            imgs: (B, C, H, W)
        """
        _, _, dim = x.shape
        c = dim // (self.patch_size ** 2)

        h = self.img_h_size // self.patch_size
        w = self.img_w_size // self.patch_size

        img = einops.rearrange(
            x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=c
        )
        return img

    def _to_bhwc(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            return x
        if x.shape[-1] == self.in_channels:
            return x
        if x.shape[1] == self.in_channels:
            return x.permute(0, 2, 3, 1).contiguous()
        if x.shape[2] == self.in_channels:
            return x.permute(0, 1, 3, 2).contiguous()
        return x