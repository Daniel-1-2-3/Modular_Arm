import torch
from torch import nn, Tensor
import numpy as np
from MAE_Model.sincos_pos_embeds import PosEmbed
from MAE_Model.vit import VitBlock
import random

class ViTMaskedEncoder(nn.Module):
    def __init__(self, 
            nviews: int = 2,
            patch_size: int = 6,
            embed_dim: int = 768,
            in_channels: int = 3,
            img_h_size: int = 84,
            img_w_fused_size: int = 168,
            heads: int = 8,
            depth: int = 8,
            masking_ratio: float = 0.75,
        ):
        
        super().__init__()
        self.nviews = nviews
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.img_h_size = img_h_size
        self.img_w_fused_size = img_w_fused_size
        self.heads = heads
        self.depth = depth
        self.masking_ratio = masking_ratio
        
        with torch.no_grad():
            each_view_w = img_w_fused_size // nviews
            each_view_h = img_h_size
            pe_np = PosEmbed.get_2d_sincos_pos_embed(
                embed_dim, int(each_view_h // patch_size), int(each_view_w // patch_size)
            )
            pe = torch.from_numpy(pe_np).repeat(nviews, 1)
        self.register_buffer("pos_embed_all", pe, persistent=False)
        
        self.forward_conv = self.construct_conv_layers()
        self.vit_blocks = nn.ModuleList([
            VitBlock(
                embed_dim=self.embed_dim,
                heads=self.heads,
                mlp_ratio=4.0,
                attn_drop_rate=0.1,
                mlp_drop_rate=0.1,
                path_drop_rate=0.05
            ) for _ in range(self.depth)
        ])
        self.norm_masked = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-6)
        self.norm_unmasked = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-6)

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

    def forward(self, x: Tensor, mask_x: bool):
        x = self._to_bhwc(x)
        
        x = self.forward_early_conv(x)
        
        x = self.add_pos_embeds(x)
        mask = None
        if mask_x:
            x, mask = self.random_view_masking(x)

        for block in self.vit_blocks:
            x = block(x)
        
        x = self.norm_masked(x) if mask_x else self.norm_unmasked(x)
        return x, mask

    def random_view_masking(self, x: torch.Tensor):
        device = x.device
        batch, num_patches, embed_dim = x.shape
        half = num_patches // 2

        mask_ratio = (self.masking_ratio - 0.5) / 0.5
        num_mask = int(mask_ratio * half)
        keep_per_half = half - num_mask

        mask_right = torch.rand(batch, device=device) > 0.5
        scores = torch.rand(batch, half, device=device)
        keep_left  = torch.topk(-scores, keep_per_half, dim=1).indices
        keep_right = torch.topk(-scores, keep_per_half, dim=1).indices + half

        keep_idx = torch.where(mask_right.unsqueeze(1), keep_left, keep_right)
        mask = torch.ones(batch, num_patches, device=device, dtype=torch.float32)
        row = torch.arange(batch, device=device).unsqueeze(1)
        mask[row, keep_idx] = 0.0

        gather_idx = keep_idx.unsqueeze(-1).expand(batch, keep_per_half, embed_dim)
        x_masked = torch.gather(x, dim=1, index=gather_idx)

        return x_masked, mask
    
    def add_pos_embeds(self, x: Tensor):
        return x + self.pos_embed_all.to(dtype=x.dtype)
    
    def construct_conv_layers(self):
        layers = []
        in_channels = self.in_channels

        stride_factors = []
        
        n = self.patch_size
        while n > 1:
            if n % 2 == 0:
                stride_factors.append(2)
                n //= 2
            else:
                stride_factors.append(n)
                break

        for i, s in enumerate(stride_factors):
            out_channels = max(self.embed_dim // (2 ** (len(stride_factors) - i)), 32)
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=max(3, s),
                stride=s,
                padding=max(3, s) // 2
            ))
            layers.append(nn.ReLU())
            in_channels = out_channels

        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.embed_dim,
            kernel_size=1,
            stride=1
        ))

        forward_conv = nn.Sequential(*layers)
        return forward_conv
        
    def forward_early_conv(self, x: Tensor):
        x = self._to_bhwc(x)
        
        batch, height, width_total, channels = x.shape

        width_per_view = width_total // self.nviews
        x = torch.split(x, width_per_view, dim=2)
        x = torch.cat(x, dim=0)
        x = x.permute(0, 3, 1, 2)
        
        x = self.forward_conv(x)
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = torch.chunk(x, self.nviews, dim=0)
        x = torch.cat(x, dim=1)
        
        return x