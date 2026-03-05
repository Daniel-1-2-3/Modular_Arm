import torch
from torch import nn, Tensor
from MAE_Model.vit import VitBlock

class ViTMaskedDecoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 8,
        encoder_embed_dim: int = 768,
        img_h_size: int = 64,
        img_w_size: int = 64,
        decoder_embed_dim: int = 512,
        in_channels: int = 3,
        heads: int = 8,
        depth: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.in_channels = in_channels
        self.heads = heads
        self.depth = depth

        self.num_total_patches = int((img_h_size * img_w_size) // (patch_size ** 2))

        self.proj = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)
        self.mask_tokens = nn.Parameter(torch.randn(1, 1, self.decoder_embed_dim))
        self.pos_embeds = nn.Parameter(torch.randn(1, self.num_total_patches, self.decoder_embed_dim))

        self.vit_blocks = nn.ModuleList([
            VitBlock(
                embed_dim=self.decoder_embed_dim,
                heads=self.heads,
                mlp_ratio=4.0,
                attn_drop_rate=0.1,
                mlp_drop_rate=0.1,
                path_drop_rate=0.05
            ) for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(normalized_shape=self.decoder_embed_dim, eps=1e-6)

    def forward(self, x: Tensor, mask: Tensor):
        x = self.proj(x)

        b, k, d = x.shape
        n = int(mask.shape[1])

        unmasked = (mask == 0)
        keep = unmasked.sum(dim=1)
        if not torch.all(keep == k):
            raise ValueError(f"x has {k} tokens but mask keeps {keep.tolist()}")

        if self.pos_embeds.shape[1] < n:
            raise ValueError(f"pos_embeds has {self.pos_embeds.shape[1]} patches but need {n}")

        x_full = self.mask_tokens.expand(b, n, d).clone()
        idx3 = unmasked.unsqueeze(-1).expand(b, n, d)
        x_full.masked_scatter_(idx3, x)

        x = x_full + self.pos_embeds[:, :n]
        for block in self.vit_blocks:
            x = block(x)
        x = self.norm(x)
        return x