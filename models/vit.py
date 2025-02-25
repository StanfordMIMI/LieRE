# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import ipdb

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import warnings
# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, positional_transforms=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout=0.0,
            checkpoint_attn=False
            ):
        super().__init__()
        self.checkpoint_attn = checkpoint_attn
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, positional_transforms=None):
        for i, (attn, ff) in enumerate(self.layers):
            if positional_transforms is not None:
                if len(positional_transforms.shape) == 6:
                    # This should only be called for liere and naver
                    if positional_transforms.shape[2] > 1:
                        # Use layer-specific transforms if available
                        # [1, num_token, layers, head, generator_dim, generator_dim]
                        positional_transforms = positional_transforms[
                            :, :, i, ...
                        ].unsqueeze(2)
                    else:
                        assert (
                            positional_transforms.shape[2] == 1
                        ), "positional_transforms.shape[2] must be 1 or equal to depth"
                else:
                    positional_transforms = positional_transforms.unsqueeze(2)

            x = attn(x, positional_transforms=positional_transforms) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        shuffle_patches=False,
        checkpoint_attn=False,
        enable_ape=True
    ):
        super().__init__()
        self.shuffle_patches = shuffle_patches
        self.enable_ape=enable_ape
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        #self.pos_embedding = nn.Parameter(torch.randn(1, 14*14 + 1, dim)) # not used
        self.pos_embedding_side_channel = None  # Brian added to deal with scaling.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, checkpoint_attn=checkpoint_attn)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))


    def forward(self, img, positional_transforms=None):

        x = self.to_patch_embedding(img)
        b, n, p = x.shape

        if self.shuffle_patches:
            torch.manual_seed(0)
            permutations = torch.stack([torch.randperm(n) for _ in range(b)]).to(
                x.device
            )
            permutations = permutations.unsqueeze(-1).expand(-1, -1, p)
            x = torch.gather(x, dim=1, index=permutations)

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embedding = (
            self.pos_embedding
            if self.pos_embedding_side_channel is None
            else self.pos_embedding_side_channel
        )
        if self.enable_ape:
            x += pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, positional_transforms=positional_transforms)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
