from collections.abc import Iterable
import numpy as np
import ipdb
import math
from enum import Enum

import torch
import torch.nn as nn
import math
from .vit import ViT, PreNorm, Attention, FeedForward
from dataclasses import dataclass, field
import yaml
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def triplet(t):
    if isinstance(t, tuple) and len(t) == 3:
        return t
    elif isinstance(t, tuple) and len(t) == 2:
        return (t[0], t[1], t[0])  # Repeat the first element if there are only two
    else:
        return (t, t, t)  # Repeat the single value three times


class DummyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, image_sizes, dtype):
        return None


class PositionEncoderBase(nn.Module):
    def __init__(
        self, image_size, patch_size, input_dimensionality, position_sequencing_type
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        if not isinstance(self.patch_size, Iterable):
            self.patch_size = [self.patch_size] * input_dimensionality
        if not isinstance(self.image_size, Iterable):
            self.image_size = [self.image_size] * input_dimensionality
        self.expected_steps_per_axis = tuple(
            image_dim_size // patch_dim_size for image_dim_size, patch_dim_size in zip(self.image_size, self.patch_size))
        self.position_sequencing_type = position_sequencing_type

    def forward(self, image_sizes: torch.Tensor):
        assert (
            image_sizes.shape[0] == 1
        )  # only support one image size for the batch for now
        image_size = image_sizes.tolist()[0]
        steps_per_axis = tuple(
            math.ceil(dim_size / patch_dim)
            for dim_size, patch_dim in zip(image_size, self.patch_size)
        )

        # Default is to scale zero to 1.
        max_values_per_dim = tuple(step_per_axis / expected_step_per_axis for step_per_axis, expected_step_per_axis in zip(steps_per_axis, self.expected_steps_per_axis))

        assert (
            self.position_sequencing_type == "sequential"
        ), "Only sequential encoding is supported for now."

        # scale 0 to 1.
        scale_0_to_1 = False
        if scale_0_to_1:
            normalized_positions = torch.cartesian_prod(
                *(
                    torch.linspace(0, 1, steps, device=image_sizes.device)
                    for steps in steps_per_axis
                )
            )
        else:
            normalized_positions = torch.cartesian_prod(
                *(
                    torch.linspace(0, scale, steps, device=image_sizes.device)
                    for scale, steps in zip(max_values_per_dim, steps_per_axis)
                )
            )
        return normalized_positions.unsqueeze(0)


class TiledPositionEncoder(PositionEncoderBase):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        phase_type,
        position_sequencing_type,
        heads,
        input_dimensionality,
    ):
        super().__init__(
            image_size, patch_size, input_dimensionality, position_sequencing_type
        )

        self.phase_type = phase_type
        self.head_dim = dim // heads

        self.input_dimensionality = input_dimensionality

    def forward(self, image_sizes: torch.Tensor, dtype):

        assert image_sizes.shape[-1] == self.input_dimensionality

        num_phases = self.head_dim // self.input_dimensionality // 2

        if self.phase_type == "inverse_exp":
            phases = torch.exp(
                -4 * torch.linspace(0, 1, num_phases, device=image_sizes.device)
            )
            phases = torch.stack([phases] * self.input_dimensionality)
        # for testing for now
        elif self.phase_type == "even":
            phases = torch.linspace(0, 1, num_phases, device=image_sizes.device)
            phases = torch.stack([phases] * 2)
        else:
            raise NotImplementedError(f"Phase type {self.phase_type} not implement")

        base = torch.tensor([[0, -1], [1, 0]], device=image_sizes.device)

        # shape = [num_phases, 2, 2]
        bases = phases.reshape(self.input_dimensionality, num_phases, 1, 1) * base

        # shape = [bs, num_tokens, dimensionality]
        positions = super().forward(image_sizes)

        # shape = [bs, num_tokens, dimensionality, num_phases, 2, 2]
        generators = positions.reshape(list(positions.shape) + [1] * 3) * bases
        # CLS token gets identity transform. Note that the second dim is the token dim.
        cls_generator = torch.zeros_like(generators[:, 0, ...]).unsqueeze(1)
        generators = torch.cat((cls_generator, generators), dim=1)

        rotations = torch.matrix_exp(generators.to(dtype=torch.float32)).to(dtype=dtype)
        # shape = [bs, num_tokens, num_phases*dimensionality, 2, 2]
        bs_and_tokens_shape = list(positions.shape[:-1])
        bs_and_tokens_shape[1] += 1
        return rotations.reshape(bs_and_tokens_shape + [-1, 2, 2])

class ExponentiationMethod(Enum):
    RETURN_POSITIONS = 1
    EXPONENTIATE_ONE_SHOT = 2
    EXPONENTIATE_LOOP = 3

class LierePositionEncoder(PositionEncoderBase):
    def __init__(
        self,
        image_size,
        patch_size,
        dim: int,
        phase_type: str,
        position_sequencing_type: str,
        heads: int,
        input_dimensionality: int,
        generator_dim: int,
        depth: int,
        rotary_embedding_per_layer: bool,
        rotary_embedding_per_head: bool,
        cls_token=True,
    ):
        super().__init__(
            image_size, patch_size, input_dimensionality, position_sequencing_type
        )
        self.cls_token = cls_token
        self.input_dimensionality = input_dimensionality
        self.head_dim = dim // heads
        self.phase_type = phase_type

        print(f"Starting with liere generator dim {generator_dim}")
        self.generator_dim = generator_dim

        self.num_generators = self.head_dim // self.generator_dim

        # whether to apply a different 2D rotation to each layer or not
        self.rotary_embedding_per_layer = rotary_embedding_per_layer
        self.rotary_embedding_per_head = rotary_embedding_per_head
        generators_for_layers = 1
        
        self.initialization_factor = 2 * math.pi
        # RoPE-Mixed drew random angles from zero to 2 pi. https://github.com/naver-ai/rope-vit/blob/c6aa201ee795daa4f841e2f9585164bb23a0b819/deit/models_v2_rope.py#L25

        if self.rotary_embedding_per_layer:
            generators_for_layers = depth
            # if layer ipdb> generator_init.shape ipdb torch.Size([2, 12, 32, 2, 2])

        if self.rotary_embedding_per_head:
            self.num_generators = self.num_generators * heads
            # if layer and head ipdb> generator_init.shape torch.Size([2, 12, 384, 2, 2])

        # initialize the generator parameters
        # https://github.com/naver-ai/rope-vit/blob/c6aa201ee795daa4f841e2f9585164bb23a0b819/deit/models_v2_rope.py#L150C13-L150C76
        generator_init = (
            torch.rand(
                self.input_dimensionality,
                generators_for_layers,
                self.num_generators,
                self.generator_dim,
                self.generator_dim,
            ) * self.initialization_factor 
        )
        # Shape: (generators_for_layers, input_dimensionality, num_generators, generator_dim, generator_dim)

        self.generator_raw_params = None
        if self.phase_type in ("learned", "naver", "learned_inverse_exp"):
            self.generator_raw_params = nn.Parameter(
                generator_init.clone(), requires_grad=True
            )
            print(f"Initialized generator params to {self.generator_raw_params.shape}")

        self.generator_transform = nn.Identity()

    def forward(self, image_sizes: torch.Tensor | None, dtype, positions=None):
        assert (image_sizes is None) != (
            positions is None
        ), f"Exactly one of image sizes and positions must be None. Image sizes: {image_sizes}, positions: {positions}"
        # shape = [bs, num_tokens, dimensionality]
        if positions is None:
            positions = super().forward(image_sizes)

        if self.phase_type in ("learned", "naver"):
            # Shape: (generators_for_layers, input_dimensionality, num_generators, generator_dim, generator_dim)
            upper_triangle = (
                self.generator_transform(
                    torch.triu(self.generator_raw_params, diagonal=1)
                )
                - 0.5
            )  # sigmoid maps 0 to 0.5

            # Shape: (generators_for_layers, input_dimensionality, num_generators, generator_dim, generator_dim)
            bases = upper_triangle - torch.transpose(upper_triangle, -1, -2)

            # .unsqueeze(0) for the repeat dimension, # shape = [bs, num_tokens, dimensionality]
            in_basis_positions = (
                positions.reshape(list(positions.shape) + [1] * 4) * bases
            )

            generator_pos = torch.sum(in_basis_positions, dim=-5)  # sum over dimensions
        elif self.phase_type in ["even"]:
            # shape = [2,3,3] before reshape
            bases = torch.tensor(
                [
                    [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                ],
                device=image_sizes.device,
            ).reshape((1, 1, 2, 3, 3))

            num_phases = self.head_dim // self.generator_dim
            phases = torch.linspace(0, 1, num_phases, device=image_sizes.device)

            # shape = [bs, num_tokens, 1, 3, 3]
            in_basis_positions = (
                positions.reshape(list(positions.shape) + [1] * 2) * bases
            )
            # shape = [bs, num_tokens, num_phases, 3, 3]
            generator_pos = torch.sum(in_basis_positions, dim=-3, keepdim=True)
            generator_pos = generator_pos * phases.reshape(1, 1, num_phases, 1, 1)
        else:
            raise NotImplementedError(self.phase_type)

        # add an identity for the CLS token.
        if self.cls_token:
            cls_generator = torch.zeros_like(generator_pos[:, 0, ...]).unsqueeze(1)
            generator_pos = torch.cat((cls_generator, generator_pos), dim=1)
        
        # print("The shape of generater_pos is: ", generator_pos.shape)
        # Commented out for experiment with learned bases.
        
        exp_method = ExponentiationMethod.EXPONENTIATE_ONE_SHOT
        # more memory efficient, especially for 3D, but nummerically not the same
        # exp_method = ExponentiationMethod.EXPONENTIATE_LOOP

        match exp_method:
            case ExponentiationMethod.RETURN_POSITIONS:
                return generator_pos
            case ExponentiationMethod.EXPONENTIATE_ONE_SHOT:
                return torch.matrix_exp(generator_pos.to(dtype=torch.float32)).to(dtype=dtype)
            case ExponentiationMethod.EXPONENTIATE_LOOP:
                exponentiated_results = []
                for i in range(generator_pos.shape[2]):
                    # we used torch.float32 in the paper, torch.float16 could be sufficient and much more efficient
                    exponentiated_results.append(torch.matrix_exp(generator_pos[:, :, i:i+1,...].to(dtype=torch.float32).contiguous().to(dtype=dtype)))
                return torch.cat(exponentiated_results, dim=2)
            case _:
                raise NotImplementedError()


def identities_like(existing_rotations, num_new_tokens):
    transform_shape = list(existing_rotations.shape)
    transform_shape[1] = num_new_tokens
    transform = torch.ones(
        list(transform_shape[:-2]) + [1, 1], device=existing_rotations.device
    ) * torch.eye(transform_shape[-1], device=existing_rotations.device)
    return transform


class FlexibleAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        dropout,
        rotary_embedding_per_head,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        assert self.dim_head == 64, "Only support 64 head size for now."

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        self.learnable_heads_rotations = self.heads if rotary_embedding_per_head else 1

    def apply_transforms(self, x, positional_transforms):
        generator_dim = positional_transforms.shape[-1]
        batch_size, num_heads, num_tokens, head_size = x.shape
        num_rotators = positional_transforms.shape[-3] // self.learnable_heads_rotations

        rotatable_dim = generator_dim * num_rotators

        assert head_size == self.dim_head, "Head dims and head size have to be the same"
        # shape [batch size, heads_num, tokens_num, head_dim]
        rotatable_states = x[:, :, :, :rotatable_dim]
        unrotatable_states = x[:, :, :, rotatable_dim:]

        states_split = rotatable_states.reshape(
            (
                batch_size,
                num_heads,
                num_tokens,
                num_rotators,  # num_rotators,
                generator_dim,
                1,
            )
        )
        # positional_transforms.shape: [1, num_tokens, 1, num_rotators, generator_dim, generator_dim]
        assert positional_transforms.shape[0] == 1, f"positional_transforms.shape[0]: {positional_transforms.shape[0]}"
        assert positional_transforms.shape[2] == 1, f"positional_transforms.shape[2]: {positional_transforms.shape[2]}"
        positional_transforms = positional_transforms.squeeze(2) # artifact of learnable over layers not being handled consistently.

        positional_transforms = positional_transforms.view(
            (
                1,
                num_tokens,
                self.learnable_heads_rotations,
                num_rotators,
                generator_dim,
                generator_dim,
            )
        )
        # Align the token dimensions
        positional_transforms = positional_transforms.transpose(1, 2)

        rotated_states = torch.matmul(positional_transforms, states_split)
        return torch.cat(
            [rotated_states.flatten(start_dim=-3), unrotatable_states], axis=-1
        )

    def forward(self, x, positional_transforms=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if positional_transforms is not None:
            # k is transformed in the next step.

            q, k = self.apply_transforms(
                q, positional_transforms
            ), self.apply_transforms(k, positional_transforms)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class AbsolutePositionEmbeddingContext:
    def __init__(self, image_size: Iterable | int, target):
        self.input_dimensionality = target.input_dimensionality

        def ensure_list(val):
            if isinstance(val, int):
                return [val] * self.input_dimensionality
            return val

        self.image_size = ensure_list(image_size)
        self.model_image_size = ensure_list(target.image_size)
        self.patch_size = ensure_list(target.patch_size)

        self.target = target
        self.hidden_dim = target.pos_embedding.shape[-1]
        assert len(image_size) == len(
            self.model_image_size
        ), f"Dimensionality mismatch. Images are {image_size} and expected {self.model_image_size}."
        assert len(self.patch_size) == len(
            self.image_size
        ), f"Patch size mismatch {self.patch_size} for images {self.image_size}."

    def __enter__(self):
        if tuple(self.image_size) == tuple(self.model_image_size):
            self.original_pos_embed = None
            return

        self.original_pos_embed = self.target.pos_embedding

        original_patch_arrangement = [
            math.ceil(image_dim / patch_dim - 1e-6)
            for image_dim, patch_dim in zip(self.model_image_size, self.patch_size)
        ]
        num_patches = np.prod(original_patch_arrangement)
        assert (
            num_patches + 1 == self.original_pos_embed.shape[-2]
        ), f"Number of patches does not match number of pos embeddings. Expected {num_patches}, got embed shape of {self.original_pos_embed.shape}."

        new_patch_arrangement = [
            math.ceil(image_dim / patch_dim - 1e-6)
            for image_dim, patch_dim in zip(self.image_size, self.patch_size)
        ]
        num_new_patches = np.prod(new_patch_arrangement)

        class_pos_embed = self.original_pos_embed[:, 0]
        patch_pos_embed = self.original_pos_embed[:, 1:].reshape(
            [1] + list(original_patch_arrangement) + [self.hidden_dim]
        )

        interpolation_permutation = (
            (0, 3, 1, 2) if self.input_dimensionality == 2 else (0, 4, 1, 2, 3)
        )
        inverse_permutation = (
            (0, 2, 3, 1) if self.input_dimensionality == 2 else (0, 2, 3, 4, 1)
        )
        patch_pos_embed = patch_pos_embed.permute(*interpolation_permutation)
        interpolated_embeddings = nn.functional.interpolate(
            patch_pos_embed,
            size=new_patch_arrangement,
            mode="bicubic" if self.input_dimensionality == 2 else "trilinear",
            align_corners=False,
        )
        interpolated_embeddings = interpolated_embeddings.permute(
            *inverse_permutation
        ).view(1, -1, self.hidden_dim)

        new_embeds = torch.cat(
            (class_pos_embed.unsqueeze(0), interpolated_embeddings), dim=1
        ).clone()

        self.target.pos_embedding_side_channel = new_embeds

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_pos_embed == None:
            return
        self.target.pos_embedding_side_channel = None


class RoPEViT(ViT):
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
        positional_encoding_type,
        input_dimensionality=2,
        phase_type=None,  # rope
        position_sequencing_type="sequential",  # rope
        generator_dim=None,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
        force_absolute_encodings=False,
        rotary_embedding_per_layer=False,
        rotary_embedding_per_head=False,
        checkpoint_attn=False,
        shuffle_patches=False,
        enable_ape=True,
    ):
        super().__init__(
            image_size=image_size[1] if type(image_size) == list else image_size,
            patch_size=patch_size[1] if type(patch_size) == list else patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            shuffle_patches=shuffle_patches,
            checkpoint_attn=checkpoint_attn,
            enable_ape=enable_ape
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_dimensionality = input_dimensionality
        self.rotary_embedding_per_layer = rotary_embedding_per_layer
        self.rotary_embedding_per_head = rotary_embedding_per_head

        if input_dimensionality == 3:
            if isinstance(image_size, int):
                image_height, image_width, image_depth = triplet(image_size)
                patch_height, patch_width, patch_depth = triplet(patch_size)
            elif isinstance(image_size, list) and len(image_size) == 3:
                image_depth, image_height, image_width = image_size
                patch_depth, patch_height, patch_width = patch_size
            else:
                raise ValueError(
                    "`image_size` must be an integer or a tuple of three integers."
                )
            assert (
                image_height % patch_height == 0
                and image_width % patch_width == 0
                and image_depth % patch_depth == 0
            ), "Image dimensions must be divisible by the patch size."

            num_patches = (
                (image_height // patch_height)
                * (image_width // patch_width)
                * (image_depth // patch_depth)
            )
            patch_dim = channels * patch_height * patch_width * patch_depth

            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    "b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)",
                    p1=patch_depth,
                    p2=patch_height,
                    p3=patch_width,
                ),
                nn.Linear(patch_dim, dim),
            )
            # for absolute
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        if positional_encoding_type == "absolute":
            self.position_encoder = None
        else:
            if not force_absolute_encodings:
                ref = self.pos_embedding
                self.pos_embedding.data = torch.zeros_like(
                    ref, device=ref.device, dtype=ref.dtype
                )
                self.pos_embedding.requires_grad = False
            if positional_encoding_type == "tiled":
                self.position_encoder = TiledPositionEncoder(
                    image_size=image_size,
                    patch_size=patch_size,
                    dim=dim,
                    phase_type=phase_type,
                    position_sequencing_type=position_sequencing_type,
                    heads=heads,
                    input_dimensionality=input_dimensionality,
                )
            elif positional_encoding_type == "liere":
                self.position_encoder = LierePositionEncoder(
                    image_size=image_size,
                    patch_size=patch_size,
                    dim=dim,
                    phase_type=phase_type,
                    position_sequencing_type=position_sequencing_type,
                    heads=heads,
                    input_dimensionality=input_dimensionality,
                    generator_dim=generator_dim,
                    rotary_embedding_per_layer=self.rotary_embedding_per_layer,
                    depth=depth,
                    rotary_embedding_per_head=self.rotary_embedding_per_head,
                )
            else:
                raise NotImplementedError(positional_encoding_type)

        transformer_layers = nn.ModuleList([])
        for idx, layer in enumerate(self.transformer.layers):
            transformer_layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            FlexibleAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                rotary_embedding_per_head=self.rotary_embedding_per_head,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout=dropout,
                            ),
                        ),
                    ]
                )
            )
        self.transformer.layers = transformer_layers

    def forward(self, input_ids: torch.tensor):
        image_size = input_ids.shape[2:]

        assert len(image_size) == self.input_dimensionality, f"Input dimensionality mismatch, {len(image_size)} == {self.input_dimensionality}"
        if self.position_encoder is not None:

            image_sizes = torch.tensor(image_size, device=input_ids.device).unsqueeze(0)
            positional_transforms = self.position_encoder(
                image_sizes, dtype=input_ids.dtype
            )
            return super().forward(
                img=input_ids, positional_transforms=positional_transforms
            )

        # Absolute encodings (basically standard ViT)
        with AbsolutePositionEmbeddingContext(image_size, self):
            return super().forward(input_ids)
