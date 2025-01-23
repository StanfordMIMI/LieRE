from dataclasses import dataclass
from typing import Sequence, Optional
from enum import Enum

@dataclass
class ModelArgs:
    num_channels: Sequence | int | None = 3
    patch_size: Sequence | int | None = None
    input_dimensionality: int = 1

    num_classes: int = 100

    emb_dropout: float = 0.1
    attn_dropout: float = 0.1

    mixup: bool = False

    pretrained_weights_path: str = None

    shuffle_patches: bool = False
    freeze_liere: bool = False
    rotary_embedding_per_layer: bool = False
    rotary_embedding_per_head: bool = False
    checkpoint_attn: bool = False
    generator_dim: int = 64

    model_size: str = "tiny"
    model_architecture: str = "absolute"

    @property
    def size_params(self):
        size_configs = {
            "tiny": {"dim": 384, "depth": 12, "heads": 6, "mlp_dim": 768},
            "base": {"dim": 768, "depth": 12, "heads": 12, "mlp_dim": 3072},
            "large": {"dim": 1024, "depth": 24, "heads": 16, "mlp_dim": 4096},
            "huge": {"dim": 1280, "depth": 32, "heads": 16, "mlp_dim": 5120},
        }
        if self.model_size not in size_configs:
            raise ValueError(f"Invalid model size '{self.model_size}'. Choose from {list(size_configs.keys())}.")
        return size_configs[self.model_size]