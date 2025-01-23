import torch
import unittest
from torch.testing import assert_close
from models.rope_vit import (
    PositionEncoderBase,
    TiledPositionEncoder,
    LierePositionEncoder,
)
import os
import ipdb


def rotation_matrix(theta):
    theta = torch.tensor(theta)
    return torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )


def rotation_matrix_3d(theta1, theta2):
    return torch.matrix_exp(
        torch.tensor([[0.0, -theta2, 0], [theta2, 0, -theta1], [0, theta1, 0]])
    )


class TestPositionEncoding(unittest.TestCase):
    def test_base(self):
        encoder = PositionEncoderBase(
            image_size=[64, 64],
            patch_size=[16, 16],
            input_dimensionality=2,
            position_sequencing_type="sequential",
        )

        image_sizes = torch.tensor([[64, 64]])
        positions = encoder(image_sizes)

        expected_positions = torch.tensor(
            [
                [
                    [0.0000, 0.0000],
                    [0.0000, 0.3333],
                    [0.0000, 0.6667],
                    [0.0000, 1.0000],
                    [0.3333, 0.0000],
                    [0.3333, 0.3333],
                    [0.3333, 0.6667],
                    [0.3333, 1.0000],
                    [0.6667, 0.0000],
                    [0.6667, 0.3333],
                    [0.6667, 0.6667],
                    [0.6667, 1.0000],
                    [1.0000, 0.0000],
                    [1.0000, 0.3333],
                    [1.0000, 0.6667],
                    [1.0000, 1.0000],
                ]
            ]
        )

        assert_close(positions, expected_positions, atol=1e-4, rtol=1e-3)

    def test_tiled(self):
        encoder = TiledPositionEncoder(
            position_sequencing_type="sequential",
            patch_size=[16, 16],
            image_size=[32, 32],
            dim=8,
            phase_type="even",
            heads=1,
            input_dimensionality=2,
        )

        image_sizes = torch.tensor([[32, 32]])
        rotations = encoder(image_sizes, dtype=torch.bfloat16)
        rotations = rotations[:, 1:, ...]  # ignore the CLS token

        upper_left = torch.stack([torch.eye(2) for _ in range(4)])
        upper_right = torch.stack(
            [torch.eye(2) for _ in range(3)] + [rotation_matrix(1)]
        )
        lower_left = torch.stack(
            [torch.eye(2), rotation_matrix(1)] + [torch.eye(2) for _ in range(2)]
        )
        lower_right = torch.stack([torch.eye(2), rotation_matrix(1)] * 2)

        # [bs, num_tokens, head_dim // 2, 2, 2]
        expected_rotations = torch.stack(
            [upper_left, upper_right, lower_left, lower_right]
        ).reshape(1, 4, 4, 2, 2)
        assert_close(
            rotations, expected_rotations.to(dtype=torch.bfloat16), atol=2e-3, rtol=1e-3
        )

    # TODO Brian
    def test_liere(self):
        encoder = LierePositionEncoder(
            position_sequencing_type="sequential",
            patch_size=[16, 16],
            image_size=[32, 32],
            dim=8,
            phase_type="even",
            heads=1,
            depth=1,
            input_dimensionality=2,
            generator_dim=3,
            rotary_embedding_per_layer=False,
            rotary_embedding_per_head=False
        )

        image_sizes = torch.tensor([[32, 32]])
        rotations = encoder(image_sizes, dtype=torch.bfloat16)
        rotations = rotations[:, 1:, ...]  # ignore the CLs token

        id = torch.eye(3)
        upper_left = torch.stack([id, id])
        upper_right = torch.stack([id, rotation_matrix_3d(1, 0)])
        lower_left = torch.stack([id, rotation_matrix_3d(0, 1)])
        lower_right = torch.stack([id, rotation_matrix_3d(1, 1)])

        # [bs, num_tokens, head_dim // 3, 3, 3]
        expected_rotations = torch.stack(
            [upper_left, upper_right, lower_left, lower_right]
        ).view(1, 4, 2, 3, 3)

        assert_close(
            rotations, expected_rotations.to(dtype=torch.bfloat16), atol=2e-3, rtol=1e-3
        )
