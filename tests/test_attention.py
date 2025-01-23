# You can run this test using `pytest tests/test_attention.py`
import torch
import unittest
from torch.testing import assert_close

# from models.vit import Attention
from models.rope_vit import FlexibleAttention
import ipdb


def to_identity(linear):
    parameters = list(linear.parameters())
    parameters[0].data = torch.eye(*parameters[0].shape)
    parameters[1].data = torch.zeros_like(parameters[1])


def duplicate_parameter_values(a, b):
    for p1, p2 in zip(a.parameters(), b.parameters()):
        p1.data = p2.data.clone()


def construct_absolute_attention(dim, heads):
    torch.manual_seed(0)
    attention_layer = FlexibleAttention(
        dim=dim, heads=heads, dim_head=dim // heads, dropout=0.0, rotary_embedding_per_head=False
    )
    # zero out the bias for reproducibility
    attention_layer.to_out[0].bias.data = torch.zeros_like(
        attention_layer.to_out[0].bias.data
    )
    return attention_layer


def construct_relative_attention(dim, heads):
    torch.manual_seed(0)
    attention_layer = FlexibleAttention(
        dim=dim, heads=heads, dim_head=dim // heads, dropout=0.0, rotary_embedding_per_head=False
    )
    # zero out the bias for reproducibility
    attention_layer.to_out[0].bias.data = torch.zeros_like(
        attention_layer.to_out[0].bias.data
    )
    return attention_layer

# positional_transforms.shape: [1, num_tokens, 1, num_rotators, generator_dim, generator_dim]
operator_shape = (1, 3, 1, 12 // 2 // 3, 1, 1)
token_shape = (2, 3, 64 * 3)


class TestAttention(unittest.TestCase):
    def test_absolute_attention(self):
        attention_layer = construct_absolute_attention(dim=64 * 3, heads=3)

        torch.manual_seed(0)
        tokens = torch.zeros(token_shape)
        result = attention_layer(x=tokens)
        self.assertEqual(result.shape, tokens.shape)
        assert_close(result, torch.zeros_like(result))

    def test_relative_encoding_shapes(self):
        attention_layer: FlexibleAttention = construct_absolute_attention(
            dim=64 * 3, heads=3
        )

        # First test that the shapes are correct. [batch_size, number_heads, head_dim]
        tokens = torch.zeros(token_shape)
        identity_rotation = torch.ones(operator_shape) * torch.eye(
            2, device=tokens.device
        )
        result = attention_layer(x=tokens, positional_transforms=identity_rotation)
        self.assertEqual(result.shape, tokens.shape)
        assert_close(result, torch.zeros_like(result))

    def test_relative_encoding_identity_noop(self):
        torch.manual_seed(0)
        tokens = torch.rand(token_shape)
        identity_rotation = torch.ones(operator_shape) * torch.eye(
            2, device=tokens.device
        )

        attention_layer: FlexibleAttention = construct_relative_attention(
            dim=64 * 3, heads=3
        )
        absolute_attention_layer = construct_absolute_attention(dim=64 * 3, heads=3)
        duplicate_parameter_values(absolute_attention_layer, attention_layer)
        reference_results = absolute_attention_layer(x=tokens)
        result = attention_layer(x=tokens, positional_transforms=identity_rotation)
        assert_close(result, reference_results, atol=1e-3, rtol=1)

    def test_relative_encoding_effect(self):
        torch.manual_seed(0)
        tokens = torch.rand(token_shape)
        random_transform = torch.rand(operator_shape) * torch.rand(2, 2) * 10
        attention_layer = construct_relative_attention(dim=64 * 3, heads=3)
        absolute_attention_layer = construct_absolute_attention(dim=64 * 3, heads=3)
        duplicate_parameter_values(absolute_attention_layer, attention_layer)
        reference_results = absolute_attention_layer(x=tokens)
        result = attention_layer(x=tokens, positional_transforms=random_transform)
        self.assertFalse(torch.allclose(result, reference_results, atol=1e-3, rtol=0.1))


if __name__ == "__main__":
    unittest.main()
