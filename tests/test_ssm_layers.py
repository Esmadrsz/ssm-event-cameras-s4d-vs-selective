"""
Unit tests for S4DLayer and SelectiveSSMLayer.

test_fft_convolution_matches_brute_force_recurrence is the single most
important test in this repo: it numerically proves the FFT-based forward()
computes the exact same thing as a brute-force sequential recurrence,
which is the entire mathematical claim underlying the S4/S4D architecture.

Run with: pytest tests/
"""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from s4d_layer import S4DLayer          # noqa: E402
from selective_ssm import SelectiveSSMLayer  # noqa: E402
from blocks import S4Block, DeepSSM      # noqa: E402


@pytest.fixture
def s4d_layer():
    torch.manual_seed(0)
    return S4DLayer(d_model=4, state_dim=8)


def test_fft_convolution_matches_brute_force_recurrence(s4d_layer):
    """The core correctness property of S4D: the parallel FFT-convolution
    forward pass must agree with a brute-force sequential recurrence,
    since they compute the exact same linear operator two different ways."""
    torch.manual_seed(1)
    u = torch.randn(2, 30, 4)

    y_fft = s4d_layer(u)
    y_recurrent = s4d_layer.forward_recurrent(u)

    max_diff = (y_fft - y_recurrent).abs().max().item()
    assert max_diff < 1e-4, f"FFT and recurrent forms disagree by {max_diff}"


def test_s4d_A_matrix_is_stable(s4d_layer):
    """The real part of A must always be negative by construction."""
    A = -torch.exp(s4d_layer.log_neg_real_A)
    assert (A < 0).all()


def test_s4d_output_shape(s4d_layer):
    u = torch.randn(3, 20, 4)
    y = s4d_layer(u)
    assert y.shape == (3, 20, 4)


def test_selective_ssm_output_shape():
    torch.manual_seed(0)
    layer = SelectiveSSMLayer(d_model=4, state_dim=8)
    u = torch.randn(2, 15, 4)
    y = layer(u)
    assert y.shape == (2, 15, 4)


def test_selective_ssm_A_is_stable():
    torch.manual_seed(0)
    layer = SelectiveSSMLayer(d_model=4, state_dim=8)
    A = -torch.exp(layer.log_neg_A)
    assert (A < 0).all()


def test_s4_block_residual_shape():
    torch.manual_seed(0)
    block = S4Block(d_model=8, state_dim=16, layer_type="s4d")
    x = torch.randn(2, 25, 8)
    y = block(x)
    assert y.shape == x.shape


def test_deep_ssm_end_to_end_shapes():
    torch.manual_seed(0)
    model = DeepSSM(input_dim=1, output_dim=1, d_model=16, state_dim=32, n_layers=2)
    x = torch.randn(2, 40, 1)
    y = model(x)
    assert y.shape == (2, 40, 1)


def test_gradients_flow_through_s4d(s4d_layer):
    u = torch.randn(2, 20, 4, requires_grad=False)
    target = torch.randn(2, 20, 4)
    y = s4d_layer(u)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    for name, param in s4d_layer.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
