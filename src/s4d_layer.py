"""
blocks.py

Deep architecture built from S4D or Selective SSM layers: encoder ->
[Norm -> SSM -> GLU -> residual] x N -> decoder. Follows the standard
recipe used in the S4 paper's deep models.
"""

import torch.nn as nn

from s4d_layer import S4DLayer
from selective_ssm import SelectiveSSMLayer


class S4Block(nn.Module):
    """
    One S4/S4D-style block: pre-norm -> SSM mixing -> GLU nonlinearity ->
    dropout -> residual connection. Stacking several of these lets the
    network build up hierarchical temporal features, the same way
    stacking Transformer blocks builds up hierarchical attention
    features.
    """

    def __init__(self, d_model, state_dim, layer_type="s4d", dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if layer_type == "s4d":
            self.ssm = S4DLayer(d_model, state_dim)
        elif layer_type == "selective":
            self.ssm = SelectiveSSMLayer(d_model, state_dim)
        else:
            raise ValueError(f"unknown layer_type: {layer_type}")
        # GLU (Gated Linear Unit): projects to 2*d_model then gates half
        # the channels with the other half via a sigmoid -- a learnable,
        # input-dependent nonlinearity, similar in spirit to the
        # feed-forward sublayer in a Transformer block.
        self.glu = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GLU(dim=-1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.glu(x)
        x = self.dropout(x)
        return x + residual  # crucial for stacking multiple blocks without
        # vanishing/exploding signal magnitudes


class DeepSSM(nn.Module):
    """
    A full deep SSM model: a linear encoder projects the raw input up to
    a wider internal representation (d_model channels), several S4Block
    layers process it, and a linear decoder projects back down to the
    desired output dimension.
    """

    def __init__(self, input_dim, output_dim, d_model=32, state_dim=64,
                 n_layers=3, layer_type="s4d"):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [S4Block(d_model, state_dim, layer_type) for _ in range(n_layers)]
        )
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        return self.decoder(x)
