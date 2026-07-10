"""
selective_ssm.py

A simplified, pedagogical version of Mamba's "selective" SSM (the S6
block). Unlike S4DLayer, A, B, C, and dt are recomputed from the input at
EVERY time step, letting the model dynamically decide, moment by moment,
how much of the current input to let into the state and how quickly to
forget the past.

Reference: Gu, Dao (2023) "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces"
"""

import torch
import torch.nn as nn


class SelectiveSSMLayer(nn.Module):
    """
    IMPORTANT CAVEAT: real Mamba computes this recurrence with a custom
    "hardware-aware parallel scan" CUDA kernel that achieves O(log L)
    parallel time despite the input-dependent parameters. Because the
    system is no longer time-invariant, we can no longer use the FFT/
    convolution trick from S4DLayer. This implementation falls back to a
    plain SEQUENTIAL scan (a Python for-loop over time) -- fine for
    understanding the mechanism and short-to-medium sequences, but
    noticeably slower than S4DLayer on long sequences.
    """

    def __init__(self, d_model: int, state_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # A is still a fixed, per-channel negative diagonal (for
        # stability), unlike Mamba's full parameterization -- but B, C,
        # and dt below ARE selective (input-dependent), which is the core
        # idea we want to demonstrate.
        self.log_neg_A = nn.Parameter(torch.log(0.5 * torch.ones(d_model, state_dim)))

        # Small linear layers that turn the current input into B, C, and
        # dt at every time step -- this is the "selection mechanism".
        self.proj_B = nn.Linear(d_model, state_dim)
        self.proj_C = nn.Linear(d_model, state_dim)
        self.proj_dt = nn.Linear(d_model, d_model)
        self.D = nn.Parameter(torch.randn(d_model))

    def forward(self, u):
        """u: (batch, seq_len, d_model) -> y: (batch, seq_len, d_model)"""
        batch, seq_len, d_model = u.shape
        N = self.state_dim

        A = -torch.exp(self.log_neg_A)                        # (d_model, N)
        # softplus keeps dt strictly positive, as required for a valid
        # ZOH discretization
        dt = torch.nn.functional.softplus(self.proj_dt(u))    # (batch, L, d_model)
        B = self.proj_B(u)                                     # (batch, L, N)
        C = self.proj_C(u)                                     # (batch, L, N)

        # ZOH discretization, but now recomputed at every single time
        # step because dt and B depend on the current input.
        dtA = dt.unsqueeze(-1) * A.view(1, 1, d_model, N)       # (batch,L,d_model,N)
        A_bar = torch.exp(dtA)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)               # (batch,L,d_model,N)

        # Sequential scan: this is the part that real Mamba replaces with
        # a parallel-scan kernel. We keep it as an explicit loop here so
        # the recurrence is easy to read and debug.
        h = torch.zeros(batch, d_model, N, device=u.device)
        outputs = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar[:, t] * u[:, t].unsqueeze(-1)
            y_t = torch.einsum('bdn,bn->bd', h, C[:, t])
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)                         # (batch, L, d_model)
        y = y + u * self.D.view(1, 1, -1)                       # skip connection
        return y
