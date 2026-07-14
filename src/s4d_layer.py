"""
s4d_layer.py

S4D: a linear time-invariant (LTI), diagonal, complex-valued state-space
layer. Each of the d_model channels gets its own independent complex
diagonal SSM of size state_dim.

Because A, B, C, dt do NOT depend on the input (the system is LTI), the
whole-sequence output can be written as a single convolution of the input
with a fixed kernel. forward() builds that kernel and applies it via one
FFT / inverse-FFT pair -- no Python loop over time steps.
forward_recurrent() computes the exact same operator the slow way (one
step at a time) and exists purely so tests/test_ssm_layers.py can check
the two forms agree numerically; it is never used during training.

Reference: Gu, Gupta, Berant (2022) "On the Parameterization and
Initialization of Diagonal State Space Models" (S4D)
"""

import torch
import torch.nn as nn

from hippo_init import hippo_s4d_lin_init


class S4DLayer(nn.Module):
    def __init__(self, d_model: int, state_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # start every channel from the same HiPPO spectrum, then let each
        # channel's copy drift independently once training starts
        real_part, imag_part = hippo_s4d_lin_init(state_dim)
        real_part = real_part.unsqueeze(0).repeat(d_model, 1)   # (d_model, N)
        imag_part = imag_part.unsqueeze(0).repeat(d_model, 1)   # (d_model, N)

        # log-parameterize the real part so that A = -exp(log_neg_real_A)
        # is guaranteed strictly negative (stable) after every gradient
        # step, no matter how training pushes this parameter around
        self.log_neg_real_A = nn.Parameter(torch.log(-real_part))
        self.A_imag = nn.Parameter(imag_part)

        # B, C: complex, learnable, independent per (channel, state)
        self.B_real = nn.Parameter(torch.randn(d_model, state_dim) / state_dim ** 0.5)
        self.B_imag = nn.Parameter(torch.randn(d_model, state_dim) / state_dim ** 0.5)
        self.C_real = nn.Parameter(torch.randn(d_model, state_dim) / state_dim ** 0.5)
        self.C_imag = nn.Parameter(torch.randn(d_model, state_dim) / state_dim ** 0.5)

        # plain per-channel skip connection
        self.D = nn.Parameter(torch.randn(d_model))

        # learnable per-channel timescale, kept strictly positive via exp()
        self.log_dt = nn.Parameter(torch.log(0.1 * torch.ones(d_model)))

    def _discretize(self):
        """Exact Zero-Order-Hold discretization of the diagonal system."""
        A = -torch.exp(self.log_neg_real_A) + 1j * self.A_imag   # (d_model, N)
        B = self.B_real + 1j * self.B_imag                        # (d_model, N)
        C = self.C_real + 1j * self.C_imag                        # (d_model, N)
        dt = torch.exp(self.log_dt)                                # (d_model,)

        dtA = dt.unsqueeze(-1) * A                                  # (d_model, N)
        A_bar = torch.exp(dtA)
        # ZOH for B: (A_bar - 1) / A -- safe here since Re(A) is always
        # strictly negative, so A is never exactly zero
        B_bar = (A_bar - 1) / A * B
        return A_bar, B_bar, C, dt

    def forward(self, u):
        """u: (batch, seq_len, d_model) -> y: (batch, seq_len, d_model)"""
        batch, seq_len, d_model = u.shape
        A_bar, B_bar, C, dt = self._discretize()

        # powers of A_bar for l = 0 .. seq_len-1: (d_model, N, seq_len)
        exponents = torch.arange(seq_len, dtype=torch.float32, device=u.device)
        A_bar_pows = A_bar.unsqueeze(-1) ** exponents

        # convolution kernel K[d, l] = 2 * Re( C_d . (A_bar_d^l * B_bar_d) )
        # the factor of 2 accounts for the implicit conjugate pole that is
        # never explicitly stored (standard S4D-Lin trick, since only
        # n = 0..N-1 of the HiPPO spectrum are kept, not their conjugates)
        CB = (C * B_bar).unsqueeze(-1)                              # (d_model, N, 1)
        K = 2 * (CB * A_bar_pows).sum(dim=1).real                   # (d_model, L)

        # zero-pad to 2L before FFT so the multiplication in frequency
        # domain is a genuine linear convolution, not a circular one
        u_t = u.transpose(1, 2)                                     # (batch, d_model, L)
        n_fft = 2 * seq_len
        U_f = torch.fft.rfft(u_t, n=n_fft, dim=-1)
        K_f = torch.fft.rfft(K, n=n_fft, dim=-1)
        y_f = U_f * K_f.unsqueeze(0)
        y = torch.fft.irfft(y_f, n=n_fft, dim=-1)[..., :seq_len]    # (batch, d_model, L)

        y = y.transpose(1, 2) + u * self.D.view(1, 1, -1)           # skip connection
        return y

    def forward_recurrent(self, u):
        """Same operator as forward(), computed by literally stepping the
        discretized recurrence h_t = A_bar * h_{t-1} + B_bar * u_t one
        timestep at a time. O(L) sequential Python steps -- only used to
        numerically verify forward() is correct (see tests/), never during
        actual training.
        """
        batch, seq_len, d_model = u.shape
        A_bar, B_bar, C, dt = self._discretize()
        N = self.state_dim

        h = torch.zeros(batch, d_model, N, dtype=torch.cfloat, device=u.device)
        outputs = []
        for t in range(seq_len):
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * u[:, t].unsqueeze(-1)
            y_t = 2 * (C.unsqueeze(0) * h).sum(dim=-1).real
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y + u * self.D.view(1, 1, -1)
        return y
