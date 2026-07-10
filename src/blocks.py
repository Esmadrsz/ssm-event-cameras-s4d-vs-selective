"""
hippo_init.py

HiPPO / S4D initialization for diagonal state-space models.
"""

import torch


def hippo_s4d_lin_init(state_dim: int):
    """
    S4D-Lin initialization: a diagonal approximation of the HiPPO-LegS matrix.

    The original HiPPO matrix (used in S4) is upper-triangular plus diagonal
    (see Gu et al. 2022, "How to Train Your HiPPO"), which makes it
    expensive and awkward to implement directly. The S4D paper showed that
    simply keeping the DIAGONAL part of the HiPPO eigen-decomposition
    already recovers almost all of the modeling power, while being far
    simpler and much faster (no custom Cauchy-kernel algebra needed).

    The eigenvalues used here are:
        lambda_n = -1/2 + i * pi * n   for n = 0, 1, ..., state_dim - 1

    Why these specific numbers?
      - The real part (-1/2) is strictly negative for every state, which
        guarantees the continuous-time system x'(t) = A x(t) is stable
        (the state decays exponentially instead of exploding). Without a
        stable A, a randomly-initialized model over a long sequence will
        either vanish or blow up, exactly the RNN gradient problem this
        initialization is designed to avoid.
      - The imaginary part (pi * n) gives EACH state dimension a distinct
        natural oscillation frequency. This is what allows the state
        vector to act like a bank of damped oscillators / a Fourier-like
        basis: low-index dimensions capture slow, long-range trends while
        high-index dimensions capture fast, local details. This is the
        mechanism that lets S4-style models remember information over
        very long sequences without a special "memory" module.

    Returns
    -------
    real_part : (state_dim,) tensor, all equal to -0.5
    imag_part : (state_dim,) tensor, equal to pi * n for each index n
    """
    n = torch.arange(state_dim, dtype=torch.float32)
    real_part = -0.5 * torch.ones(state_dim)
    imag_part = torch.pi * n
    return real_part, imag_part
