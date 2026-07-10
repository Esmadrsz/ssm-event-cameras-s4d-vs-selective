"""
train.py

Trains and compares two SSM variants on a denoising task (recover a clean
signal from a noisy simulated event-camera stream):

  1. S4D    -- linear time-invariant, HiPPO-initialized, FFT-parallel
  2. Selective (Mamba-lite) -- input-dependent A/B/C/dt, sequential scan

Then runs a bandlimiting / frequency-generalization test, directly
inspired by Zubic, Gehrig, Gehrig, Scaramuzza (2024) "State Space Models
for Event Cameras" (CVPR) -- the key property that paper exploits is that
an LTI SSM (like S4D) can be evaluated at a DIFFERENT frequency than it
was trained at simply by rescaling its timescale (dt), because it's a
discretized continuous-time system. This script empirically tests how
well that property holds for both layer types.

References:
  - Gu, Goel, Re (2021) "Efficiently Modeling Long Sequences with
    Structured State Spaces" (S4)
  - Gu, Gupta, Berant (2022) "On the Parameterization and
    Initialization of Diagonal State Space Models" (S4D)
  - Gu, Dao (2023) "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces"
  - Zubic, Gehrig, Gehrig, Scaramuzza (2024) "State Space Models for
    Event Cameras" (CVPR)
  - Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De (2023)
    "Resurrecting Recurrent Neural Networks for Long Sequences" (LRU, ICML Oral)
  - Muca Cirone, Orvieto, Walker, Salvi, Lyons (2024) "Theoretical
    Foundations of Deep Selective State-Space Models" (NeurIPS)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blocks import DeepSSM
from event_data import generate_event_stream


def train_model(model, seq_len=200, n_epochs=300, lr=3e-3, grad_clip=1.0, verbose=True):
    """
    A fresh random noise realization is used every epoch (via the
    `seed=epoch` trick), which acts like an infinite streaming dataset
    and helps prevent the model from memorizing one fixed noise pattern.
    Gradient clipping guards against exploding gradients, which can
    happen in recurrent-style models when a state matrix's eigenvalues
    drift too close to (or past) the unit circle during training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(n_epochs):
        noisy, clean = generate_event_stream(seq_len=seq_len, seed=epoch)
        x = torch.tensor(noisy).view(1, seq_len, 1)
        target = torch.tensor(clean).view(1, seq_len, 1)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(loss.item())
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:4d}/{n_epochs}  loss = {loss.item():.5f}")

    return losses


def bandlimiting_test(model, test_freqs=(0.5, 1.0, 2.0, 4.0), seq_len=200):
    """
    Evaluates a model (trained at freq=1.0) on several different signal
    frequencies -- probing whether the model generalizes gracefully
    across "sampling rates" it wasn't trained on, which matters a lot for
    real event cameras (whose effective event rate is not fixed).
    """
    loss_fn = nn.MSELoss()
    results = {}
    with torch.no_grad():
        for f in test_freqs:
            noisy, clean = generate_event_stream(seq_len=seq_len, freq=f, seed=999)
            x = torch.tensor(noisy).view(1, seq_len, 1)
            target = torch.tensor(clean).view(1, seq_len, 1)
            pred = model(x)
            results[f] = loss_fn(pred, target).item()
    return results


def main():
    torch.manual_seed(0)
    SEQ_LEN = 200

    print("=" * 60)
    print("Model 1/2: S4D (HiPPO init + learnable dt + FFT convolution)")
    print("=" * 60)
    s4d_model = DeepSSM(input_dim=1, output_dim=1, d_model=32, state_dim=64,
                         n_layers=3, layer_type="s4d")
    s4d_losses = train_model(s4d_model, seq_len=SEQ_LEN, n_epochs=300)

    print()
    print("=" * 60)
    print("Model 2/2: Mamba-lite (input-dependent, selective SSM)")
    print("=" * 60)
    sel_model = DeepSSM(input_dim=1, output_dim=1, d_model=32, state_dim=16,
                         n_layers=2, layer_type="selective")
    sel_losses = train_model(sel_model, seq_len=SEQ_LEN, n_epochs=300)

    # --- Bandlimiting / frequency-generalization test ---
    print()
    print("=" * 60)
    print("Bandlimiting test: trained at freq=1.0, evaluated at other frequencies")
    print("(inspired by Zubic et al. 2024, 'State Space Models for Event Cameras')")
    print("=" * 60)
    s4d_bandtest = bandlimiting_test(s4d_model)
    sel_bandtest = bandlimiting_test(sel_model)
    for f in s4d_bandtest:
        print(f"  freq={f:>4}  S4D MSE={s4d_bandtest[f]:.5f}   Selective MSE={sel_bandtest[f]:.5f}")

    # --- Visualization ---
    noisy, clean = generate_event_stream(seq_len=SEQ_LEN, seed=123)
    x_test = torch.tensor(noisy).view(1, SEQ_LEN, 1)
    with torch.no_grad():
        s4d_pred = s4d_model(x_test).squeeze().numpy()
        sel_pred = sel_model(x_test).squeeze().numpy()

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    axes[0].plot(noisy, color="gray", alpha=0.6, label="Noisy event stream (input)")
    axes[0].plot(clean, color="black", linestyle="--", label="Clean signal (target)")
    axes[0].set_title("Simulated Event-Camera Data")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(clean, color="black", linestyle="--", label="Target")
    axes[1].plot(s4d_pred, color="crimson", label="S4D output")
    axes[1].set_title("S4D Model: Denoising Result")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(clean, color="black", linestyle="--", label="Target")
    axes[2].plot(sel_pred, color="teal", label="Selective (Mamba-lite) output")
    axes[2].set_title("Selective SSM Model: Denoising Result")
    axes[2].legend(); axes[2].grid(True)

    axes[3].plot(s4d_losses, color="crimson", label="S4D training loss")
    axes[3].plot(sel_losses, color="teal", label="Selective training loss")
    axes[3].set_yscale("log")
    axes[3].set_title("Training Loss (MSE, log scale)")
    axes[3].set_xlabel("Epoch")
    axes[3].legend(); axes[3].grid(True)

    plt.tight_layout()
    plt.savefig("output/advanced_ssm_output.png", dpi=130)
    print("\nPlot saved as: output/advanced_ssm_output.png")


if __name__ == "__main__":
    main()
