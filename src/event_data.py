"""
event_data.py

Simulates an event-camera-like stream as a noisy sine wave: a clean
underlying signal (what we'd like to recover) plus Gaussian noise
(mimicking sensor/quantization noise in a real event stream).
"""

import numpy as np


def generate_event_stream(seq_len=200, freq=1.0, noise_std=0.1, seed=None):
    """
    `freq` scales how many oscillation cycles fit inside the fixed
    sequence length -- used in bandlimiting_test() to simulate testing
    the model at a different "sampling rate" than the one it was trained on.
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi * freq, seq_len)
    clean = np.sin(t)
    noisy = clean + noise_std * np.random.randn(seq_len)
    return noisy.astype(np.float32), clean.astype(np.float32)
