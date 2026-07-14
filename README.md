# S4D vs. Selective State-Space Models for Event-Camera-Style Denoising

A from-scratch PyTorch implementation and comparison of two State-Space
Model (SSM) variants — a linear time-invariant **S4D** layer (HiPPO
initialization, FFT-parallel) and a **Mamba-lite selective SSM** layer
(input-dependent A/B/C/dt) — applied to denoising a simulated event-camera
signal, with a specific focus on **frequency generalization**: how well
each model handles a different "sampling rate" than the one it was
trained on.

## Motivation

Event cameras report asynchronous per-pixel brightness-change events
rather than fixed-rate frames, so the effective "sampling rate" of the
data isn't fixed the way it is for a conventional video sensor. Zubic,
Gehrig, Gehrig & Scaramuzza (2024, CVPR) — *["State Space Models for
Event Cameras"](https://arxiv.org/abs/2402.15584)* — show that this is
exactly where linear SSMs have a structural advantage: because they're
discretized continuous-time systems, they can in principle be evaluated
at a different frequency than they were trained at, just by rescaling
the timescale parameter (`dt`), without retraining. This project builds
both an LTI (S4D) and a non-LTI (selective) SSM specifically to test that
claim empirically, on a small denoising task.

This project is built with a specific goal: preparing for a research
conversation on SSMs and event cameras with **Antonio Orvieto**
(ELLIS Institute Tübingen / MPI-IS), whose own work spans exactly the
theoretical ground this project touches:
- *"Resurrecting Recurrent Neural Networks for Long Sequences"* (LRU,
  ICML 2023 Oral) — the stable log-space diagonal parameterization used
  in `S4DLayer` and `SelectiveSSMLayer` follows this line of work.
- *"Theoretical Foundations of Deep Selective State-Space Models"*
  (NeurIPS 2024) — the theory behind exactly the kind of input-dependent
  selection mechanism implemented in `SelectiveSSMLayer`.

## What's Implemented

**1. `S4DLayer`** — a linear time-invariant diagonal SSM:
- HiPPO-inspired initialization (`hippo_init.py`): complex eigenvalues
  `λ_n = -1/2 + iπn`, giving each state dimension a distinct decay rate
  and oscillation frequency (Gu, Gupta, Berant 2022).
- Exact Zero-Order-Hold (ZOH) discretization, with a learnable per-channel
  timescale (`dt`).
- Computed via **FFT convolution** over the whole sequence — no
  sequential Python loop.

**2. `SelectiveSSMLayer`** — a simplified Mamba-style selective SSM:
- `A` is fixed (stable, diagonal); `B`, `C`, and `dt` are recomputed from
  the input at *every* time step via small linear projections.
- Necessarily computed via a sequential scan (real Mamba uses a
  hardware-aware parallel-scan kernel; this is a readable, pedagogical
  stand-in for that).

**3. `DeepSSM`** — encoder → `[LayerNorm → SSM → GLU → residual] × N` →
decoder, the standard deep S4-style architecture, usable with either
layer type.

## Results

All numbers below are real output from `train.py` / averaged over 5
random seeds, not illustrative.

**Correctness (verified in `tests/`):** `S4DLayer`'s FFT-convolution
forward pass was checked against a brute-force sequential recurrence
computing the exact same system. Max absolute difference: **3.6 × 10⁻⁷**
— floating-point precision, i.e. the two computation paths are
mathematically identical, as they must be for an LTI system.

**Bandlimiting / frequency-generalization test** (trained at `freq=1.0`,
evaluated at other frequencies — directly testing the Zubic et al. claim.
Averaged over 5 seeds; single-seed runs vary by roughly ±30% but the
ranking below was stable across all 5):

| Frequency | S4D MSE (avg) | Selective MSE (avg) |
|---:|---:|---:|
| 0.5× | **0.0098** | 0.0126 |
| 1.0× (training freq) | **0.0025** | 0.0035 |
| 2.0× | **0.0202** | 0.0356 |
| 4.0× | 0.1940 | **0.1477** |

**Finding:** across 5 seeds, S4D is consistently *better* than the
selective model at moderate frequency shifts (0.5× and 2×) as well as at
the training frequency — closer to what the "LTI systems transfer across
frequencies via `dt` rescaling" argument would predict. The two models
only swap places at the most extreme shift tested (4×), where the
selective model's error grows more slowly than S4D's. So the honest
summary is narrower than "selective always generalizes better": S4D's
advantage holds over a 4x range around the training frequency, and only
breaks down at the far edge of what was tested here. This is a real,
measured result from this specific toy task with a specific set of
hyperparameters (`state_dim`, `d_model`, `n_layers` differ between the
two models here) — not a general claim about S4D vs. Mamba — and it's
exactly the kind of result, and the kind of "well, it depends on the
shift magnitude" nuance, I'd want to discuss and stress-test further in
a research conversation.

## Tech Stack

`Python` · `PyTorch` (core linear algebra, complex-valued tensors, autograd) · `NumPy` · `Matplotlib`

## Repository Structure

```
.
├── src/
│   ├── hippo_init.py       # HiPPO/S4D eigenvalue initialization
│   ├── s4d_layer.py         # S4DLayer (LTI, FFT-parallel)
│   ├── selective_ssm.py      # SelectiveSSMLayer (Mamba-lite)
│   ├── blocks.py              # S4Block, DeepSSM (deep architecture)
│   └── event_data.py           # simulated noisy event-camera signal
├── tests/
│   └── test_ssm_layers.py       # FFT/recurrent equivalence + sanity checks
├── train.py                       # training + bandlimiting test + plots
├── output/
│   └── advanced_ssm_output.png
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

python3 train.py       # trains both models, runs the bandlimiting test,
                          # saves output/advanced_ssm_output.png
pytest tests/ -v          # equivalence + sanity tests
```

## Future Work

- Test on a genuinely long sequence with a long-range recall task, where
  the selective mechanism's advantages are expected to be most pronounced.
- Replace the sequential scan in `SelectiveSSMLayer` with a proper
  parallel-scan implementation.
- Move from the synthetic sine-wave signal to a real event-camera dataset
  (DVS-Gesture, N-MNIST) or, ideally, real event data relevant to the
  aerospace domain.
- Investigate *why* S4D's advantage flips at the most extreme frequency
  shift (4×) tested here — does explicitly rescaling `dt` at test time
  (as in the LTI-frequency-transfer argument) push that crossover point
  further out, or is there a structural reason the selective model
  should eventually win at large enough shifts?
- Run more seeds and a finer grid of test frequencies to pin down where
  exactly the crossover between the two models happens.

## References

- Gu, Goel, Ré (2021). *Efficiently Modeling Long Sequences with
  Structured State Spaces* (S4). ICLR 2022.
- Gu, Gupta, Berant (2022). *On the Parameterization and Initialization
  of Diagonal State Space Models* (S4D). NeurIPS 2022.
- Gu, Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces*.
- Zubic, Gehrig, Gehrig, Scaramuzza (2024). *State Space Models for Event
  Cameras*. CVPR 2024.
- Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De (2023).
  *Resurrecting Recurrent Neural Networks for Long Sequences* (LRU).
  ICML 2023 (Oral).
- Muca Cirone, Orvieto, Walker, Salvi, Lyons (2024). *Theoretical
  Foundations of Deep Selective State-Space Models*. NeurIPS 2024.

