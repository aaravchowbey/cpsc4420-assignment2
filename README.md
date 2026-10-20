# CPSC 4420 — HW2: Tabular Q-Learning (Fall 2026)

Tabular Q-learning from scratch (no RL libraries) on `FrozenLake-v1`
(deterministic variant, `is_slippery=False`) using OpenAI Gymnasium.

## Setup

```bash
cd cpsc4420_hw2
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install gymnasium numpy matplotlib imageio imageio-ffmpeg
```

`ffmpeg` is also used for the MP4 video (`imageio-ffmpeg` ships its own
binary, so a system ffmpeg is optional).

## Run

```bash
# Runs everything: baseline training (C), sweeps (D-F), best-agent
# eval (G), noise-robustness study (H), and the MP4 video (~a few minutes)
./venv/bin/python assignment2.py
```

All hyperparameters (alpha/gamma/epsilon, episode count, seed) and sweep
values are plain constants at the top of `assignment2.py`.

## Layout

```
cpsc4420_hw2/
├── assignment2.py   # everything: Q-learning agent, training loop, env,
│                    #   action-noise wrapper (H), all experiments (C-H),
│                    #   plots, and the MP4 video render
├── results/
│   ├── plots/         # PNG learning curves
│   ├── videos/        # MP4 of trained agent
│   ├── summary.json   # sweep results + evaluation metrics
│   ├── noise_robustness.json  # part H results
│   └── qtable_*.npy   # saved Q-tables
├── README.md
└── REPORT_OUTLINE.md  # bullet outline for writing the PDF report
```

## Environment choice

FrozenLake-v1 (`is_slippery=False`): 16 states × 4 actions = a 64-entry
Q-table — ideal for *tabular* methods with no function approximation and no
discretization. Sparse reward (1.0 only at the goal) and terminal holes make
credit assignment non-trivial, yet it trains in minutes. Deterministic
transitions keep experiments reproducible; the stochastic (`is_slippery=True`)
variant is the natural extension for the graduate extra-credit study (H).
