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
# (C) Train the baseline agent (~a few minutes)
./venv/bin/python train.py

# (D, E, F, G) Hyperparameter sweeps + best-config greedy evaluation
./venv/bin/python experiments.py

# (H, extra credit) Robustness to action noise (p = 0, 0.10, 0.25)
./venv/bin/python experiments_parth.py

# Render a short MP4 of the trained greedy policy
./venv/bin/python make_video.py [path/to/qtable.npy]
```

Baseline hyperparameters (alpha/gamma/epsilon, episode count, seed) live
at the top of `qlearning.py`; sweep values are at the top of
`experiments.py`.

## Layout

```
cpsc4420_hw2/
├── qlearning.py       # agent, training loop, greedy eval, env factory,
│                      #   action-noise wrapper (H), plot helpers
├── train.py           # part (C): baseline training + plot
├── experiments.py     # parts (D)-(G): sweeps, plots, best-config eval
├── experiments_parth.py  # part (H, extra credit): noise robustness
├── make_video.py      # MP4 of the trained agent
├── results/
│   ├── plots/         # PNG learning curves
│   ├── videos/        # MP4 of trained agent
│   ├── summary.json   # sweep results + evaluation metrics
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
