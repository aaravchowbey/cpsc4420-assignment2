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

# Greedy evaluation of a saved Q-table (default: results/qtable_best.npy)
./venv/bin/python evaluate.py [path/to/qtable.npy] [--episodes 50]

# Render a short MP4 of the trained greedy policy
./venv/bin/python make_video.py [path/to/qtable.npy]
```

All hyperparameters live in `src/config.py` (baseline alpha/gamma/epsilon,
sweep values, episode counts, seeds).

## Layout

```
cpsc4420_hw2/
├── src/
│   ├── agent.py       # QLearningAgent: Q-table, epsilon-greedy, Q-update
│   ├── config.py      # baseline + sweep hyperparameters
│   ├── env_utils.py   # FrozenLake-v1 factory (+ why this environment)
│   ├── training.py    # epsilon-greedy training loop
│   ├── evaluate.py    # greedy evaluation metrics
│   └── plot_utils.py  # moving-average learning-curve plots
├── train.py           # part (C): baseline training + plot
├── experiments.py     # parts (D)-(G): sweeps, plots, best-config eval
├── evaluate.py        # greedy eval of any saved Q-table
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
