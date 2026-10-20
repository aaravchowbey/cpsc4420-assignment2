# CPSC 4420 — HW2: Tabular Q-Learning (Fall 2026)

Tabular Q-learning from scratch (no RL libraries) on `FrozenLake-v1`
(8x8 map, deterministic variant, `is_slippery=False`) using OpenAI Gymnasium.

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

FrozenLake-v1 on the **8x8 map** (`map_name="8x8"`), as required by the
assignment: 64 states × 4 actions = a 256-entry Q-table — still small enough
for *tabular* methods with no function approximation and no discretization.
Sparse reward (1.0 only at the goal) and terminal holes make credit
assignment non-trivial. Deterministic transitions (`is_slippery=False` — the
handout does not specify slipperiness, so it is off) keep experiments
reproducible; controlled stochasticity is added separately in part H via an
action-noise wrapper instead of the built-in slippery variant.

## Hyperparameters (baseline, part C)

- alpha = 0.1, gamma = 0.99, epsilon 1.0 → 0.05 (decay 0.9995/episode)
- 10000 episodes, seed 42
- **Optimistic Q initialization** (Q starts at 1.0). On the 8x8 grid a
  random walk essentially never stumbles onto the goal — zero-init
  epsilon-greedy reached it exactly once in 20000 episodes and never
  learned. With optimistic init the greedy policy itself explores
  (untried actions look maximally good), and the agent reliably learns:
  baseline last-100-episode reward ≈ 0.89, greedy evaluation 100%
  success in 14.0 steps (the optimal path length on this map).
