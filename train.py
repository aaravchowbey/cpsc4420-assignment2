"""Train the baseline agent (part C).

Usage:  python train.py [--episodes 3000]
Saves:  results/qtable_baseline.npy, results/rewards_baseline.npy,
        results/plots/baseline_reward_vs_episode.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agent import QLearningAgent
from config import BASELINE, EPISODES, SEED
from env_utils import make_env, N_STATES, N_ACTIONS
from training import train_agent
from plot_utils import plot_curves

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(os.path.join(RESULTS, "plots"), exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print("Baseline config:", BASELINE)
    env = make_env()
    agent = QLearningAgent(N_STATES, N_ACTIONS, seed=args.seed, **BASELINE)
    rewards, steps = train_agent(agent, env, args.episodes, seed=args.seed)
    env.close()

    agent.save(os.path.join(RESULTS, "qtable_baseline.npy"))
    import numpy as np
    np.save(os.path.join(RESULTS, "rewards_baseline.npy"), rewards)
    plot_curves(
        [(f"baseline {BASELINE}", rewards)],
        os.path.join(RESULTS, "plots", "baseline_reward_vs_episode.png"),
        "Baseline: reward vs episode (FrozenLake-v1, deterministic)",
    )
    print(f"Done. Mean reward over last 100 episodes: {rewards[-100:].mean():.3f}")
    print(f"Saved Q-table, rewards, and plot under {RESULTS}/")


if __name__ == "__main__":
    main()
