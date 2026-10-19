"""Greedy evaluation of a saved Q-table (part G helper).

Usage:  python evaluate.py [path/to/qtable.npy] [--episodes 50]
Default: results/qtable_best.npy
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agent import QLearningAgent
from env_utils import make_env, N_STATES, N_ACTIONS
from evaluate import evaluate_policy

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qtable", nargs="?",
                    default=os.path.join(ROOT, "results", "qtable_best.npy"))
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    agent = QLearningAgent(N_STATES, N_ACTIONS)
    agent.load(args.qtable)
    env = make_env()
    m = evaluate_policy(agent, env, episodes=args.episodes)
    env.close()
    print(f"Evaluated {args.qtable} greedily over {m['episodes']} episodes:")
    print(f"  mean reward : {m['mean_reward']:.3f} (+/- {m['std_reward']:.3f})")
    print(f"  success rate: {m['success_rate']:.3f}")
    print(f"  mean steps  : {m['mean_steps']:.1f}")


if __name__ == "__main__":
    main()
