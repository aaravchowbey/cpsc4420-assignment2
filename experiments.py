"""Hyperparameter sweeps + best-config evaluation (parts D, E, F, G).

Usage:  python experiments.py
Runs:
  (D) alpha sweep        -> results/plots/alpha_comparison.png
  (E) gamma sweep        -> results/plots/gamma_comparison.png
  (F) epsilon-decay sweep-> results/plots/epsilon_decay_comparison.png
  (G) greedy eval of the best (alpha, gamma) combo over 50 episodes
Saves:  results/summary.json, results/qtable_best.npy
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agent import QLearningAgent
from config import BASELINE, EPISODES, SEED, ALPHAS, GAMMAS, EPS_DECAYS
from env_utils import make_env, N_STATES, N_ACTIONS
from training import train_agent
from evaluate import evaluate_policy
from plot_utils import plot_curves

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(RESULTS, "plots")
os.makedirs(PLOTS, exist_ok=True)


def run_config(cfg, episodes=EPISODES, seed=SEED):
    env = make_env()
    agent = QLearningAgent(N_STATES, N_ACTIONS, seed=seed, **cfg)
    rewards, _ = train_agent(agent, env, episodes, seed=seed)
    env.close()
    return agent, rewards


def sweep(values, key, title, plot_name):
    curves, final = [], {}
    for v in values:
        cfg = dict(BASELINE, **{key: v})
        _, rewards = run_config(cfg)
        curves.append((f"{key}={v}", rewards))
        final[str(v)] = round(float(rewards[-500:].mean()), 4)
        print(f"  {key}={v}: mean reward (last 500 eps) = {final[str(v)]:.3f}")
    plot_curves(curves, os.path.join(PLOTS, plot_name), title)
    return final


def main():
    summary = {"baseline": BASELINE}

    print("== (D) Learning-rate sweep ==")
    summary["alpha_sweep"] = sweep(ALPHAS, "alpha",
        "Learning-rate sweep: reward vs episode", "alpha_comparison.png")

    print("== (E) Discount-factor sweep ==")
    summary["gamma_sweep"] = sweep(GAMMAS, "gamma",
        "Discount-factor sweep: reward vs episode", "gamma_comparison.png")

    print("== (F) Exploration-decay sweep ==")
    summary["epsilon_decay_sweep"] = sweep(EPS_DECAYS, "epsilon_decay",
        "Exploration-decay sweep: reward vs episode",
        "epsilon_decay_comparison.png")

    # (G) Best (alpha, gamma) by late-training mean reward; greedy eval.
    best_alpha = max(summary["alpha_sweep"], key=summary["alpha_sweep"].get)
    best_gamma = max(summary["gamma_sweep"], key=summary["gamma_sweep"].get)
    best_cfg = dict(BASELINE, alpha=float(best_alpha), gamma=float(best_gamma))
    print(f"== (G) Evaluating best config {best_cfg} (greedy, 50 eps) ==")
    best_agent, _ = run_config(best_cfg)
    best_agent.save(os.path.join(RESULTS, "qtable_best.npy"))

    env = make_env()
    summary["best_config"] = best_cfg
    summary["best_eval_greedy_50eps"] = evaluate_policy(best_agent, env, episodes=50)
    env.close()
    print("  best eval:", summary["best_eval_greedy_50eps"])

    env = make_env()
    base_agent = QLearningAgent(N_STATES, N_ACTIONS, seed=SEED, **BASELINE)
    # re-train baseline quickly for a fair greedy comparison
    from training import train_agent as _t
    train_agent(base_agent, env, EPISODES, seed=SEED)
    summary["baseline_eval_greedy_50eps"] = evaluate_policy(base_agent, env, episodes=50)
    env.close()
    print("  baseline eval:", summary["baseline_eval_greedy_50eps"])

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {RESULTS}/summary.json and qtable_best.npy")


if __name__ == "__main__":
    main()
