"""Part H (extra credit): how does action noise affect the agent?

For each p in [0, 0.10, 0.25]: train the baseline config under action
noise p, then greedy-evaluate it under the same noise (50 episodes).
Then cross-evaluate: run the p=0 policy under p=0.10/0.25 noise with no
retraining, to see if the original policy holds up.

Saves results/noise_robustness.json, results/qtable_noise_*.npy and
results/plots/noise_robustness.png
"""
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qlearning import (QLearningAgent, train_agent, evaluate_policy,
                       make_noisy_env, BASELINE, EPISODES, SEED)

NOISE_LEVELS = [0.0, 0.10, 0.25]
EVAL_EPISODES = 50
EVAL_NOISE_SEED = SEED + 999  # same noise stream for every eval run

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PLOTS = os.path.join(RESULTS, "plots")
os.makedirs(PLOTS, exist_ok=True)


def train_under_noise(p, seed=SEED):
    env = make_noisy_env(p, seed=seed + 1000)
    agent = QLearningAgent(seed=seed, **BASELINE)
    rewards, steps = train_agent(agent, env, EPISODES, seed=seed)
    env.close()
    return agent, rewards, steps


def greedy_eval_under_noise(agent, p, episodes=EVAL_EPISODES):
    env = make_noisy_env(p, seed=EVAL_NOISE_SEED)
    metrics = evaluate_policy(agent, env, episodes=episodes)
    env.close()
    return metrics


def plot_noise_robustness(per_noise, cross_eval, path):
    # mean reward and success rate vs noise level p; two series: agents
    # trained under each noise level, and the p=0 policy tested under noise
    ps = sorted(per_noise)
    train_rewards = [per_noise[p]["eval"]["mean_reward"] for p in ps]
    train_reward_std = [per_noise[p]["eval"]["std_reward"] for p in ps]
    train_succ = [per_noise[p]["eval"]["success_rate"] for p in ps]

    xps = sorted(cross_eval)
    cross_rewards = [cross_eval[p]["mean_reward"] for p in xps]
    cross_reward_std = [cross_eval[p]["std_reward"] for p in xps]
    cross_succ = [cross_eval[p]["success_rate"] for p in xps]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.errorbar(ps, train_rewards, yerr=train_reward_std, fmt="-o",
                linewidth=1.8, capsize=4, label="trained @ noise p")
    ax.errorbar(xps, cross_rewards, yerr=cross_reward_std, fmt="--s",
                linewidth=1.8, capsize=4, label="p=0 policy, tested @ p")
    ax.set_xlabel("Action-noise probability p")
    ax.set_ylabel("Mean reward (greedy, 50 eps)")
    ax.set_title("Mean reward vs. action noise")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ps, train_succ, "-o", linewidth=1.8, label="trained @ noise p")
    ax.plot(xps, cross_succ, "--s", linewidth=1.8,
            label="p=0 policy, tested @ p")
    ax.set_xlabel("Action-noise probability p")
    ax.set_ylabel("Success rate (greedy, 50 eps)")
    ax.set_title("Success rate vs. action noise")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    t0 = time.time()
    out = {
        "config": {
            "baseline": BASELINE,
            "episodes": EPISODES,
            "seed": SEED,
            "noise_levels": NOISE_LEVELS,
            "eval_episodes": EVAL_EPISODES,
            "eval_noise_seed": EVAL_NOISE_SEED,
            "note": ("with prob. p the intended action is replaced by a "
                     "uniformly random *different* valid action"),
        },
        "per_noise": {},
        "cross_eval": {},
    }

    trained = {}
    for i, p in enumerate(NOISE_LEVELS):
        print(f"== Training baseline config under noise p={p:.2f} ==",
              flush=True)
        agent, rewards, steps = train_under_noise(p, seed=SEED + i)
        tag = f"p{int(round(p * 100)):03d}"
        agent.save(os.path.join(RESULTS, f"qtable_noise_{tag}.npy"))
        trained[p] = agent

        metrics = greedy_eval_under_noise(agent, p)
        out["per_noise"][str(p)] = {
            "train_mean_reward_last500": round(float(rewards[-500:].mean()), 4),
            "train_mean_steps_last500": round(float(steps[-500:].mean()), 2),
            "eval": metrics,
        }
        print(f"  p={p:.2f}: train(last500) reward={rewards[-500:].mean():.3f} | "
              f"eval mean_reward={metrics['mean_reward']:.3f} "
              f"success={metrics['success_rate']:.2f} "
              f"steps={metrics['mean_steps']:.1f}", flush=True)

    print("== Cross-evaluation: p=0 policy under noise (no retraining) ==",
          flush=True)
    base_agent = trained[0.0]
    for p in [0.10, 0.25]:
        metrics = greedy_eval_under_noise(base_agent, p)
        out["cross_eval"][str(p)] = metrics
        print(f"  p=0 policy @ p={p:.2f}: mean_reward={metrics['mean_reward']:.3f} "
              f"success={metrics['success_rate']:.2f} "
              f"steps={metrics['mean_steps']:.1f}", flush=True)

    json_path = os.path.join(RESULTS, "noise_robustness.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    plot_path = os.path.join(PLOTS, "noise_robustness.png")
    per = {float(k): v for k, v in out["per_noise"].items()}
    # include the p=0 point on the cross-eval series (identical policy)
    cross = {0.0: out["per_noise"]["0.0"]["eval"]}
    cross.update({float(k): v for k, v in out["cross_eval"].items()})
    plot_noise_robustness(per, cross, plot_path)

    dt = time.time() - t0
    print(f"Saved {json_path} and {plot_path} in {dt:.1f}s")


if __name__ == "__main__":
    main()
