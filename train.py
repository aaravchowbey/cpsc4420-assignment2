"""Part C: train the baseline agent.

Saves results/qtable_baseline.npy, results/rewards_baseline.npy and
results/plots/baseline_reward_vs_episode.png
"""
import os
import numpy as np
from qlearning import (QLearningAgent, train_agent, make_env, plot_curves,
                       BASELINE, EPISODES, SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(os.path.join(RESULTS, "plots"), exist_ok=True)

print("Baseline config:", BASELINE)
env = make_env()
agent = QLearningAgent(seed=SEED, **BASELINE)
rewards, steps = train_agent(agent, env, EPISODES, seed=SEED)
env.close()

agent.save(os.path.join(RESULTS, "qtable_baseline.npy"))
np.save(os.path.join(RESULTS, "rewards_baseline.npy"), rewards)
plot_curves(
    [(f"baseline {BASELINE}", rewards)],
    os.path.join(RESULTS, "plots", "baseline_reward_vs_episode.png"),
    "Baseline: reward vs episode (FrozenLake-v1, deterministic)",
)
print(f"Done. Mean reward over last 100 episodes: {rewards[-100:].mean():.3f}")
