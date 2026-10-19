"""Greedy (exploration-off) evaluation of a trained policy."""
import numpy as np


def evaluate_policy(agent, env, episodes=50, max_steps=100, seed=999):
    """Evaluate the greedy policy and return a metrics dict."""
    rng = np.random.default_rng(seed)
    rewards, steps = [], []
    for _ in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        done, total, n = False, 0.0, 0
        while not done and n < max_steps:
            action = agent.act(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            n += 1
        rewards.append(total)
        steps.append(n)
    rewards = np.asarray(rewards, dtype=float)
    steps = np.asarray(steps, dtype=float)
    return {
        "episodes": episodes,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "success_rate": float((rewards > 0).mean()),
        "mean_steps": float(steps.mean()),
    }
