"""Training loop for the tabular Q-learning agent."""
import numpy as np


def train_agent(agent, env, episodes, max_steps=100, seed=0):
    """Run epsilon-greedy training.

    Epsilon decays once per episode inside the agent. Returns
    (episode_rewards, episode_steps): per-episode total reward and length.
    """
    rng = np.random.default_rng(seed)
    rewards = np.zeros(episodes, dtype=np.float64)
    steps = np.zeros(episodes, dtype=np.int32)
    for ep in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        done, total, n = False, 0.0, 0
        while not done and n < max_steps:
            action = agent.act(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total += reward
            n += 1
        agent.decay_epsilon()
        rewards[ep] = total
        steps[ep] = n
    return rewards, steps
