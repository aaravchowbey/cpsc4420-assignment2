"""Tabular Q-learning from scratch for CPSC 4420 Assignment 2.

Everything for parts A-G lives here: the agent, training loop, greedy
evaluation, the FrozenLake env factory, the Part H noise wrapper, and
the plotting helpers. The experiment scripts import from this module.
"""
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt

N_STATES = 16   # 4x4 FrozenLake grid
N_ACTIONS = 4   # LEFT, DOWN, RIGHT, UP

# baseline hyperparameters (part C)
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
EPISODES = 4000
SEED = 42

BASELINE = dict(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY)


def make_env(render_mode=None):
    # deterministic FrozenLake (is_slippery=False)
    return gym.make("FrozenLake-v1", is_slippery=False,
                    render_mode=render_mode)


class QLearningAgent:
    """Tabular Q-learning agent."""

    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                 epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY,
                 seed=SEED):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        # zero init: every action looks equal, so early on the agent
        # just explores
        self.Q = np.zeros((N_STATES, N_ACTIONS))

    def act(self, state, explore=True):
        # epsilon-greedy
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        # Q-learning update; no bootstrap past terminal states
        best_next = 0.0 if done else float(np.max(self.Q[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self):
        # geometric decay once per episode, floored at epsilon_min
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)


def train_agent(agent, env, episodes, max_steps=100, seed=SEED):
    """Epsilon-greedy training. Returns (rewards, steps) per episode."""
    rng = np.random.default_rng(seed)
    rewards = np.zeros(episodes)
    steps = np.zeros(episodes, dtype=int)
    for ep in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(2 ** 31 - 1)))
        done = False
        total, n = 0.0, 0
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


def evaluate_policy(agent, env, episodes=50, max_steps=100, seed=999):
    """Greedy evaluation. Returns a dict of metrics."""
    rng = np.random.default_rng(seed)
    rewards, steps = [], []
    for _ in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(2 ** 31 - 1)))
        done = False
        total, n = 0.0, 0
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


class ActionNoiseWrapper(gym.Wrapper):
    """Part H: with prob p, replace the action by a random different one."""

    def __init__(self, env, p=0.0, seed=None):
        super().__init__(env)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        action = int(action)
        if self.rng.random() < self.p:
            others = [a for a in range(self.action_space.n) if a != action]
            action = int(self.rng.choice(others))
        return self.env.step(action)


def make_noisy_env(p=0.0, seed=None, render_mode=None):
    base = make_env(render_mode=render_mode)
    if p == 0.0:
        return base
    return ActionNoiseWrapper(base, p=p, seed=seed)


def moving_average(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_curves(curves, path, title, window=100,
                xlabel="Episode", ylabel="Reward (100-ep moving avg)"):
    """Smoothed learning curves. curves = [(label, rewards), ...]."""
    plt.figure(figsize=(8, 5))
    for label, rewards in curves:
        ma = moving_average(rewards, window)
        xs = np.arange(window - 1, window - 1 + len(ma))
        plt.plot(xs, ma, label=label, linewidth=1.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
