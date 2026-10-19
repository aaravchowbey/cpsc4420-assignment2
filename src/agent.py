"""Tabular Q-learning agent implemented from scratch.

No external RL libraries are used: the Q-table, epsilon-greedy policy, and
the Q-learning update are all written out explicitly below.
"""
import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha            # learning rate
        self.gamma = gamma            # discount factor
        self.epsilon = epsilon        # current exploration rate
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        # Q-values start at zero (neutral init). Early behavior is driven
        # purely by epsilon-greedy exploration.
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def act(self, state, explore=True):
        """Epsilon-greedy action selection."""
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        """One step of the standard Q-learning update.

        Terminal states have no future value, so the bootstrap target is
        just the immediate reward when `done` is True.
        """
        best_next = 0.0 if done else float(np.max(self.Q[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self):
        """Geometric epsilon decay, floored at epsilon_min. Call per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
