"""Environment setup.

We use FrozenLake-v1 from OpenAI Gymnasium with is_slippery=False.

Why this environment:
  * Fully discrete: 16 states (4x4 grid) x 4 actions -> a Q-table of only
    64 entries, ideal for *tabular* Q-learning with no function
    approximation and no state discretization needed.
  * Sparse reward (1.0 only on reaching the goal, 0 otherwise) plus terminal
    holes -> a genuine temporal credit-assignment problem that still trains
    in minutes on a laptop.
  * is_slippery=False makes transitions deterministic, so runs are
    reproducible and hyperparameter effects are easy to interpret. (The
    slippery/stochastic variant is the natural extension for the
    graduate extra-credit study.)
"""
import gymnasium as gym

ENV_ID = "FrozenLake-v1"
N_STATES = 16
N_ACTIONS = 4


def make_env(render_mode=None):
    """Create the FrozenLake environment (deterministic variant)."""
    return gym.make(ENV_ID, is_slippery=False, render_mode=render_mode)


def describe_action(a):
    return {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}.get(a, str(a))
