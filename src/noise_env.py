"""Part H (extra credit): controlled action-noise wrapper.

To study robustness to stochasticity, we wrap the deterministic
FrozenLake-v1 environment so that, with probability ``p``, the agent's
intended action is replaced by a uniformly random *different* valid
action before being passed to the underlying environment.

This is deliberately *not* the built-in ``is_slippery=True`` variant:
that changes the transition dynamics themselves (sliding on ice),
whereas here the noise is injected at the action-selection boundary.
The base environment stays exactly the parts A-G environment, so the
only thing that varies between runs is the single parameter ``p`` --
a clean, controlled experiment.

Note: nothing in this module changes the behavior of the existing
``make_env()`` path; it is purely additive for Part H.
"""
import gymnasium as gym
import numpy as np

from env_utils import make_env


class ActionNoiseWrapper(gym.Wrapper):
    """Inject action noise: with probability p, swap in a random *different* action.

    Parameters
    ----------
    env : gymnasium.Env
        The underlying (deterministic) environment.
    p : float
        Probability of replacing the intended action. Must be in [0, 1].
    seed : int or None
        Seed for the wrapper's own RNG, independent of the env/agent RNGs
        so noise streams are reproducible and decoupled from training seeds.
    """

    def __init__(self, env, p=0.0, seed=None):
        super().__init__(env)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"noise probability p must be in [0, 1], got {p}")
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        action = int(action)
        if self.rng.random() < self.p:
            others = [a for a in range(self.action_space.n) if a != action]
            action = int(self.rng.choice(others))
        return self.env.step(action)


def make_noisy_env(p=0.0, seed=None, render_mode=None):
    """Build the deterministic FrozenLake env wrapped with action noise.

    With p=0.0 this returns the plain parts A-G environment unchanged.
    """
    base = make_env(render_mode=render_mode)
    if p == 0.0:
        return base
    return ActionNoiseWrapper(base, p=p, seed=seed)
