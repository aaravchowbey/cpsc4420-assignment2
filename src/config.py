"""Shared hyperparameters for all experiments."""

# (C) Baseline configuration
# NOTE: epsilon must decay slowly here. FrozenLake has a sparse reward
# (1.0 only at the goal), so if exploration collapses too early the greedy
# policy gets stuck walking into walls/holes and never sees a reward.
# With decay=0.999, epsilon is still ~0.37 at episode 1000, giving the
# agent time to discover the goal and propagate Q-values.
BASELINE = dict(
    alpha=0.1,          # learning rate
    gamma=0.99,         # discount factor
    epsilon=1.0,        # initial exploration rate
    epsilon_min=0.05,   # exploration floor
    epsilon_decay=0.999 # per-episode geometric decay of epsilon
)

EPISODES = 4000
SEED = 42

# (D) Learning-rate sweep values
ALPHAS = [0.01, 0.1, 0.5]
# (E) Discount-factor sweep values
GAMMAS = [0.8, 0.95, 0.999]
# (F) Extra-hyperparameter sweep: exploration decay schedule
EPS_DECAYS = [0.99, 0.995, 0.999]
