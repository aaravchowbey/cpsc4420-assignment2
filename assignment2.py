"""CPSC 4420 Assignment 2: Tabular Q-learning on FrozenLake-v1 (Fall 2026).

Implements tabular Q-learning from scratch (no RL libraries) on the
deterministic FrozenLake-v1 env (is_slippery=False), then studies how
hyperparameters affect learning:

  Part C  baseline training
  Part D  learning-rate sweep
  Part E  discount-factor sweep
  Part F  exploration-decay sweep
  Part G  greedy evaluation of the best agent
  Part H  (extra credit) robustness to action noise p = 0, 0.10, 0.25
          + a rendered MP4 video of the trained greedy policy

Run:  python assignment2.py   (takes a few minutes)
Outputs go to results/: plots/*.png, *.json, qtable_*.npy, videos/*.mp4
"""
import json
import os
import time

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # headless: just save PNGs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio

# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------
N_STATES = 16   # 4x4 FrozenLake grid
N_ACTIONS = 4   # LEFT, DOWN, RIGHT, UP

# baseline (part C)
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
EPISODES = 4000
SEED = 42

BASELINE = dict(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY)

# sweep values (parts D-F)
ALPHAS = [0.01, 0.1, 0.5]
GAMMAS = [0.8, 0.95, 0.999]
EPS_DECAYS = [0.99, 0.995, 0.999]

# part H settings
NOISE_LEVELS = [0.0, 0.10, 0.25]
EVAL_EPISODES = 50
EVAL_NOISE_SEED = SEED + 999  # same noise stream for every eval run

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PLOTS = os.path.join(RESULTS, "plots")
VIDEOS = os.path.join(RESULTS, "videos")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(VIDEOS, exist_ok=True)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# training + evaluation
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Part C: baseline training
# ---------------------------------------------------------------------------
def part_c_baseline():
    print("== (C) Baseline training ==", flush=True)
    print("Baseline config:", BASELINE, flush=True)
    env = make_env()
    agent = QLearningAgent(seed=SEED, **BASELINE)
    rewards, steps = train_agent(agent, env, EPISODES, seed=SEED)
    env.close()

    agent.save(os.path.join(RESULTS, "qtable_baseline.npy"))
    np.save(os.path.join(RESULTS, "rewards_baseline.npy"), rewards)
    plot_curves(
        [(f"baseline {BASELINE}", rewards)],
        os.path.join(PLOTS, "baseline_reward_vs_episode.png"),
        "Baseline: reward vs episode (FrozenLake-v1, deterministic)",
    )
    print(f"Done. Mean reward over last 100 episodes: "
          f"{rewards[-100:].mean():.3f}", flush=True)


# ---------------------------------------------------------------------------
# Parts D-G: sweeps + best-agent evaluation
# ---------------------------------------------------------------------------
def run_config(cfg, episodes=EPISODES, seed=SEED):
    env = make_env()
    agent = QLearningAgent(seed=seed, **cfg)
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
        print(f"  {key}={v}: mean reward (last 500 eps) = {final[str(v)]:.3f}",
              flush=True)
    plot_curves(curves, os.path.join(PLOTS, plot_name), title)
    return final


def parts_d_to_g():
    summary = {"baseline": BASELINE}

    print("== (D) Learning-rate sweep ==", flush=True)
    summary["alpha_sweep"] = sweep(ALPHAS, "alpha",
        "Learning-rate sweep: reward vs episode", "alpha_comparison.png")

    print("== (E) Discount-factor sweep ==", flush=True)
    summary["gamma_sweep"] = sweep(GAMMAS, "gamma",
        "Discount-factor sweep: reward vs episode", "gamma_comparison.png")

    print("== (F) Exploration-decay sweep ==", flush=True)
    summary["epsilon_decay_sweep"] = sweep(EPS_DECAYS, "epsilon_decay",
        "Exploration-decay sweep: reward vs episode",
        "epsilon_decay_comparison.png")

    # (G) pick best (alpha, gamma) by late-training reward, evaluate greedily
    best_alpha = max(summary["alpha_sweep"], key=summary["alpha_sweep"].get)
    best_gamma = max(summary["gamma_sweep"], key=summary["gamma_sweep"].get)
    best_cfg = dict(BASELINE, alpha=float(best_alpha), gamma=float(best_gamma))
    print(f"== (G) Evaluating best config {best_cfg} (greedy, 50 eps) ==",
          flush=True)
    best_agent, _ = run_config(best_cfg)
    best_agent.save(os.path.join(RESULTS, "qtable_best.npy"))

    env = make_env()
    summary["best_config"] = best_cfg
    summary["best_eval_greedy_50eps"] = evaluate_policy(best_agent, env,
                                                        episodes=50)
    env.close()
    print("  best eval:", summary["best_eval_greedy_50eps"], flush=True)

    # re-train the baseline for a fair greedy comparison
    env = make_env()
    base_agent = QLearningAgent(seed=SEED, **BASELINE)
    train_agent(base_agent, env, EPISODES, seed=SEED)
    summary["baseline_eval_greedy_50eps"] = evaluate_policy(base_agent, env,
                                                           episodes=50)
    env.close()
    print("  baseline eval:", summary["baseline_eval_greedy_50eps"],
          flush=True)

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {RESULTS}/summary.json and qtable_best.npy", flush=True)


# ---------------------------------------------------------------------------
# Part H (extra credit): noise robustness
# ---------------------------------------------------------------------------
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


def part_h_noise():
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
        print(f"  p=0 policy @ p={p:.2f}: "
              f"mean_reward={metrics['mean_reward']:.3f} "
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
    print(f"Saved {json_path} and {plot_path} in {dt:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# video of the trained greedy policy
# ---------------------------------------------------------------------------
MAP = ["SFFF",
       "FHFH",
       "FFFH",
       "HFFG"]
COLORS = {"S": "#9ecbff", "F": "#f2f2f2", "H": "#ff9e9e", "G": "#9eff9e"}


def draw_state(state):
    # draw the 4x4 grid with matplotlib (FrozenLake's own rgb_array renderer
    # needs pygame, so this is simpler)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    for r in range(4):
        for c in range(4):
            ch = MAP[r][c]
            ax.add_patch(patches.Rectangle((c, 3 - r), 1, 1,
                                           facecolor=COLORS[ch],
                                           edgecolor="black", linewidth=1.5))
            ax.text(c + 0.5, 3 - r + 0.5, ch, ha="center", va="center",
                    fontsize=22, fontweight="bold")
    r, c = divmod(state, 4)
    ax.plot(c + 0.5, 3 - r + 0.5, "o", ms=30, color="#1f6feb",
            markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def rollout_states(agent, env, seed):
    states = []
    state, _ = env.reset(seed=seed)
    states.append(state)
    done = False
    while not done:
        action = agent.act(state, explore=False)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(state)
    return states


def render_video():
    print("== Rendering trained-agent video ==", flush=True)
    agent = QLearningAgent()
    agent.load(os.path.join(RESULTS, "qtable_best.npy"))
    env = make_env()

    frames = []
    for ep in range(3):  # three greedy episodes stitched together
        for s in rollout_states(agent, env, seed=2026 + ep):
            frames.extend([draw_state(s)] * 2)  # ~0.4 s per position at 5 fps
    env.close()

    out = os.path.join(VIDEOS, "trained_agent.mp4")
    imageio.mimsave(out, frames, fps=5, macro_block_size=None)
    print(f"Wrote {out} ({len(frames)} frames)", flush=True)


# ---------------------------------------------------------------------------
def main():
    part_c_baseline()
    parts_d_to_g()
    part_h_noise()
    render_video()
    print("All done.")


if __name__ == "__main__":
    main()
