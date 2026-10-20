"""Render the trained greedy policy to an MP4 video.

FrozenLake's built-in rgb_array renderer needs pygame, so the 4x4 grid
is drawn directly with matplotlib and encoded with imageio-ffmpeg.

Usage:  python make_video.py [path/to/qtable.npy]   (default: results/qtable_best.npy)
Output: results/videos/trained_agent.mp4
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import imageio.v2 as imageio

from qlearning import QLearningAgent, make_env

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.join(HERE, "results", "videos")
os.makedirs(VIDEOS, exist_ok=True)

# FrozenLake-v1 4x4 map (row 0 at top)
MAP = ["SFFF",
       "FHFH",
       "FFFH",
       "HFFG"]
COLORS = {"S": "#9ecbff", "F": "#f2f2f2", "H": "#ff9e9e", "G": "#9eff9e"}


def draw_state(state):
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


def main():
    qtable = (sys.argv[1] if len(sys.argv) > 1
              else os.path.join(HERE, "results", "qtable_best.npy"))
    agent = QLearningAgent()
    agent.load(qtable)
    env = make_env()

    frames = []
    for ep in range(3):  # three greedy episodes stitched together
        for s in rollout_states(agent, env, seed=2026 + ep):
            frames.extend([draw_state(s)] * 2)  # ~0.4 s per position at 5 fps
    env.close()

    out = os.path.join(VIDEOS, "trained_agent.mp4")
    imageio.mimsave(out, frames, fps=5, macro_block_size=None)
    print(f"Wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
