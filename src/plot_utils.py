"""Plotting helpers (headless-safe)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_curves(curves, path, title, window=100,
                xlabel="Episode", ylabel="Reward (100-ep moving avg)"):
    """Plot smoothed learning curves. curves = [(label, rewards_array), ...]."""
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
