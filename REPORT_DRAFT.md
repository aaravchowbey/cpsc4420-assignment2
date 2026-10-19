# Assignment 2: Tabular Q-Learning in a Gymnasium Environment
### CPSC 4420 — Artificial Intelligence, Fall 2026
### Aarav Chowbey

> **Draft prepared by Muse from your REPORT_OUTLINE.md and results/summary.json.**
> **You must revise this into your own words before submitting** — treat this
> as a first draft, not a final product. Add your name above, then convert to
> PDF with the four figures embedded.

---

## 1. Introduction

The goal of this assignment was to implement tabular Q-learning from scratch
and to study how key hyperparameters affect learning in a Gymnasium
environment. Rather than relying on an existing reinforcement-learning
library, I wrote the agent, the training loop, the evaluation code, and the
plotting utilities myself, so that every part of the learning process is
explicit and observable.

Tabular Q-learning is a natural starting point for studying reinforcement
learning because it stores an exact value estimate for every
state–action pair. There is no function approximation, and therefore no
approximation error or generalization behavior to disentangle from the
learning dynamics themselves. Whatever the agent learns is visible directly
in the table, which makes it an ideal setting for isolating the effect of
individual hyperparameters — learning rate, discount factor, and the
exploration schedule — before moving on to deep reinforcement learning, where
those effects interact with neural-network training dynamics.

## 2. Environment (Part A)

I chose the deterministic variant of `FrozenLake-v1` (`is_slippery=False`).
The environment is a 4×4 grid, so the Markov decision process has 16 states
and 4 actions (LEFT, DOWN, RIGHT, UP). The map contains a start state (S),
several holes (H), and a goal (G):

```
SFFF
FHFH
FFFH
HFFG
```

Stepping into a hole ends the episode with reward 0; reaching the goal ends
the episode with reward +1. Every other transition yields reward 0, so the
reward signal is sparse — the agent receives feedback only at the goal.
Transitions are fully deterministic: the agent moves exactly where it
intends, and attempting to move into a wall simply leaves it in place.
Episodes also terminate after a 100-step cap.

Because the state space is already discrete, no state discretization was
needed; a 16×4 table covers the entire problem exactly, which is precisely
what makes this environment well suited to a tabular method.

## 3. Method: Tabular Q-Learning Agent (Part B)

The agent maintains a Q-table of shape 16×4, zero-initialized. Zero
initialization means every action initially looks equally good, so early
behavior is driven entirely by exploration rather than by any prior bias in
the values.

Action selection is epsilon-greedy: with probability ε the agent picks a
uniformly random action, and otherwise it acts greedily with respect to the
current Q-values. Epsilon starts at 1.0 and decays geometrically at the end
of each episode down to a floor of ε_min = 0.05, so exploration is gradually
replaced by exploitation as training progresses.

The agent learns with the standard Q-learning update, an off-policy
temporal-difference rule:

> Q(s,a) ← Q(s,a) + α [ r + γ · max_a′ Q(s′,a′) − Q(s,a) ]

where α is the learning rate and γ the discount factor. On terminal
transitions (hole or goal) the bootstrap term γ·max_a′ Q(s′,a′) is dropped —
there is no future value after termination, so the target reduces to the
immediate reward r. All hyperparameters (α, γ, ε, ε-decay, ε_min) are
constructor arguments of the agent, so each experiment below changes exactly
one of them while holding the rest fixed. One implementation detail worth
noting: on ties, `argmax` selects the lowest action index (LEFT). This
deterministic tie-breaking becomes important when interpreting the
exploration experiment in Section 7.

## 4. Baseline Training (Part C)

The baseline configuration was α = 0.1, γ = 0.99, ε decaying from 1.0 to
0.05 with a per-episode decay of 0.999, trained for 4000 episodes with seed
42. Figure 1 shows the learning curve (100-episode moving average of
per-episode reward).

**Figure 1.** `results/plots/baseline_reward_vs_episode.png` — 100-episode
moving average of reward during baseline training.

The curve starts near zero and climbs steeply between roughly episodes 500
and 1500 before plateauing at approximately 0.94–0.95. This shape is exactly
what one expects under a sparse reward: early on, near-random exploration
rarely stumbles onto the goal, so almost every episode returns 0. Once the
goal is discovered a few times, the +1 reward propagates backward through
the Q-table along the successful trajectories, exploitation takes over, and
the curve flattens near the optimum. The remaining gap to 1.0 reflects the
residual ε = 0.05 random actions taken even late in training.

## 5. Learning-Rate Experiment (Part D)

I trained three agents with α ∈ {0.01, 0.1, 0.5}, keeping every other
hyperparameter at its baseline value. Figure 2 compares their learning
curves.

**Figure 2.** `results/plots/alpha_comparison.png` — learning curves for
α = 0.01, 0.1, 0.5 (100-episode moving average).

Measured by mean reward over the final 500 episodes, all three converged to
essentially the same performance: 0.946 (α=0.01), 0.948 (α=0.1), and 0.946
(α=0.5). The differences are in the dynamics rather than the destination.
The α=0.01 agent learns noticeably more slowly early in training, since each
update takes only a small step toward the TD target. The α=0.5 agent catches
up quickly but its curve is slightly noisier, because large updates
overwrite previously learned values more aggressively. α=0.1 offers the best
trade-off between speed and stability here.

The broader lesson is that, on this tiny deterministic task, final
performance is insensitive to the learning rate once exploration is
adequate. With only 64 state–action pairs and a deterministic environment,
every reasonable learning rate eventually propagates the goal reward through
the whole table; the learning rate mostly controls how quickly that happens,
not whether it happens.

## 6. Discount-Factor Experiment (Part E)

Next I varied γ ∈ {0.8, 0.95, 0.999} with all other settings at baseline
(Figure 3).

**Figure 3.** `results/plots/gamma_comparison.png` — learning curves for
γ = 0.8, 0.95, 0.999 (100-episode moving average).

Final-500-episode mean rewards were 0.946, 0.948, and 0.948 — again,
practically identical. The reason is the short horizon: the optimal path is
only 6 steps, so even γ=0.8 discounts the goal reward to 0.8⁶ ≈ 0.26, which
is still a clear and unambiguous learning signal relative to the zero reward
available everywhere else. The general principle still holds — a small
discount factor makes the agent myopic and can cause it to prefer small
immediate rewards over larger delayed ones, while a large discount factor is
far-sighted but slows credit assignment and increases variance — but on this
environment the horizon is too short for those forces to separate the
curves. A longer-horizon or denser-reward environment would be needed to see
meaningful differences between discount factors.

## 7. Extra Hyperparameter: Exploration-Decay Schedule (Part F)

The most striking result came from varying the epsilon-decay rate over
{0.99, 0.995, 0.999} (Figure 4).

**Figure 4.** `results/plots/epsilon_decay_comparison.png` — learning curves
for ε-decay = 0.99, 0.995, 0.999 (100-episode moving average).

The final-500-episode mean rewards were 0.000, 0.000, and 0.948. With the
two faster decay schedules, learning completely flatlines at zero.

The mechanism is worth spelling out. With a decay of 0.99, ε collapses to
its floor within a few hundred episodes — before the agent has reliably
discovered the sparse goal reward. The greedy policy then exploits an
all-zero Q-table, and because `argmax` breaks ties toward the lowest action
index, the agent deterministically repeats LEFT, walking into a wall forever.
The goal is never rediscovered, no non-zero TD target is ever generated, and
learning stalls permanently at zero. With the slow decay of 0.999
(ε ≈ 0.37 as late as episode 1000), random exploration persists long enough
for the agent to find the goal repeatedly and propagate values back through
the table.

The takeaway: in sparse-reward tasks, the exploration schedule can matter
far more than the learning rate or discount factor. Both α and γ were
forgiving; exploration was decisive.

## 8. Evaluation of the Best Agent (Part G)

By late-training reward, the best configuration was α=0.1, γ=0.95 with
ε-decay 0.999 (the baseline run with γ=0.99 was essentially tied, but the
γ=0.95 sweep run edged it). I evaluated this agent over 50 episodes with
ε=0 (pure greedy policy) on fresh seeds, alongside the baseline agent under
the identical protocol.

| Agent | Mean reward | Std | Success rate | Mean steps |
|---|---|---|---|---|
| Best (α=0.1, γ=0.95) | 1.00 | 0.00 | 100% | 6.0 |
| Baseline (α=0.1, γ=0.99) | 1.00 | 0.00 | 100% | 6.0 |

Both agents achieve a 100% success rate with a mean of 6.0 steps to the goal.
Six steps is the optimal path length on this map (e.g., DOWN, DOWN, RIGHT,
RIGHT, DOWN, RIGHT while avoiding the holes), so the learned policy is not
merely good — it is optimal.

Reporting success rate and steps-to-goal alongside mean reward matters here:
under a 0/1 sparse reward, mean reward and success rate coincide, but the
success rate is the more directly interpretable metric — it answers "how
often does the agent actually solve the task?" — while steps-to-goal
distinguishes an optimal policy from a merely successful one.

A video of the trained greedy agent is available at
`results/videos/trained_agent.mp4` (also linked from the public code
repository).

## 9. Conclusion

Tabular Q-learning solved the FrozenLake task completely: the best agent's
greedy policy reaches the goal on every evaluation episode via a 6-step
optimal path. The hyperparameter studies showed that the learning rate and
discount factor were forgiving on this small deterministic problem, while
the exploration schedule was decisive — decaying exploration too quickly
collapses learning to zero in a sparse-reward setting, a failure mode worth
remembering whenever rewards are rare.

**Limitations.** Each configuration was run with a single seed, so the
reported numbers carry no variance estimates; rerunning with several seeds
and reporting mean ± std would strengthen the claims, particularly for the
noisy early-learning comparisons. The environment was deterministic, so the
results say nothing about robustness to stochastic transitions — the
slippery variant would be the natural test. Finally, this is a tiny grid;
on larger state spaces a tabular method breaks down entirely, which is where
function approximation becomes necessary.

**Future work.** I did not attempt the extra-credit noise-robustness study
(Part H). The natural extension is to train and evaluate on
`is_slippery=True` at several noise levels (e.g., 0, 0.10, 0.25) and measure
how the success rate and the optimal exploration schedule change when the
environment itself injects randomness.

---

## Figures checklist (embed these PNGs in the PDF)

1. `results/plots/baseline_reward_vs_episode.png` — Section 4
2. `results/plots/alpha_comparison.png` — Section 5
3. `results/plots/gamma_comparison.png` — Section 6
4. `results/plots/epsilon_decay_comparison.png` — Section 7
5. Video: link `results/videos/trained_agent.mp4` (include the public
   GitHub link once the repo is up).

## Submission checklist

- [ ] Revise this draft into your own words; put your name on the PDF
- [ ] Create a public GitHub repo, push this whole folder
- [ ] Check the README renders on GitHub
- [ ] Confirm `results/plots/*.png` and `results/videos/trained_agent.mp4`
      are in the repo and the links work
- [ ] Upload the PDF (**only** the PDF — not code files) to Canvas before
      Sun Oct 18, 11:59pm (10% per day late)
