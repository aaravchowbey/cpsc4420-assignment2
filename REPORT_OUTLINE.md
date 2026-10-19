# REPORT_OUTLINE.md — bullet outline for YOUR PDF report (write it in your own words)

> This is a writing scaffold, not the report itself. Expand each bullet into
> 2–5 sentences of your own. Do not copy these bullets verbatim into the PDF.

## 1. Introduction
- Goal of the assignment in one paragraph: implement tabular Q-learning from
  scratch and study how hyperparameters affect learning in a Gymnasium env.
- One sentence on why tabular methods are a good starting point (exact
  Q-table, no approximation error) before moving to deep RL.

## 2. Environment (Part A, 10 pts)
- Chose `FrozenLake-v1` with `is_slippery=False` (deterministic).
- Describe the MDP: 16 states (4×4 grid), 4 actions (LEFT/DOWN/RIGHT/UP).
- Map layout — name the start (S), holes (H, terminal, reward 0), goal (G,
  terminal, reward +1); all other transitions give reward 0.
- Transition dynamics: deterministic; bumping into a wall leaves the agent
  in place. Episode ends on hole, goal, or 100-step cap.
- No state discretization was needed (already discrete); justify that choice
  in one sentence.
- Figure idea: include a small diagram of the 4×4 map (you can screenshot a
  frame from `results/videos/trained_agent.mp4`).

## 3. Method: tabular Q-learning agent (Part B, 25 pts)
- Q-table shape 16×4, zero-initialized; explain what zero init implies
  (all actions look equal → early behavior is pure exploration).
- Epsilon-greedy action selection; epsilon decays geometrically per episode
  down to a floor.
- Write the update rule you implemented (Q-learning, off-policy TD):
  Q(s,a) ← Q(s,a) + α [r + γ·max_a' Q(s',a') − Q(s,a)], with the bootstrap
  term dropped (0) on terminal transitions — explain why.
- Note that α, γ, ε, ε-decay, ε-min are constructor arguments, so every
  experiment below just changes one of them.
- Mention tie-breaking: `argmax` picks the lowest action index on ties
  (relevant to the exploration discussion later).

## 4. Baseline training (Part C, 10 pts)
- Config: α=0.1, γ=0.99, ε: 1.0 → 0.05 with per-episode decay 0.999;
  4000 episodes, seed 42.
- Figure: `results/plots/baseline_reward_vs_episode.png` (100-ep moving avg).
- Describe the curve in your words: starts near 0, climbs steeply around
  episodes 500–1500 as the goal is discovered, plateaus ≈0.94–0.95.
- Explain the shape: early random exploration rarely hits the goal (sparse
  reward); once Q-values propagate back from the goal, exploitation takes
  over and the curve flattens near the optimum.

## 5. Learning-rate experiment (Part D, 15 pts)
- Setup: α ∈ {0.01, 0.1, 0.5}, everything else = baseline.
- Figure: `results/plots/alpha_comparison.png`.
- Results (mean reward, last 500 eps): 0.01 → 0.946, 0.1 → 0.948,
  0.5 → 0.946 — all three converge.
- Discussion points: 0.01 learns noticeably slower early (small steps);
  0.5 catches up fast but its curve is slightly noisier (large updates
  overwrite); 0.1 is the best trade-off here. On this tiny deterministic
  task the final performance is insensitive to α once exploration is
  adequate — say why you think that is.

## 6. Discount-factor experiment (Part E, 15 pts)
- Setup: γ ∈ {0.8, 0.95, 0.999}, everything else = baseline.
- Figure: `results/plots/gamma_comparison.png`.
- Results (last 500 eps): 0.8 → 0.946, 0.95 → 0.948, 0.999 → 0.948.
- Discussion: differences are tiny because the optimal path is only 6
  steps — even γ=0.8 discounts the goal reward to 0.8⁶ ≈ 0.26, still a
  clear signal. Explain the general principle (small γ = myopic, may
  prefer short-term rewards; large γ = far-sighted but slower credit
  assignment / higher variance) and why it barely mattered here.
- Honest note: a longer-horizon or denser-reward env would separate them.

## 7. Extra hyperparameter: exploration-decay schedule (Part F, 10 pts)
- Setup: ε-decay ∈ {0.99, 0.995, 0.999}, everything else = baseline.
- Figure: `results/plots/epsilon_decay_comparison.png` — your strongest plot.
- Results (last 500 eps): 0.99 → 0.000, 0.995 → 0.000, 0.999 → 0.948.
- This is the headline finding: with fast decay, ε hits its floor within a
  few hundred episodes; the greedy policy then exploits an all-zero
  Q-table, and tie-breaking makes it repeat LEFT into a wall forever —
  the goal is never rediscovered, so learning flatlines at 0.
- With slow decay (0.999, ε≈0.37 at episode 1000), exploration persists
  long enough to find the goal repeatedly and propagate values.
- Takeaway to state: in sparse-reward tasks, the exploration schedule can
  matter more than α or γ.

## 8. Evaluation of the best agent (Part G, 15 pts)
- Best config by late-training reward: α=0.1, γ=0.95 (ε-decay 0.999).
- Protocol: 50 episodes, ε=0 (pure greedy), fresh seeds.
- Results: mean reward 1.00 ± 0.00, success rate 100%, mean steps 6.0.
- Baseline for comparison (same protocol): identical — 100%, 6.0 steps.
- Point out that 6 steps is the optimal path length on this map
  (e.g. DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT avoiding holes), so the
  learned policy is optimal, not just good.
- Extra metrics reported: success rate and steps-to-goal alongside mean
  reward; explain why success rate is the more informative metric under a
  0/1 sparse reward.

## 9. Conclusion
- 3–4 sentences: tabular Q-learning solved the task; α/γ were forgiving,
  exploration scheduling was decisive; greedy policy is optimal (6 steps).
- Limitations (1 paragraph): single seed per config — rerunning with
  several seeds and reporting mean±std would strengthen the claims;
  deterministic env — the stochastic variant would test robustness;
  tiny grid — results may not transfer to larger spaces where tabular
  methods break down.
- Optional extra credit (H): you skipped the noise-robustness study; name
  it as future work (test noise levels 0 / 0.10 / 0.25 on
  `is_slippery=True`).

## Figures checklist (embed these PNGs)
1. `baseline_reward_vs_episode.png` — Section 4
2. `alpha_comparison.png` — Section 5
3. `gamma_comparison.png` — Section 6
4. `epsilon_decay_comparison.png` — Section 7
- Video: link `results/videos/trained_agent.mp4` in the report and include
  its public link via the GitHub repo.

## Submission checklist (do this yourself)
- [ ] Write the PDF in your own words from this outline; put your name on it
- [ ] Create a public GitHub repo, push this whole folder
- [ ] README.md is already written — check the repo renders it
- [ ] Confirm `results/plots/*.png` and `results/videos/trained_agent.mp4`
      are in the repo and the links work
- [ ] Upload the PDF (only the PDF — not code files) to Canvas before
      Sun Oct 18, 11:59pm (10% per day late)

## Key numbers cheat-sheet (from results/summary.json)
- alpha sweep (last-500-ep mean): 0.01→0.946, 0.1→0.948, 0.5→0.946
- gamma sweep: 0.8→0.946, 0.95→0.948, 0.999→0.948
- epsilon-decay sweep: 0.99→0.000, 0.995→0.000, 0.999→0.948
- best greedy eval (50 eps): mean 1.00, success 100%, 6.0 steps
- baseline greedy eval (50 eps): mean 1.00, success 100%, 6.0 steps
