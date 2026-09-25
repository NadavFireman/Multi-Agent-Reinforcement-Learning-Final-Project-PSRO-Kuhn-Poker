# Multi-Agent Reinforcement Learning Final Project - PSRO on Kuhn Poker

**Final Project (Grade 100, M.Sc. Data Science, HIT). Policy-Space Response Oracles built from scratch on Kuhn Poker — a population of policies per player, an empirical meta-game, three response oracles and eight meta-strategy solvers, in NumPy/SciPy/pandas with no RL library. Does solving a population produce a more robust agent than self-play under the same training budget?**

## Headline Results
- **Same training budget:** exploitability **0.0315** against **1.2500** for self-play — a factor of **39.7**, no overlap across five seeds.
- **The right game value:** error **0.0004** against **0.8056**.
- **The equilibrium, unaided:** the learned mixture lands on Kuhn's analytic equilibrium family, which no training module references.
- **The trade-off, kept in:** against weak unseen opponents self-play earns more. Robustness and exploitation are not the same metric.

## Key Features
- **Exact Ground Truth:** small enough to enumerate every pure strategy, so best responses and exploitability are computed rather than estimated — three independent checks land at machine precision.
- **PSRO Loop with Checkpointing:** every parameter in one configuration object, everything serialized so an interrupted run resumes. With the exact oracle and the exact matrix it becomes the classical double oracle.
- **Three Oracles, Eight Solvers:** tabular Q-learning, REINFORCE and exact best response; uniform, latest-policy, fictitious play, Nash by linear programming, replicator, projected replicator, regret matching and rectified Nash. Choosing the solver chooses the algorithm — latest-policy gives self-play, Nash gives double oracle.
- **Error Decomposition:** the two approximations switched off one at a time, separating the cost of sampling the payoff matrix from the cost of approximating the best response. Same order of magnitude.
- **Statistics and Ablations:** five seeds, Student-t intervals, Mann-Whitney where a t-test divides by zero, rliable-style reporting. Five ablations rank what does the work: the population and the mixture, not the learner.
- **Bonus Extension - Rectified PSRO:** implemented independently, its documented failure reproduced.
- **89 Automated Tests:** rules, the analytic solution, exact best response, exploitability, every solver, the loop and checkpoint round-trips. All pass in about 40 seconds.

## Repository Structure
- `Multi_Agent_Reinforcement_Learning_Final_Project.ipynb`: Full solution notebook — validation against the analytic solution, one complete PSRO run with save and resume, all five experiments, the ablations and the analysis (explanations in Hebrew). Written in Colab; to run it locally, skip the Drive-mount cell and point `PATH` at the project directory.
- `game.py`: Kuhn Poker — game tree, information sets, policies, episode simulation, head-to-head matrices.
- `exact.py`: Exact quantities — game value, strategy enumeration, best response, exploitability, population exploitability, Kuhn equilibrium parameters.
- `metagame.py`: The empirical meta-game — incremental estimator with standard errors and confidence intervals, dominated policies, cycle counting.
- `oracles.py`: The three response oracles and the single-policy self-play learner.
- `metasolvers.py`: The eight meta-strategy solvers.
- `psro.py`: The configurable PSRO loop, per-iteration logging, save and resume.
- `analysis.py`: Behavioural diversity measures, rliable-style statistics, and every figure.
- `experiments.py`: The five experiments and the ablation study, with result caching and a CLI.
- `test_psro.py`: The 89 correctness tests.
- `configs/`: One file per experiment, listing every run it launches with all its parameters — 131 runs in total.
  - `experiment1_selfplay_vs_psro.json`: 10 runs.
  - `experiment2_metasolvers.json`: 40 runs.
  - `experiment3_population.json`: 1 run.
  - `experiment4_oracle_quality.json`: 30 runs.
  - `experiment5_generalisation.json`: 20 runs.
  - `ablations.json`: 30 runs.
- `results/`: `*.csv` has one row per iteration, `*_summary.csv` one row per run, and `*_extra.json` holds payoff matrices, meta-strategies and curves.
  - `experiment1_selfplay_vs_psro.csv`: Per-iteration metrics, self-play vs. PSRO.
  - `experiment1_selfplay_vs_psro_summary.csv`: One row per run.
  - `experiment1_selfplay_vs_psro_extra.json`: Exploitability curves.
  - `experiment2_metasolvers.csv`: Per-iteration metrics, eight meta-solvers.
  - `experiment2_metasolvers_summary.csv`: One row per run.
  - `experiment2_metasolvers_extra.json`: Payoff matrices, meta-strategies and curves per run.
  - `experiment3_population.csv`: Per-iteration metrics of the population run.
  - `experiment3_population_extra.json`: Estimated and exact payoff matrices, head-to-head, cycles and dominated policies.
  - `experiment4_oracle_quality.csv`: Per-iteration metrics, oracle budgets and the exact oracle.
  - `experiment4_oracle_quality_summary.csv`: One row per run.
  - `experiment4_oracle_quality_extra.json`: Payoff matrices, meta-strategies and curves per run.
  - `experiment5_generalisation.csv`: Per-iteration metrics, generalisation runs.
  - `experiment5_generalisation_summary.csv`: One row per run, with scores against held-out opponents.
  - `experiment5_generalisation_extra.json`: Payoff matrices, meta-strategies and curves per run.
  - `ablations.csv`: Per-iteration metrics, six ablation arms.
  - `ablations_summary.csv`: One row per run.
  - `ablations_extra.json`: Payoff matrices, meta-strategies and curves per run.
- `psro_kuhn_report.pdf`: The eight-page report.
- `psro_kuhn_presentation.pdf`: The presentation, 20 slides: 17 for the talk and three appendix slides.
- `requirements.txt`: NumPy, pandas, SciPy, matplotlib, pytest. Python 3.10 or newer. `pip install -r requirements.txt`, then `pytest test_psro.py -q` or `python experiments.py --all` (which loads the saved results; `--force` recomputes them).
- `final_project_instructions.pdf`: Original course project specification.

## References
- Lanctot, M. et al. (2017). [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832). NeurIPS - PSRO itself.
- McMahan, H. B., Gordon, G. J. and Blum, A. (2003). [Planning in the Presence of Cost Functions Controlled by an Adversary](https://www.cs.cmu.edu/~ggordon/mcmahan-ggordon-blum.icml2003.pdf). ICML - double oracle.
- Balduzzi, D. et al. (2019). [Open-ended Learning in Symmetric Zero-sum Games](https://arxiv.org/abs/1901.08106). ICML - rectified Nash response, the bonus extension.
- Wang, Y., Ma, G. Q. and Wellman, M. P. (2022). [Evaluating Strategy Exploration in Empirical Game-Theoretic Analysis](https://arxiv.org/abs/2105.10423). AAMAS - population exploitability.
- Kuhn, H. W. (1950). *A Simplified Two-Person Poker*. Contributions to the Theory of Games I - the equilibrium family.

Also cited in the report: [McAleer et al. (2020)](https://arxiv.org/abs/2006.08555) and [(2021)](https://arxiv.org/abs/2103.06426), [Muller et al. (2020)](https://arxiv.org/abs/1909.12823), [Zhang and Sandholm (2024)](https://arxiv.org/abs/2405.06797), [Agarwal et al. (2021)](https://arxiv.org/abs/2108.13264).
