# Multi-Agent Reinforcement Learning Final Project - PSRO on Kuhn Poker

**Final Project (Grade 100, M.Sc. Data Science, HIT). Policy-Space Response Oracles built from scratch on Kuhn Poker — a population of policies per player, an empirical meta-game, three response oracles and eight meta-strategy solvers, in NumPy/SciPy/pandas with no RL library. Does solving a population produce a more robust agent than self-play under the same training budget?**

## Headline Results

- **Same training budget:** exploitability **0.0315** against **1.2500** for self-play — a factor of **39.7**, no overlap across five seeds.
- **The right game value:** error **0.0004** against **0.8056**.
- **The equilibrium, unaided:** the learned mixture lands on Kuhn's analytic equilibrium family, which no training module references.
- **The trade-off, kept in:** against weak unseen opponents self-play earns more. Robustness and exploitation are not the same metric.

## Key Features

- **Exact Ground Truth:** every pure strategy enumerated, so exploitability is computed, not estimated.
- **PSRO Loop with Checkpointing:** one configuration object, and interrupted runs resume where they stopped.
- **Three Oracles, Eight Solvers:** choosing the solver chooses the algorithm — from self-play to double oracle.
- **Error Decomposition:** the cost of sampling the payoff matrix separated from the cost of approximate training.
- **Statistics and Ablations:** five seeds, confidence intervals, and five ablations showing what does the work.
- **Bonus Extension - Rectified PSRO:** implemented independently, its documented failure reproduced.
- **89 Automated Tests:** all pass in about 40 seconds.

## Repository Structure

- **`Multi_Agent_Reinforcement_Learning_Final_Project.ipynb`**: Full solution notebook — validation against the analytic solution, one complete PSRO run with save and resume, all five experiments, the ablations and the analysis (explanations in Hebrew). Written in Colab — to run it locally, skip the `google.colab` mount cell and point `PATH` at the project directory.
- **`game.py`**: Kuhn Poker — game tree, information sets, policies, episode simulation, head-to-head matrices.
- **`exact.py`**: Exact quantities — game value, strategy enumeration, best response, exploitability, population exploitability, Kuhn equilibrium parameters.
- **`metagame.py`**: The empirical meta-game — incremental estimator with standard errors and confidence intervals, dominated policies, cycle counting.
- **`oracles.py`**: The three response oracles and the single-policy self-play learner.
- **`metasolvers.py`**: The eight meta-strategy solvers.
- **`psro.py`**: The configurable PSRO loop, per-iteration logging, save and resume.
- **`analysis.py`**: Behavioural diversity measures, rliable-style statistics, and every figure.
- **`experiments.py`**: The five experiments and the ablation study, with result caching and a CLI.
- **`test_psro.py`**: The 89 correctness tests.
- **`__pycache__/`** — the compiled bytecode CPython 3.13 writes the first time each module is imported. One `.pyc` per module, regenerated automatically whenever a source file changes.
  - `analysis.cpython-313.pyc`
  - `exact.cpython-313.pyc`
  - `experiments.cpython-313.pyc`
  - `game.cpython-313.pyc`
  - `metagame.cpython-313.pyc`
  - `metasolvers.cpython-313.pyc`
  - `oracles.cpython-313.pyc`
  - `psro.cpython-313.pyc`
- **`configs/`** — one file per experiment, listing every run it launches with all its parameters. 131 runs in total.
  - `experiment1_selfplay_vs_psro.json` (10 runs)
  - `experiment2_metasolvers.json` (40 runs)
  - `experiment3_population.json` (1 run)
  - `experiment4_oracle_quality.json` (30 runs)
  - `experiment5_generalisation.json` (20 runs)
  - `ablations.json` (30 runs)
- **`results/`** — `*.csv` is one row per iteration, `*_summary.csv` one row per run, and `*_extra.json` holds payoff matrices, meta-strategies and curves.
  - `experiment1_selfplay_vs_psro.csv` (120 rows)
  - `experiment1_selfplay_vs_psro_summary.csv` (10 rows)
  - `experiment1_selfplay_vs_psro_extra.json`
  - `experiment2_metasolvers.csv` (480 rows)
  - `experiment2_metasolvers_summary.csv` (40 rows)
  - `experiment2_metasolvers_extra.json`
  - `experiment3_population.csv` (12 rows)
  - `experiment3_population_extra.json`
  - `experiment4_oracle_quality.csv` (360 rows)
  - `experiment4_oracle_quality_summary.csv` (30 rows)
  - `experiment4_oracle_quality_extra.json`
  - `experiment5_generalisation.csv` (240 rows)
  - `experiment5_generalisation_summary.csv` (20 rows)
  - `experiment5_generalisation_extra.json`
  - `ablations.csv` (360 rows)
  - `ablations_summary.csv` (30 rows)
  - `ablations_extra.json`
- **`psro_kuhn_report.pdf`**: The eight-page report.
- **`psro_kuhn_presentation.pdf`**: The presentation, 20 slides: 17 for the talk and three appendix slides.
- **`requirements.txt`**: NumPy, pandas, SciPy, matplotlib, pytest. Python 3.10 or newer. `pip install -r requirements.txt`, then `pytest test_psro.py -q` or `python experiments.py --all` (which loads the saved results; `--force` recomputes them).
- **`final_project_instructions.pdf`**: Original course project specification.

## References

- Lanctot, M. et al. (2017). [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832). NeurIPS - PSRO itself.
- McMahan, H. B. et al. (2003). [Planning in the Presence of Cost Functions Controlled by an Adversary](https://www.cs.cmu.edu/~ggordon/mcmahan-ggordon-blum.icml2003.pdf). ICML - double oracle.
- Balduzzi, D. et al. (2019). [Open-ended Learning in Symmetric Zero-sum Games](https://arxiv.org/abs/1901.08106). ICML - rectified Nash response, the bonus extension.
- Wang, Y. et al. (2022). [Evaluating Strategy Exploration in Empirical Game-Theoretic Analysis](https://arxiv.org/abs/2105.10423). AAMAS - population exploitability.
- Kuhn, H. W. (1950). *A Simplified Two-Person Poker*. Contributions to the Theory of Games I - the equilibrium family.

Also cited in the report: [McAleer et al. (2020)](https://arxiv.org/abs/2006.08555) and [(2021)](https://arxiv.org/abs/2103.06426), [Muller et al. (2020)](https://arxiv.org/abs/1909.12823), [Zhang and Sandholm (2024)](https://arxiv.org/abs/2405.06797), [Agarwal et al. (2021)](https://arxiv.org/abs/2108.13264).
