# Multi-Agent Reinforcement Learning Final Project - PSRO on Kuhn Poker

**Final Project (Grade 100, M.Sc. Data Science, HIT). Policy-Space Response Oracles built from scratch on Kuhn Poker — an empirical meta-game, three response oracles and eight meta-strategy solvers, in NumPy/SciPy/pandas with no RL library. Does solving a population produce a more robust agent than self-play under the same budget?**

## Headline Results

- **Same training budget:** exploitability **0.0315** against self-play's **1.2500** — **39.7×** lower, no overlap across five seeds.
- **The right game value:** error **0.0004** against **0.8056**.
- **The equilibrium, unaided:** the learned mixture lands on Kuhn's equilibrium family, which no code references.
- **The trade-off, kept in:** against weak opponents self-play earns more. Robustness and exploitation are distinct.

## Key Features

- **Exact Ground Truth:** every pure strategy enumerated, so exploitability is computed, not estimated.
- **PSRO Loop with Checkpointing:** one configuration object, and interrupted runs resume where they stopped.
- **Three Oracles, Eight Solvers:** choosing the solver chooses the algorithm — from self-play to double oracle.
- **Error Decomposition:** the cost of sampling the payoff matrix separated from the cost of approximate training.
- **Statistics and Ablations:** five seeds, confidence intervals, and five ablations showing what does the work.
- **Bonus Extension - Rectified PSRO:** implemented independently, its documented failure reproduced.
- **89 Automated Tests:** all pass in about 40 seconds.

## Repository Structure

- **`__pycache__/`** — compiled bytecode, one `.pyc` per module for each Python version (3.11 and 3.13).
  - `analysis.cpython-311.pyc`
  - `analysis.cpython-313.pyc`
  - `exact.cpython-311.pyc`
  - `exact.cpython-313.pyc`
  - `experiments.cpython-313.pyc`
  - `game.cpython-311.pyc`
  - `game.cpython-313.pyc`
  - `metagame.cpython-311.pyc`
  - `metagame.cpython-313.pyc`
  - `metasolvers.cpython-311.pyc`
  - `metasolvers.cpython-313.pyc`
  - `oracles.cpython-311.pyc`
  - `oracles.cpython-313.pyc`
  - `psro.cpython-311.pyc`
  - `psro.cpython-313.pyc`
  - `test_psro.cpython-311-pytest-9.0.2.pyc`
- **`configs/`** — one file per experiment, listing every run it launches with all its parameters. 131 runs in total.
  - `ablations.json` (30 runs)
  - `experiment1_selfplay_vs_psro.json` (10 runs)
  - `experiment2_metasolvers.json` (40 runs)
  - `experiment3_population.json` (1 run)
  - `experiment4_oracle_quality.json` (30 runs)
  - `experiment5_generalisation.json` (20 runs)
- **`results/`** — per-iteration (`*.csv`), per-run (`*_summary.csv`), payoff matrices (`*_extra.json`).
  - `ablations.csv` (360 rows)
  - `ablations_extra.json`
  - `ablations_summary.csv` (30 rows)
  - `experiment1_selfplay_vs_psro.csv` (120 rows)
  - `experiment1_selfplay_vs_psro_extra.json`
  - `experiment1_selfplay_vs_psro_summary.csv` (10 rows)
  - `experiment2_metasolvers.csv` (480 rows)
  - `experiment2_metasolvers_extra.json`
  - `experiment2_metasolvers_summary.csv` (40 rows)
  - `experiment3_population.csv` (12 rows)
  - `experiment3_population_extra.json`
  - `experiment4_oracle_quality.csv` (360 rows)
  - `experiment4_oracle_quality_extra.json`
  - `experiment4_oracle_quality_summary.csv` (30 rows)
  - `experiment5_generalisation.csv` (240 rows)
  - `experiment5_generalisation_extra.json`
  - `experiment5_generalisation_summary.csv` (20 rows)
- **`analysis.py`**: Behavioural diversity measures, rliable-style statistics, and every figure.
- **`exact.py`**: Exact game value, best response, exploitability and Kuhn equilibrium parameters.
- **`experiments.py`**: The five experiments and the ablation study, with result caching and a CLI.
- **`final_project_instructions.pdf`**: Original course project specification.
- **`game.py`**: Kuhn Poker — game tree, information sets, policies, episode simulation, head-to-head matrices.
- **`metagame.py`**: The empirical meta-game — payoff estimates with standard errors, dominance and cycles.
- **`metasolvers.py`**: The eight meta-strategy solvers.
- **`Multi_Agent_Reinforcement_Learning_Final_Project.ipynb`**: Notebook — experiments, ablations, analysis (Colab).
- - **`oracles.py`**: The three response oracles and the single-policy self-play learner.
- **`psro.py`**: The configurable PSRO loop, per-iteration logging, save and resume.
- **`psro_kuhn_presentation.pdf`**: The presentation, 20 slides: 17 for the talk and three appendix slides.
- **`psro_kuhn_report.pdf`**: The eight-page report.
- **`requirements.txt`**: Dependencies (Python 3.10+).
- **`test_psro.py`**: The 89 correctness tests.

## References

- Balduzzi et al. (2019). [Open-ended Learning in Symmetric Zero-sum Games](https://arxiv.org/abs/1901.08106). Rectified Nash (bonus extension).
- Kuhn (1950). *A Simplified Two-Person Poker*. The equilibrium family.
- Lanctot et al. (2017). [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832). PSRO.
- McMahan et al. (2003). [Planning in the Presence of Cost Functions Controlled by an Adversary](https://www.cs.cmu.edu/~ggordon/mcmahan-ggordon-blum.icml2003.pdf). Double oracle.
- Wang et al. (2022). [Evaluating Strategy Exploration in Empirical Game-Theoretic Analysis](https://arxiv.org/abs/2105.10423). Population exploitability.

Also cited: [Agarwal et al. (2021)](https://arxiv.org/abs/2108.13264), [McAleer et al. (2020)](https://arxiv.org/abs/2006.08555) and [(2021)](https://arxiv.org/abs/2103.06426), [Muller et al. (2020)](https://arxiv.org/abs/1909.12823), [Zhang and Sandholm (2024)](https://arxiv.org/abs/2405.06797).
