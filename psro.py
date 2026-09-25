# -*- coding: utf-8 -*-
"""The PSRO training loop (Part E of the project specification).

One iteration of Policy-Space Response Oracles:

1. evaluate every missing pair of population policies;
2. update the empirical payoff matrix;
3. run the meta-solver to get a meta-strategy for each player;
4. train an approximate best response against the opponent's meta-strategy;
5. add it to the population;
6. repeat until the iteration budget runs out or exploitability is small enough.

Choosing the meta-solver is what selects the algorithm: ``self_play`` recovers
ordinary self-play, ``nash`` gives the double-oracle algorithm, and the others
sit in between. This is what makes the project's central comparison a
controlled one, because every arm shares the same oracle, the same budget and
the same evaluation code.

Every policy, payoff matrix and meta-strategy is checkpointed, so an
interrupted experiment resumes exactly where it stopped.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from exact import ExactBestResponse, exact_payoff_matrix, exploitability, mixture_value
from metagame import EmpiricalMetaGame, cycle_strength, dominated_policies
from metasolvers import get_solver, strategy_entropy
from oracles import Oracle, get_oracle
from game import (Policy, ZeroSumGame, head_to_head_matrix,
                  policy_from_dict, policy_to_dict)


@dataclass
class PSROConfig:
    """Every knob of a PSRO run, so an experiment is one serialisable object."""

    game: str = "kuhn_poker"
    meta_solver: str = "nash"
    oracle: str = "q_learning"
    oracle_kwargs: dict = field(default_factory=dict)
    solver_kwargs: dict = field(default_factory=dict)

    iterations: int = 12
    oracle_budget: int = 10_000
    episodes_per_seed: int = 200
    num_seeds: int = 5

    seed: int = 0
    max_population: int | None = None
    exploitability_tolerance: float = 0.0

    #: Train against a single opponent drawn once per iteration, instead of
    #: resampling one per episode. This isolates the specification's rule that
    #: "the sampled opponent remains fixed for that episode" - here it is fixed
    #: for the whole iteration, so the oracle never sees the mixture at all.
    single_opponent: bool = False

    #: Drop policies that another single policy weakly dominates, at the end of
    #: every iteration. Pruning shrinks the meta-game, but a dominated policy
    #: can still be a useful stepping stone, so this is an ablation and not an
    #: optimisation.
    prune_dominated: bool = False

    #: Compute exact exploitability every iteration. Cheap for Kuhn Poker;
    #: switch off for larger games where enumeration is infeasible.
    track_exact_exploitability: bool = True

    #: Compute every meta-game entry by exact tree traversal instead of
    #: simulating episodes. Setting this together with ``oracle="exact"``
    #: removes every approximation from PSRO at once and turns it into the
    #: literal double-oracle algorithm of McMahan, Gordon and Blum (2003), for
    #: which convergence to an exact equilibrium is a theorem.
    exact_metagame: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IterationRecord:
    """Everything measured at one PSRO iteration."""

    iteration: int
    population0: int
    population1: int
    meta_strategy0: list[float]
    meta_strategy1: list[float]
    entropy0: float
    entropy1: float
    empirical_value: float
    exact_value: float | None
    exploitability: float | None
    br0_value: float | None
    br1_value: float | None
    oracle_return0: float
    oracle_return1: float
    oracle_seconds: float
    evaluation_episodes: int
    wall_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _pad(sigma: np.ndarray, size: int) -> np.ndarray:
    """Extend a meta-strategy with zeros for policies added after it was solved."""
    if len(sigma) == size:
        return sigma
    if len(sigma) > size:
        raise ValueError(f"meta-strategy of length {len(sigma)} exceeds population {size}")
    out = np.zeros(size)
    out[: len(sigma)] = sigma
    return out


def _restrict(sigma: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Keep the named entries of a meta-strategy and renormalise."""
    kept = np.array([sigma[i] for i in keep if i < len(sigma)], dtype=float)
    if len(kept) == 0:
        return np.array([1.0])
    total = kept.sum()
    return kept / total if total > 0 else np.full(len(kept), 1.0 / len(kept))


def _one_hot(sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Collapse a meta-strategy onto a single policy drawn from it."""
    out = np.zeros(len(sigma))
    out[int(np.searchsorted(np.cumsum(sigma), rng.random(), side="right"))
        if sigma.sum() > 0 else 0] = 1.0
    if out.sum() == 0.0:
        out[-1] = 1.0
    return out


class PSRO:
    """A configurable PSRO run over one two-player zero-sum game."""

    def __init__(
        self,
        game: ZeroSumGame,
        config: PSROConfig,
        initial_policies: tuple[Sequence[Policy], Sequence[Policy]],
        oracle: Oracle | None = None,
    ):
        if not initial_policies[0] or not initial_policies[1]:
            raise ValueError("both players need at least one initial policy")
        self.game = game
        self.config = config
        self.oracle = oracle or get_oracle(config.oracle, **config.oracle_kwargs)
        self.solver = get_solver(config.meta_solver)

        self.metagame = EmpiricalMetaGame(
            game,
            episodes_per_seed=config.episodes_per_seed,
            num_seeds=config.num_seeds,
            base_seed=config.seed,
            exact=config.exact_metagame,
        )
        for policy in initial_policies[0]:
            self.metagame.add_policy(0, policy)
        for policy in initial_policies[1]:
            self.metagame.add_policy(1, policy)

        self._br = (
            {0: ExactBestResponse(game, 0), 1: ExactBestResponse(game, 1)}
            if config.track_exact_exploitability
            else {}
        )
        self.history: list[IterationRecord] = []
        self.meta_strategies: tuple[np.ndarray, np.ndarray] | None = None
        self.iteration = 0
        self.converged = False
        self.oracle_episodes = 0
        self.pruned = 0

    # ------------------------------------------------------------ properties
    @property
    def populations(self) -> tuple[list[Policy], list[Policy]]:
        return self.metagame.populations

    def current_meta_strategies(self) -> tuple[np.ndarray, np.ndarray]:
        """The latest meta-strategies, padded to the current population size.

        The solver runs *before* the iteration appends its new best responses,
        so the stored vectors are one entry shorter than the populations by the
        time a run finishes. Padding with zeros is the faithful reading: those
        policies were added after the solve and carry no probability in it.
        """
        if self.meta_strategies is None:
            raise RuntimeError("run at least one iteration first")
        sigma0, sigma1 = self.meta_strategies
        return (_pad(sigma0, len(self.populations[0])),
                _pad(sigma1, len(self.populations[1])))

    # ------------------------------------------------------------------ loop
    def step(self, verbose: bool = True) -> IterationRecord:
        """Run one PSRO iteration."""
        started = time.perf_counter()
        cfg = self.config

        # 1-2. Fill in the payoff matrix for any pair we have not played yet.
        self.metagame.update(verbose=False)
        matrix = self.metagame.matrix
        pop0, pop1 = self.populations

        # 3. Meta-solver.
        sigma0, sigma1 = self.solver(matrix, **cfg.solver_kwargs)
        self.meta_strategies = (sigma0, sigma1)

        # Metrics, measured before the populations grow.
        metrics = self._measure(matrix, sigma0, sigma1, pop0, pop1)

        # 4-5. Train and add one best response per player.
        oracle_start = time.perf_counter()
        oracle_sigma0, oracle_sigma1 = sigma0, sigma1
        if cfg.single_opponent:
            rng = np.random.default_rng([cfg.seed, self.iteration, 7])
            oracle_sigma0 = _one_hot(sigma0, rng)
            oracle_sigma1 = _one_hot(sigma1, rng)
        grow = cfg.max_population is None or len(pop0) < cfg.max_population
        if grow and not self.converged:
            result0 = self.oracle.train(
                self.game, 0, pop1, oracle_sigma1, cfg.oracle_budget,
                seed=cfg.seed * 1000 + self.iteration,
            )
            result1 = self.oracle.train(
                self.game, 1, pop0, oracle_sigma0, cfg.oracle_budget,
                seed=cfg.seed * 1000 + self.iteration + 500,
            )
            result0.policy.name = f"p0_it{self.iteration + 1}"
            result1.policy.name = f"p1_it{self.iteration + 1}"
            self.metagame.add_policy(0, result0.policy)
            self.metagame.add_policy(1, result1.policy)
            oracle_returns = (result0.final_return, result1.final_return)
            self.oracle_episodes += result0.episodes + result1.episodes
        else:
            oracle_returns = (float("nan"), float("nan"))
        oracle_seconds = time.perf_counter() - oracle_start

        if cfg.prune_dominated:
            self._prune_dominated()

        record = IterationRecord(
            iteration=self.iteration,
            population0=len(pop0),
            population1=len(pop1),
            meta_strategy0=sigma0.tolist(),
            meta_strategy1=sigma1.tolist(),
            entropy0=strategy_entropy(sigma0),
            entropy1=strategy_entropy(sigma1),
            empirical_value=mixture_value(matrix, sigma0, sigma1),
            exact_value=metrics.get("value0"),
            exploitability=metrics.get("exploitability"),
            br0_value=metrics.get("br0"),
            br1_value=metrics.get("br1"),
            oracle_return0=oracle_returns[0],
            oracle_return1=oracle_returns[1],
            oracle_seconds=oracle_seconds,
            evaluation_episodes=self.metagame.episodes_simulated,
            wall_seconds=time.perf_counter() - started,
        )
        self.history.append(record)
        self.iteration += 1

        if verbose:
            expl = record.exploitability
            expl_text = f"{expl:8.4f}" if expl is not None else "     n/a"
            print(
                f"  it {record.iteration:>3}  |Pi|={record.population0:>3}  "
                f"expl={expl_text}  H(sigma0)={record.entropy0:5.3f}  "
                f"V={record.empirical_value:+.4f}  {record.wall_seconds:5.1f}s"
            )
        return record

    def _prune_dominated(self) -> None:
        """Remove weakly dominated policies from both populations.

        The meta-game must be re-estimated afterwards because the payoff matrix
        is indexed by position, so the cached entries no longer line up once a
        row or column disappears.
        """
        self.metagame.update(verbose=False)
        matrix = self.metagame.matrix
        drop0 = set(dominated_policies(matrix, 0))
        drop1 = set(dominated_policies(matrix, 1))
        # Never prune the whole population away.
        if len(drop0) >= matrix.shape[0]:
            drop0.discard(max(drop0))
        if len(drop1) >= matrix.shape[1]:
            drop1.discard(max(drop1))
        if not drop0 and not drop1:
            return
        keep0_index = [i for i in range(matrix.shape[0]) if i not in drop0]
        keep1_index = [i for i in range(matrix.shape[1]) if i not in drop1]
        keep0 = [self.populations[0][i] for i in keep0_index]
        keep1 = [self.populations[1][i] for i in keep1_index]

        # The stored meta-strategy is indexed by position, so it has to survive
        # the same surgery: keep the entries of the policies that remain and
        # renormalise. Without this the vector would outlive its population.
        if self.meta_strategies is not None:
            sigma0, sigma1 = self.meta_strategies
            self.meta_strategies = (_restrict(sigma0, keep0_index),
                                    _restrict(sigma1, keep1_index))
        fresh = EmpiricalMetaGame(
            self.game, episodes_per_seed=self.metagame.episodes_per_seed,
            num_seeds=self.metagame.num_seeds, base_seed=self.metagame.base_seed,
            exact=self.metagame.exact,
        )
        fresh.episodes_simulated = self.metagame.episodes_simulated
        for policy in keep0:
            fresh.add_policy(0, policy)
        for policy in keep1:
            fresh.add_policy(1, policy)
        self.metagame = fresh
        self.pruned = getattr(self, "pruned", 0) + len(drop0) + len(drop1)

    def _measure(self, matrix, sigma0, sigma1, pop0, pop1) -> dict:
        if not self.config.track_exact_exploitability:
            return {}
        exact_matrix = exact_payoff_matrix(self.game, pop0, pop1)
        result = exploitability(
            self.game, pop0, sigma0, pop1, sigma1,
            br0=self._br[0], br1=self._br[1], matrix=exact_matrix,
        )
        if result["exploitability"] <= self.config.exploitability_tolerance:
            self.converged = True
        return result

    def run(self, verbose: bool = True) -> list[IterationRecord]:
        """Run until the iteration budget is exhausted or the run converges."""
        if verbose:
            print(
                f"PSRO on {self.game.name}: solver={self.config.meta_solver}, "
                f"oracle={self.config.oracle}, budget={self.config.oracle_budget:,}, "
                f"seed={self.config.seed}"
            )
        while self.iteration < self.config.iterations:
            record = self.step(verbose=verbose)
            if self.converged:
                if verbose:
                    print(
                        f"  converged: exploitability "
                        f"{record.exploitability:.2e} at iteration {record.iteration}"
                    )
                break
        return self.history

    # ------------------------------------------------------------- reporting
    def final_report(self) -> dict:
        """Summary of the finished run, including population diagnostics."""
        # The last iteration appends best responses that have not been played
        # yet. Evaluate them and solve once more, so the reported solution can
        # actually use them - otherwise the final iteration's training is wasted
        # and the support is reported against a population one policy larger
        # than the strategy that was solved for.
        self.metagame.update(verbose=False)
        pop0, pop1 = self.populations
        matrix = self.metagame.matrix
        exact_matrix = exact_payoff_matrix(self.game, pop0, pop1)
        sigma0, sigma1 = self.solver(matrix, **self.config.solver_kwargs)
        self.meta_strategies = (sigma0, sigma1)
        head_to_head = head_to_head_matrix(self.game, pop0, 0)
        report = {
            "config": self.config.to_dict(),
            "iterations_run": self.iteration,
            "converged": self.converged,
            "population_sizes": [len(pop0), len(pop1)],
            "policy_names": [[p.name for p in pop0], [p.name for p in pop1]],
            "meta_strategy0": sigma0.tolist(),
            "meta_strategy1": sigma1.tolist(),
            "support0": [i for i, w in enumerate(sigma0) if w > 1e-6],
            "support1": [i for i, w in enumerate(sigma1) if w > 1e-6],
            "entropy0": strategy_entropy(sigma0),
            "entropy1": strategy_entropy(sigma1),
            "empirical_value": mixture_value(matrix, sigma0, sigma1),
            "exact_value": mixture_value(exact_matrix, sigma0, sigma1),
            "dominated0": dominated_policies(exact_matrix, 0),
            "dominated1": dominated_policies(exact_matrix, 1),
            "head_to_head": head_to_head.tolist(),
            "cycles": cycle_strength(head_to_head),
            "total_evaluation_episodes": self.metagame.episodes_simulated,
            "total_oracle_episodes": self.oracle_episodes,
        }
        if self.config.track_exact_exploitability:
            resolved = exploitability(self.game, pop0, sigma0, pop1, sigma1,
                                      self._br[0], self._br[1], exact_matrix)
            report["final_exploitability"] = resolved["exploitability"]
            report["exploitability_curve"] = (
                [r.exploitability for r in self.history] + [resolved["exploitability"]]
            )
        return report

    def history_frame(self):
        """The iteration history as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([r.to_dict() for r in self.history])

    # ---------------------------------------------------------- checkpointing
    def save(self, directory: str | Path) -> Path:
        """Write policies, payoff data, meta-strategies and history to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Policies appended by the final iteration still need their payoffs.
        self.metagame.update(verbose=False)
        payload = {
            "config": self.config.to_dict(),
            "iteration": self.iteration,
            "converged": self.converged,
            "populations": [
                [policy_to_dict(p) for p in self.populations[0]],
                [policy_to_dict(p) for p in self.populations[1]],
            ],
            "metagame": self.metagame.state_dict(),
            "history": [r.to_dict() for r in self.history],
            "meta_strategies": (
                [s.tolist() for s in self.meta_strategies]
                if self.meta_strategies is not None
                else None
            ),
        }
        (directory / "checkpoint.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8"
        )
        self.metagame.save_matrix(directory / "payoff_matrix.json")
        return directory / "checkpoint.json"

    @classmethod
    def load(
        cls, directory: str | Path, game: ZeroSumGame, oracle: Oracle | None = None
    ) -> "PSRO":
        """Rebuild a run from a checkpoint so it can continue."""
        directory = Path(directory)
        payload = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        config = PSROConfig(**payload["config"])
        populations = (
            [policy_from_dict(d) for d in payload["populations"][0]],
            [policy_from_dict(d) for d in payload["populations"][1]],
        )
        run = cls(game, config, populations, oracle=oracle)
        run.metagame.load_state_dict(payload["metagame"])
        run.iteration = payload["iteration"]
        run.converged = payload["converged"]
        run.history = [IterationRecord(**r) for r in payload["history"]]
        if payload["meta_strategies"] is not None:
            run.meta_strategies = tuple(
                np.asarray(s) for s in payload["meta_strategies"]
            )
        return run
