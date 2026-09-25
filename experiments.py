# -*- coding: utf-8 -*-
"""The five required experiments and the ablation study.

Each experiment is a function that returns a tidy :class:`pandas.DataFrame` and
caches it under ``results/``. Re-running an experiment therefore costs nothing
unless ``force=True`` is passed, which is what makes the accompanying notebook
quick to re-execute while remaining fully reproducible: delete ``results/`` and
everything is recomputed from scratch.

Runs are independent, so they are dispatched to a process pool. On a machine
with several cores this turns roughly forty minutes of work into a few minutes.

Command line
------------
``python experiments.py --all``            run everything
``python experiments.py --exp 1 2``        run selected experiments
``python experiments.py --all --quick``    smaller budgets, for a smoke test
``python experiments.py --all --force``    ignore the cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis import (dpp_expected_cardinality, effective_diversity, marginal_diversity,
                      population_diversity, unique_policy_count)
from exact import (ExactBestResponse, exact_payoff_matrix, exact_value, exploitability,
                   kuhn_behaviour_parameters, population_exploitability)
from game import (
    KUHN_GAME_VALUE,
    KuhnPoker,
    Policy,
    RandomPolicy,
    TabularPolicy,
    kuhn_always_bet,
    kuhn_heuristic,
    kuhn_nash_policies,
)
import oracles
from metagame import cycle_strength, dominated_policies
from metasolvers import strategy_entropy
from psro import PSRO, PSROConfig

RESULTS = Path(__file__).resolve().parent / "results"
CONFIGS = Path(__file__).resolve().parent / "configs"

# Defaults shared by every experiment, so comparisons are controlled.
DEFAULTS = dict(
    iterations=12,
    oracle_budget=5_000,
    episodes_per_seed=400,
    num_seeds=5,
)
QUICK = dict(iterations=6, oracle_budget=1_000, episodes_per_seed=40, num_seeds=3)

#: Rounds of best response for the single-policy self-play baseline used as the
#: "final self-play" opponent in Experiment 5.
SELF_PLAY_ROUNDS = 8

ALL_SOLVERS = (
    "self_play",
    "uniform",
    "fictitious_play",
    "nash",
    "replicator",
    "projected_replicator",
    "regret_matching",
    "rectified_nash",
)


# --------------------------------------------------------------- run spec
@dataclass
class RunSpec:
    """One PSRO run: everything needed to reproduce it, and nothing else."""

    label: str
    solver: str = "nash"
    oracle: str = "q_learning"
    seed: int = 0
    iterations: int = DEFAULTS["iterations"]
    oracle_budget: int = DEFAULTS["oracle_budget"]
    episodes_per_seed: int = DEFAULTS["episodes_per_seed"]
    num_seeds: int = DEFAULTS["num_seeds"]
    initial: str = "random"
    exact_metagame: bool = False
    solver_kwargs: dict = field(default_factory=dict)
    oracle_kwargs: dict = field(default_factory=dict)
    group: str = ""

    def to_config(self) -> PSROConfig:
        return PSROConfig(
            meta_solver=self.solver,
            oracle=self.oracle,
            oracle_kwargs=dict(self.oracle_kwargs),
            solver_kwargs=dict(self.solver_kwargs),
            iterations=self.iterations,
            oracle_budget=self.oracle_budget,
            episodes_per_seed=self.episodes_per_seed,
            num_seeds=self.num_seeds,
            seed=self.seed,
            exact_metagame=self.exact_metagame,
        )


def initial_population(game, kind: str, seed: int = 0):
    """The population PSRO starts from (Part A: random and heuristic baselines)."""
    if kind == "random":
        return ([RandomPolicy("random")], [RandomPolicy("random")])
    if kind == "baselines":
        return (
            [RandomPolicy("random"), kuhn_heuristic(0, "tight"), kuhn_always_bet(0, "aggro")],
            [RandomPolicy("random"), kuhn_heuristic(1, "tight"), kuhn_always_bet(1, "aggro")],
        )
    if kind == "heuristic":
        return ([kuhn_heuristic(0, "tight")], [kuhn_heuristic(1, "tight")])
    raise ValueError(f"unknown initial population {kind!r}")


# ------------------------------------------------------------ evaluation
def random_policy_set(game, player: int, count: int, seed: int) -> list[Policy]:
    """Held-out opponents that PSRO has never trained against."""
    rng = np.random.default_rng([seed, player, 99])
    out = []
    for i in range(count):
        probs = {inf: float(rng.uniform()) for inf in game.all_infosets(player)}
        out.append(TabularPolicy.from_bet_probabilities(game, player, probs, name=f"unseen{i}"))
    return out


def perturb(game, player: int, policy: Policy, epsilon: float, name: str) -> Policy:
    """Mix a policy with uniform noise: ``(1 - eps) * pi + eps * uniform``."""
    infosets = game.all_infosets(player)
    actions = game.legal_actions_at_infoset(infosets[0])
    n = len(actions)
    probs = {
        inf: ((1 - epsilon) * np.asarray(policy.action_probs(inf, actions)) + epsilon / n)
        for inf in infosets
    }
    return TabularPolicy({k: v / v.sum() for k, v in probs.items()}, actions, name=name)


def evaluate_agent(
    game, player: int, population, weights, opponents: Sequence[Policy]
) -> dict:
    """Score a meta-strategy against a fixed opponent set, exactly.

    ``mean`` is the average payoff over the opponent set, ``worst`` the
    minimum (how the agent does against its single worst matchup), and
    ``spread`` the standard deviation, which is the specification's
    "sensitivity to opponent selection": a robust agent scores similarly
    whoever it meets.
    """
    values = []
    for opponent in opponents:
        v = 0.0
        for policy, w in zip(population, weights):
            if w <= 0:
                continue
            v += w * (
                exact_value(game, policy, opponent)
                if player == 0
                else -exact_value(game, opponent, policy)
            )
        values.append(v)
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "worst": float(values.min()),
        "best": float(values.max()),
        "spread": float(values.std(ddof=0)),
    }


# ------------------------------------------------------------- single run
def run_one(spec: RunSpec) -> dict:
    """Execute one PSRO run and return every metric as plain data."""
    game = KuhnPoker()
    run = PSRO(game, spec.to_config(), initial_population(game, spec.initial, spec.seed))
    run.run(verbose=False)
    report = run.final_report()

    pop0, pop1 = run.populations
    sigma0 = np.asarray(report["meta_strategy0"])
    sigma1 = np.asarray(report["meta_strategy1"])

    unseen = random_policy_set(game, 1, 8, seed=1234)
    baselines = [RandomPolicy("random"), kuhn_heuristic(1, "tight"), kuhn_always_bet(1, "aggro")]
    nash_opponent = [kuhn_nash_policies(1 / 6)[1]]
    perturbed = [
        perturb(game, 1, nash_opponent[0], eps, f"nash_eps{eps}") for eps in (0.1, 0.25, 0.5)
    ]
    early = pop1[: max(1, len(pop1) // 3)]
    # The specification asks Experiment 5 to evaluate against the final
    # self-play agent, so that arm is trained here at the same seed and budget
    # and used as an opponent. This is the head-to-head the whole project is
    # about, and without it the comparison is only ever indirect.
    self_play_final = [oracles.self_play_baseline(
        game, 1, rounds=SELF_PLAY_ROUNDS, budget=max(1, spec.oracle_budget),
        seed=spec.seed).policy]

    rows = []
    for record in run.history:
        rows.append(
            dict(
                label=spec.label, group=spec.group, solver=spec.solver,
                oracle=spec.oracle, seed=spec.seed, budget=spec.oracle_budget,
                episodes_per_entry=spec.episodes_per_seed * spec.num_seeds,
                iteration=record.iteration + 1,
                population=record.population0,
                exploitability=record.exploitability,
                entropy0=record.entropy0, entropy1=record.entropy1,
                value=record.exact_value,
                empirical_value=record.empirical_value,
                oracle_seconds=record.oracle_seconds,
                wall_seconds=record.wall_seconds,
                evaluation_episodes=record.evaluation_episodes,
            )
        )

    diversity = population_diversity(game, 0, pop0, "response",
                                     opponents=baselines + unseen, weights=sigma0)

    # Solver-independent scoring. final_exploitability grades each meta-solver
    # by the mixture it happens to output, which makes a comparison between
    # solvers partly a comparison of solution concepts; population
    # exploitability grades the population itself and is the same question for
    # every solver.
    result_matrix = exact_payoff_matrix(game, pop0, pop1)
    population_gap = population_exploitability(game, pop0, pop1)
    # Diversity as the open-ended-learning literature measures it.
    head_to_head = np.asarray(report["head_to_head"])
    uniqueness = unique_policy_count(game, 0, pop0)
    parameters0 = kuhn_behaviour_parameters(game, pop0, sigma0, 0)
    parameters1 = kuhn_behaviour_parameters(game, pop1, sigma1, 1)
    # Double oracle is guaranteed to terminate at an equilibrium but not to
    # improve on the way there; McAleer et al. (ICLR 2024) show constructions
    # where exploitability rises on every iteration but the last. Counting the
    # rises turns that into something measurable rather than assumed.
    curve = [float(v) for v in (report.get("exploitability_curve") or []) if v is not None]
    rises = [b - a for a, b in zip(curve, curve[1:]) if b > a + 1e-12]

    summary = dict(
        label=spec.label, group=spec.group, solver=spec.solver, oracle=spec.oracle,
        seed=spec.seed, budget=spec.oracle_budget, iterations=spec.iterations,
        initial=spec.initial,
        episodes_per_entry=spec.episodes_per_seed * spec.num_seeds,
        final_exploitability=report.get("final_exploitability"),
        final_value=report["exact_value"],
        value_error=abs(report["exact_value"] - KUHN_GAME_VALUE),
        population=report["population_sizes"][0],
        support=len(report["support0"]),
        entropy=report["entropy0"],
        cycles=report["cycles"]["num_3cycles"],
        cyclic_fraction=report["cycles"]["cyclic_triple_fraction"],
        dominated=len(report["dominated0"]),
        diversity_mean=diversity["mean"],
        diversity_weighted=diversity["weighted_mean"],
        effective_policies=diversity["effective_policies"],
        population_exploitability=population_gap["population_exploitability"],
        value_upper=population_gap["value_upper"],
        value_lower=population_gap["value_lower"],
        unique_policies=uniqueness["unique"],
        unique_fraction=uniqueness["fraction"],
        effective_diversity=effective_diversity(head_to_head, sigma0),
        dpp_cardinality=dpp_expected_cardinality(np.asarray(result_matrix)),
        kuhn_alpha=parameters0["alpha"],
        kuhn_beta=parameters0["beta"],
        kuhn_gamma=parameters0["gamma"],
        kuhn_residual0=parameters0["residual"],
        kuhn_xi=parameters1["xi"],
        kuhn_eta=parameters1["eta"],
        kuhn_residual1=parameters1["residual"],
        exploitability_rises=len(rises),
        largest_rise=max(rises) if rises else 0.0,
        total_eval_episodes=report["total_evaluation_episodes"],
        total_oracle_episodes=report["total_oracle_episodes"],
        wall_seconds=sum(r.wall_seconds for r in run.history),
    )
    for name, opponents in (
        ("vs_population", pop1), ("vs_baselines", baselines), ("vs_unseen", unseen),
        ("vs_nash", nash_opponent), ("vs_perturbed", perturbed),
        ("vs_early", early), ("vs_final_selfplay", self_play_final),
    ):
        scores = evaluate_agent(game, 0, pop0, sigma0, opponents)
        summary[f"{name}_mean"] = scores["mean"]
        summary[f"{name}_worst"] = scores["worst"]
        summary[f"{name}_spread"] = scores["spread"]

    return {
        "spec": asdict(spec),
        "rows": rows,
        "summary": summary,
        "marginal_diversity": marginal_diversity(
            game, 0, pop0, "response", opponents=baselines + unseen
        ),
        "payoff_matrix": result_matrix.tolist(),
        "estimated_payoff_matrix": run.metagame.matrix.tolist(),
        "payoff_standard_error": run.metagame.standard_error.tolist(),
        "head_to_head": report["head_to_head"],
        "policy_names": report["policy_names"],
        "meta_strategy0": sigma0.tolist(),
        "meta_strategy1": sigma1.tolist(),
        "support0": report["support0"],
        "exploitability_curve": report.get("exploitability_curve"),
    }


# ------------------------------------------------------------- dispatching
def run_many(specs: Sequence[RunSpec], workers: int | None = None, verbose: bool = True):
    """Run independent PSRO runs, in parallel when more than one core is free."""
    workers = workers or max(1, min(len(specs), (os.cpu_count() or 2) - 2))
    results = []
    if workers == 1 or len(specs) == 1:
        for i, spec in enumerate(specs, 1):
            if verbose:
                print(f"  [{i}/{len(specs)}] {spec.label}", flush=True)
            results.append(run_one(spec))
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, spec): spec for spec in specs}
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if verbose:
                print(f"  [{i}/{len(specs)}] {futures[future].label}", flush=True)
    order = {spec.label: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: order[r["spec"]["label"]])
    return results


def _cached(name: str, force: bool):
    path = RESULTS / f"{name}.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    return None


def _store(name: str, iterations: pd.DataFrame, summary: pd.DataFrame, extra: dict | None = None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    iterations.to_csv(RESULTS / f"{name}.csv", index=False)
    summary.to_csv(RESULTS / f"{name}_summary.csv", index=False)
    if extra:
        (RESULTS / f"{name}_extra.json").write_text(
            json.dumps(extra, indent=1), encoding="utf-8"
        )


def _save_config(name: str, specs: Sequence[RunSpec]) -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / f"{name}.json").write_text(
        json.dumps([asdict(s) for s in specs], indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


#: Saved for every experiment, because section 9 of the specification asks for
#: "saved payoff matrices and meta-strategies" - not only for the one experiment
#: that happens to analyse them.
PERSISTED = ("payoff_matrix", "head_to_head", "meta_strategy0", "meta_strategy1",
             "policy_names", "support0", "exploitability_curve",
             "marginal_diversity")


def _collect(name: str, specs: Sequence[RunSpec], force: bool, extra_keys=PERSISTED):
    """Run (or load) a set of specs and persist the tidy tables."""
    cached = _cached(name, force)
    if cached is not None:
        summary = pd.read_csv(RESULTS / f"{name}_summary.csv")
        return cached, summary
    _save_config(name, specs)
    results = run_many(specs)
    iterations = pd.DataFrame([row for r in results for row in r["rows"]])
    summary = pd.DataFrame([r["summary"] for r in results])
    extra = {
        r["spec"]["label"]: {k: r[k] for k in extra_keys} for r in results
    } if extra_keys else None
    _store(name, iterations, summary, extra)
    return iterations, summary


# ----------------------------------------------------------- experiment 1
def experiment_1(seeds=range(5), force=False, **overrides):
    """Self-play versus PSRO under an identical training budget.

    The two arms differ only in the meta-solver, so any difference in
    exploitability, worst-case return or generalisation is attributable to
    maintaining and solving the population.
    """
    cfg = {**DEFAULTS, **overrides}
    specs = [
        RunSpec(label=f"{solver}_s{seed}", solver=solver, seed=seed, group=solver, **cfg)
        for solver in ("self_play", "nash")
        for seed in seeds
    ]
    return _collect("experiment1_selfplay_vs_psro", specs, force,
                    extra_keys=("exploitability_curve",))


# ----------------------------------------------------------- experiment 2
def experiment_2(seeds=range(5), solvers=ALL_SOLVERS, force=False, **overrides):
    """Comparison of all seven meta-strategy solvers."""
    cfg = {**DEFAULTS, **overrides}
    specs = [
        RunSpec(label=f"{solver}_s{seed}", solver=solver, seed=seed, group=solver, **cfg)
        for solver in solvers
        for seed in seeds
    ]
    return _collect("experiment2_metasolvers", specs, force)


# ----------------------------------------------------------- experiment 3
def experiment_3(seed=0, solver="nash", force=False, **overrides):
    """Structure of the learned population: heat map, dominance, cycles, diversity."""
    name = "experiment3_population"
    path = RESULTS / f"{name}_extra.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    cfg = {**DEFAULTS, **overrides}
    spec = RunSpec(label=f"{solver}_s{seed}", solver=solver, seed=seed, group=solver, **cfg)
    _save_config(name, [spec])
    result = run_one(spec)

    matrix = np.asarray(result["payoff_matrix"])
    # Cycles must be read off the antisymmetric head-to-head matrix, never off
    # the bipartite meta-game matrix: entry (i, j) of the latter compares the
    # *seat-0* policy i with the *seat-1* policy j, so a "cycle" in it would
    # relate four different policies rather than three.
    payload = {
        "policy_names": result["policy_names"],
        "payoff_matrix": result["payoff_matrix"],
        "estimated_payoff_matrix": result["estimated_payoff_matrix"],
        "payoff_standard_error": result["payoff_standard_error"],
        "head_to_head": result["head_to_head"],
        "meta_strategy0": result["meta_strategy0"],
        "meta_strategy1": result["meta_strategy1"],
        "support0": result["support0"],
        "marginal_diversity": result["marginal_diversity"],
        "dominated0": dominated_policies(matrix, 0),
        "dominated1": dominated_policies(matrix, 1),
        "cycles": cycle_strength(np.asarray(result["head_to_head"])),
        "exploitability_curve": result["exploitability_curve"],
        "summary": result["summary"],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    pd.DataFrame(result["rows"]).to_csv(RESULTS / f"{name}.csv", index=False)
    return payload


# ----------------------------------------------------------- experiment 4
def experiment_4(budgets=(500, 2_000, 10_000, 50_000), seeds=range(5),
                 force=False, **overrides):
    """Quality of the approximate oracle: how much does the best response matter?

    The exact oracle is included as the perfect-oracle limit, which turns this
    into a comparison against the double-oracle algorithm itself.
    """
    cfg = {**DEFAULTS, **overrides}
    cfg.pop("oracle_budget", None)
    specs = [
        RunSpec(label=f"q{budget}_s{seed}", solver="nash", oracle="q_learning",
                oracle_budget=budget, seed=seed, group=f"q_learning_{budget}", **cfg)
        for budget in budgets
        for seed in seeds
    ] + [
        RunSpec(label=f"exact_s{seed}", solver="nash", oracle="exact",
                oracle_budget=0, seed=seed, group="exact_oracle", **cfg)
        for seed in seeds
    ] + [
        # Exact oracle *and* exact meta-game payoffs. This is no longer an
        # approximation of anything: it is the double-oracle algorithm itself,
        # so McMahan, Gordon and Blum's convergence theorem applies literally.
        # Comparing the three groups separates the two ways PSRO departs from
        # that theorem - an approximate best response and a sampled payoff
        # matrix - instead of lumping them together as "approximation error".
        RunSpec(label=f"double_oracle_s{seed}", solver="nash", oracle="exact",
                oracle_budget=0, seed=seed, group="double_oracle",
                exact_metagame=True, **cfg)
        for seed in seeds
    ]
    return _collect("experiment4_oracle_quality", specs, force)


# ----------------------------------------------------------- experiment 5
def experiment_5(seeds=range(5), solvers=("self_play", "uniform", "nash",
                                          "projected_replicator"),
                 force=False, **overrides):
    """Generalisation to opponents the agent never trained against.

    Uses the same runs as the other experiments but reports the held-out
    columns: baselines, unseen random policies, perturbed equilibria, and the
    analytic Nash opponent.
    """
    cfg = {**DEFAULTS, **overrides}
    specs = [
        RunSpec(label=f"{solver}_s{seed}", solver=solver, seed=seed, group=solver, **cfg)
        for solver in solvers
        for seed in seeds
    ]
    return _collect("experiment5_generalisation", specs, force)


# -------------------------------------------------------------- ablations
def ablations(seeds=range(5), force=False, **overrides):
    """Which parts of PSRO actually do the work.

    ============================  ==================================================
    Ablation                      What it removes
    ============================  ==================================================
    ``full``                      nothing (the reference arm)
    ``no_population``             the population: train only against the latest policy
    ``uniform_meta``              the solved meta-strategy: play everything equally
    ``few_eval_episodes``         payoff accuracy: 10x fewer evaluation episodes
    ``weak_oracle``               oracle quality: a tenth of the training budget
    ``heuristic_start``           the initial population: start from a heuristic
    ============================  ==================================================
    """
    cfg = {**DEFAULTS, **overrides}
    variants = [
        ("full", dict(solver="nash")),
        ("no_population", dict(solver="self_play")),
        ("uniform_meta", dict(solver="uniform")),
        ("few_eval_episodes", dict(solver="nash",
                                   episodes_per_seed=max(1, cfg["episodes_per_seed"] // 10))),
        ("weak_oracle", dict(solver="nash", oracle_budget=max(1, cfg["oracle_budget"] // 10))),
        ("heuristic_start", dict(solver="nash", initial="heuristic")),
    ]
    specs = []
    for name, changes in variants:
        merged = {**cfg, **changes}
        solver = merged.pop("solver")
        initial = merged.pop("initial", "random")
        for seed in seeds:
            specs.append(
                RunSpec(label=f"{name}_s{seed}", solver=solver, seed=seed,
                        group=name, initial=initial, **merged)
            )
    return _collect("ablations", specs, force)


# ------------------------------------------------------------------- CLI
EXPERIMENTS = {
    "1": experiment_1, "2": experiment_2, "3": experiment_3,
    "4": experiment_4, "5": experiment_5, "ablations": ablations,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="*", default=[], help="which experiments to run")
    parser.add_argument("--all", action="store_true", help="run everything")
    parser.add_argument("--quick", action="store_true", help="small budgets, for testing")
    parser.add_argument("--force", action="store_true", help="ignore cached results")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    overrides = QUICK if args.quick else {}
    names = list(EXPERIMENTS) if args.all else args.exp
    if not names:
        parser.error("pass --all or --exp <names>")

    for name in names:
        print(f"\n=== experiment {name} ===", flush=True)
        result = EXPERIMENTS[name](force=args.force, **overrides)
        if isinstance(result, tuple):
            _iterations, summary = result
            columns = [c for c in ("group", "seed", "final_exploitability",
                                   "vs_unseen_worst", "support", "wall_seconds")
                       if c in summary]
            print(summary.groupby("group")[
                [c for c in columns if c not in ("group", "seed")]
            ].mean().round(4).to_string())
    print(f"\nresults written to {RESULTS}")


if __name__ == "__main__":
    main()
