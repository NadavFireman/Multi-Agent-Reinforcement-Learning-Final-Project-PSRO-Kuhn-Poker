# -*- coding: utf-8 -*-
"""Behavioural diversity measures and the figures built from them.

The two halves belong together: the diversity measures are what the population
figures visualise, and neither is used anywhere except when analysing a
finished run.

Diversity (evaluation measure, section 7 of the specification)
--------------------------------------------------------------
Three complementary measures, because they answer different questions and can
disagree in an informative way:

1. **Action-distribution distance** - how differently two policies *act* at the
   same information sets.
2. **State-visitation distance** - how differently they *steer the game*,
   measured on exact reach probabilities. A policy can act differently at states
   it never reaches, which measure 1 counts and this one does not.
3. **Response-profile distance** - how differently they *score* against a fixed
   opponent set. This is the strategically meaningful notion: two policies that
   behave differently but win and lose against exactly the same opponents add
   nothing to a population.

All are computed exactly by tree traversal. Divergences use base-2 logarithms,
so a Jensen-Shannon divergence lies in ``[0, 1]``.

Figures
-------
Curves aggregated over seeds are drawn as a mean with a shaded band of plus or
minus one standard error, so the reader can see at a glance whether a gap
survives seed noise.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exact import exact_value
from game import Policy, ZeroSumGame


_EPS = 1e-12


# ------------------------------------------------------------------ helpers
def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two distributions, in bits.

    Symmetric, always finite, and bounded in ``[0, 1]``: zero for identical
    distributions and one for distributions with disjoint support.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), _EPS)
    q = q / max(q.sum(), _EPS)
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _kl(p: np.ndarray, m: np.ndarray) -> float:
    mask = p > _EPS
    return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(m[mask], _EPS))))


# --------------------------------------------------- 1. action distributions
def action_js_divergence(
    game: ZeroSumGame, player: int, policy_a: Policy, policy_b: Policy
) -> float:
    """Mean Jensen-Shannon divergence between two policies' action distributions.

    Averaged over every information set of ``player``, unweighted: each
    information set counts once regardless of how often it is reached. Use
    :func:`visitation_js_divergence` for the reach-weighted view.
    """
    infosets = game.all_infosets(player)
    actions = game.legal_actions_at_infoset(infosets[0])
    return float(
        np.mean(
            [
                jensen_shannon(
                    policy_a.action_probs(i, actions), policy_b.action_probs(i, actions)
                )
                for i in infosets
            ]
        )
    )


# --------------------------------------------------- 2. state visitation
def visitation_distribution(
    game: ZeroSumGame, player: int, policy: Policy, opponent: Policy
) -> np.ndarray:
    """Exact probability of reaching each of ``player``'s information sets.

    Computed by traversing the whole tree and accumulating reach probabilities,
    so it is exact rather than estimated. The vector is normalised to sum to
    one; note that a player may act more than once per episode, so the raw
    reach probabilities need not.
    """
    infosets = game.all_infosets(player)
    index = {name: k for k, name in enumerate(infosets)}
    reach = np.zeros(len(infosets))
    policies = (policy, opponent) if player == 0 else (opponent, policy)

    for chance, chance_prob in game.chance_outcomes():
        _accumulate(game, chance, game.initial_history(), chance_prob,
                    policies, player, index, reach)
    total = reach.sum()
    return reach / total if total > _EPS else np.full(len(infosets), 1.0 / len(infosets))


def _accumulate(game, chance, history, probability, policies, player, index, reach):
    if probability <= _EPS or game.is_terminal(history):
        return
    acting = game.current_player(history)
    actions = game.legal_actions(history)
    infoset = game.infoset(chance, history, acting)
    if acting == player:
        reach[index[infoset]] += probability
    action_probs = policies[acting].action_probs(infoset, actions)
    for action, p in zip(actions, action_probs):
        if p > 0.0:
            _accumulate(game, chance, history + (action,), probability * p,
                        policies, player, index, reach)


def visitation_js_divergence(
    game: ZeroSumGame,
    player: int,
    policy_a: Policy,
    policy_b: Policy,
    opponent: Policy,
) -> float:
    """How differently two policies steer the game, against a shared opponent."""
    return jensen_shannon(
        visitation_distribution(game, player, policy_a, opponent),
        visitation_distribution(game, player, policy_b, opponent),
    )


# ------------------------------------------------- 3. response profiles
def response_profile(
    game: ZeroSumGame, player: int, policy: Policy, opponents: Sequence[Policy]
) -> np.ndarray:
    """Exact payoff of ``policy`` against each opponent, from its own viewpoint."""
    if player == 0:
        return np.array([exact_value(game, policy, o) for o in opponents])
    return np.array([-exact_value(game, o, policy) for o in opponents])


def response_profile_distance(
    game: ZeroSumGame,
    player: int,
    policy_a: Policy,
    policy_b: Policy,
    opponents: Sequence[Policy],
) -> float:
    """Root-mean-square difference in payoffs against a fixed opponent set.

    Zero means the two policies are strategically interchangeable against those
    opponents, however differently they may behave.
    """
    a = response_profile(game, player, policy_a, opponents)
    b = response_profile(game, player, policy_b, opponents)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ------------------------------------------------------ population summaries
def pairwise_matrix(
    game: ZeroSumGame,
    player: int,
    population: Sequence[Policy],
    measure: str = "action",
    opponents: Sequence[Policy] | None = None,
) -> np.ndarray:
    """Symmetric matrix of pairwise distances within a population.

    Parameters
    ----------
    measure:
        ``"action"``, ``"visitation"`` or ``"response"``.
    opponents:
        Required for ``"visitation"`` (its first element is used as the shared
        opponent) and for ``"response"`` (the whole fixed opponent set).
    """
    n = len(population)
    out = np.zeros((n, n))
    if measure in ("visitation", "response") and not opponents:
        raise ValueError(f"measure {measure!r} needs an opponent set")
    for i in range(n):
        for j in range(i + 1, n):
            if measure == "action":
                d = action_js_divergence(game, player, population[i], population[j])
            elif measure == "visitation":
                d = visitation_js_divergence(
                    game, player, population[i], population[j], opponents[0]
                )
            elif measure == "response":
                d = response_profile_distance(
                    game, player, population[i], population[j], opponents
                )
            else:
                raise ValueError(f"unknown measure {measure!r}")
            out[i, j] = out[j, i] = d
    return out


def population_diversity(
    game: ZeroSumGame,
    player: int,
    population: Sequence[Policy],
    measure: str = "action",
    opponents: Sequence[Policy] | None = None,
    weights: Sequence[float] | None = None,
) -> dict:
    """Summary statistics of how diverse a population is.

    ``weights`` (a meta-strategy) turns the mean into an *effective* diversity
    that only counts the policies the solution actually plays: a population of
    twenty policies whose equilibrium uses three of them is not diverse in any
    meaningful sense.
    """
    n = len(population)
    if n < 2:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "weighted_mean": 0.0,
                "effective_policies": float(n)}
    distances = pairwise_matrix(game, player, population, measure, opponents)
    upper = distances[np.triu_indices(n, k=1)]
    summary = {
        "mean": float(upper.mean()),
        "max": float(upper.max()),
        "min": float(upper.min()),
    }
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / max(w.sum(), _EPS)
        outer = np.outer(w, w)
        np.fill_diagonal(outer, 0.0)
        denominator = outer.sum()
        summary["weighted_mean"] = (
            float((distances * outer).sum() / denominator) if denominator > _EPS else 0.0
        )
        positive = w[w > 1e-6]
        summary["effective_policies"] = (
            float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else 0.0
        )
    else:
        summary["weighted_mean"] = summary["mean"]
        summary["effective_policies"] = float(n)
    return summary


def marginal_diversity(
    game: ZeroSumGame,
    player: int,
    population: Sequence[Policy],
    measure: str = "action",
    opponents: Sequence[Policy] | None = None,
) -> list[float]:
    """How much genuinely new behaviour each policy brought when it was added.

    Entry ``k`` is the distance from policy ``k`` to its *nearest* predecessor.
    A value near zero means PSRO rediscovered something it already had, which
    is the signature of a stalled run; a sustained positive value means the
    population keeps expanding into new strategic territory.

    This answers the specification's question "determine whether new policies
    add strategic diversity" directly, one number per iteration.
    """
    distances = pairwise_matrix(game, player, population, measure, opponents)
    out = [0.0]
    for k in range(1, len(population)):
        out.append(float(distances[k, :k].min()))
    return out


#: A stable colour per meta-solver, so the same method looks the same in every
#: figure of the report.
SOLVER_COLOURS = {
    "self_play": "#d62728",
    "uniform": "#7f7f7f",
    "fictitious_play": "#9467bd",
    "nash": "#1f77b4",
    "replicator": "#2ca02c",
    "projected_replicator": "#17becf",
    "regret_matching": "#ff7f0e",
}
SOLVER_LABELS = {
    "self_play": "Self-play (latest only)",
    "uniform": "Uniform mixture",
    "fictitious_play": "Fictitious play",
    "nash": "Nash (double oracle)",
    "replicator": "Replicator dynamics",
    "projected_replicator": "Projected replicator",
    "regret_matching": "Regret matching",
}


def _mean_and_error(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    grouped = frame.groupby(x)[y]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["stderr"] = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    return out


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


# ------------------------------------------------------------------- curves
def plot_metric_by_iteration(
    iterations: pd.DataFrame,
    metric: str = "exploitability",
    groups: Sequence[str] | None = None,
    logy: bool = True,
    title: str | None = None,
    ylabel: str | None = None,
    ax=None,
):
    """One curve per group, averaged over seeds, with a standard-error band."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 5))
    groups = groups if groups is not None else sorted(iterations["group"].unique())
    for group in groups:
        subset = iterations[iterations["group"] == group]
        if subset.empty:
            continue
        stats = _mean_and_error(subset, "iteration", metric)
        colour = SOLVER_COLOURS.get(group)
        label = SOLVER_LABELS.get(group, group)
        ax.plot(stats["iteration"], stats["mean"], marker="o", markersize=4,
                color=colour, label=label)
        ax.fill_between(stats["iteration"], stats["mean"] - stats["stderr"],
                        stats["mean"] + stats["stderr"], alpha=0.18, color=colour)
    if logy and (iterations[metric] > 0).all():
        ax.set_yscale("log")
    _style(ax, "PSRO iteration", ylabel or metric.replace("_", " "),
           title or f"{metric.replace('_', ' ').capitalize()} by iteration")
    ax.legend(fontsize=9)
    return ax.figure


def plot_exploitability(iterations: pd.DataFrame, **kwargs):
    """Exploitability by iteration: the project's headline figure."""
    return plot_metric_by_iteration(
        iterations, "exploitability",
        title="Exact exploitability by PSRO iteration",
        ylabel="exploitability (lower is better)", **kwargs
    )


# --------------------------------------------------------------- heat map
def plot_payoff_heatmap(
    matrix, row_names: Sequence[str], column_names: Sequence[str],
    meta_strategy=None, title: str = "Empirical payoff matrix", ax=None,
):
    """Heat map of the meta-game, marking the policies the solution actually plays.

    A diverging colour scale centred on zero is deliberate: in a zero-sum game
    the sign of an entry is the whole story, so the eye should find the sign
    boundary immediately.
    """
    matrix = np.asarray(matrix)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(1.0 + 0.55 * matrix.shape[1],
                                         1.0 + 0.5 * matrix.shape[0]))
    limit = float(np.abs(matrix).max()) or 1.0
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(column_names)))
    ax.set_yticks(range(len(row_names)))
    ax.set_xticklabels(column_names, rotation=90, fontsize=8)
    ax.set_yticklabels(row_names, fontsize=8)
    if meta_strategy is not None:
        for index, weight in enumerate(meta_strategy):
            if weight > 1e-6 and index < len(row_names):
                ax.get_yticklabels()[index].set_color("#b8860b")
                ax.get_yticklabels()[index].set_fontweight("bold")
    ax.set_xlabel("player 1 policy")
    ax.set_ylabel("player 0 policy")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046, label="payoff to player 0")
    return ax.figure


# ------------------------------------------------------------------- bars
def plot_summary_bars(
    summary: pd.DataFrame, metric: str, groups: Sequence[str] | None = None,
    title: str | None = None, ylabel: str | None = None, ax=None,
):
    """Final value of one metric per group, with standard-error bars over seeds."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 4.5))
    groups = groups if groups is not None else sorted(summary["group"].unique())
    means, errors, labels, colours = [], [], [], []
    for group in groups:
        values = summary.loc[summary["group"] == group, metric].dropna()
        if values.empty:
            continue
        means.append(values.mean())
        errors.append(values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0)
        labels.append(SOLVER_LABELS.get(group, group))
        colours.append(SOLVER_COLOURS.get(group, "#4c72b0"))
    positions = np.arange(len(means))
    ax.bar(positions, means, yerr=errors, capsize=4, color=colours, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    _style(ax, "", ylabel or metric.replace("_", " "),
           title or metric.replace("_", " ").capitalize())
    return ax.figure


def plot_tradeoff(
    summary: pd.DataFrame, x: str = "wall_seconds", y: str = "final_exploitability",
    title: str = "Cost versus quality", ax=None,
):
    """Scatter of a cost measure against a quality measure, one point per group."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 5))
    for group, subset in summary.groupby("group"):
        ax.errorbar(
            subset[x].mean(), subset[y].mean(),
            xerr=subset[x].std(ddof=1) / np.sqrt(len(subset)) if len(subset) > 1 else 0,
            yerr=subset[y].std(ddof=1) / np.sqrt(len(subset)) if len(subset) > 1 else 0,
            fmt="o", markersize=9, capsize=3,
            color=SOLVER_COLOURS.get(group), label=SOLVER_LABELS.get(group, group),
        )
    _style(ax, x.replace("_", " "), y.replace("_", " "), title)
    ax.legend(fontsize=8)
    return ax.figure


# ------------------------------------------------------------- diversity
def plot_marginal_diversity(values: Sequence[float], title: str | None = None, ax=None):
    """How much new behaviour each added policy contributed.

    Bars near zero mean PSRO rediscovered a policy it already had, which is the
    signature of a population that has stopped growing usefully.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 4))
    values = list(values)
    ax.bar(range(len(values)), values, color="#4c72b0", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    _style(ax, "policy index (order of addition)", "distance to nearest predecessor",
           title or "Marginal strategic diversity of each new policy")
    return ax.figure


def plot_generalisation(
    summary: pd.DataFrame,
    columns=("vs_baselines_mean", "vs_unseen_mean", "vs_perturbed_mean", "vs_nash_mean"),
    groups: Sequence[str] | None = None, worst_case: bool = False, ax=None,
):
    """Grouped bars comparing performance across held-out opponent families."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(9, 5))
    groups = groups if groups is not None else sorted(summary["group"].unique())
    columns = [c.replace("_mean", "_worst") for c in columns] if worst_case else list(columns)
    width = 0.8 / max(len(groups), 1)
    positions = np.arange(len(columns))
    for index, group in enumerate(groups):
        subset = summary[summary["group"] == group]
        means = [subset[c].mean() for c in columns]
        errors = [
            subset[c].std(ddof=1) / np.sqrt(len(subset)) if len(subset) > 1 else 0.0
            for c in columns
        ]
        ax.bar(positions + index * width, means, width, yerr=errors, capsize=3,
               label=SOLVER_LABELS.get(group, group),
               color=SOLVER_COLOURS.get(group), alpha=0.85)
    ax.set_xticks(positions + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(
        [c.replace("vs_", "").replace("_mean", "").replace("_worst", "") for c in columns]
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    _style(ax, "held-out opponent family",
           "worst-case payoff" if worst_case else "mean payoff",
           "Worst-case generalisation" if worst_case else "Generalisation to unseen opponents")
    ax.legend(fontsize=9)
    return ax.figure


# ------------------------------------------------------- behavioural evidence
#: Card and action names used when rendering hands for a human reader.
CARD_LABELS = {0: "J", 1: "Q", 2: "K"}
ACTION_LABELS = {"p": "check/fold", "b": "bet/call"}


def behavioural_profile(
    game: ZeroSumGame,
    player: int,
    population: Sequence[Policy],
    weights: Sequence[float],
    opponent: Policy,
    episodes: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Action counts per information set when a meta-strategy actually plays.

    A PSRO solution is a *mixture* over policies, so it has no single behavioural
    strategy table to read off. What it does have is the behaviour it produces at
    the table, which is what this measures: play ``episodes`` hands with the
    mixture and count how often each action is taken at each information set.

    This is the evidence the specification asks for when it says the conclusion
    must rest on behavioural evidence and not only on training reward.
    """
    from game import MixturePolicy, play_episode_trajectory

    mixture = MixturePolicy(population, weights, name="learned")
    rng = np.random.default_rng([seed, player])
    actions = game.legal_actions_at_infoset(game.all_infosets(player)[0])
    index = {action: i for i, action in enumerate(actions)}
    counts = {infoset: np.zeros(len(actions)) for infoset in game.all_infosets(player)}
    policies = (mixture, opponent) if player == 0 else (opponent, mixture)
    for _ in range(episodes):
        trajectory, _value = play_episode_trajectory(game, policies, rng, player)
        for infoset, action in trajectory:
            counts[infoset][index[action]] += 1.0
    return counts


def classify_hand(chance, history: tuple) -> str:
    """Name the strategic pattern a finished Kuhn Poker hand exhibits."""
    stronger = chance[0] > chance[1]
    if history[0] == "b":
        return "value bet" if stronger else "bluff (bet the weaker card)"
    if len(history) >= 3 and history[2] == "b":
        return "check-call" if stronger else "crying call"
    if len(history) >= 3 and history[2] == "p":
        return "check-fold to a bet"
    return "check-check showdown"


def play_example_hands(
    game: ZeroSumGame,
    population: Sequence[Policy],
    weights: Sequence[float],
    opponent: Policy,
    hands: int,
    seed: int,
):
    """Play hands with a meta-strategy and return one row per hand.

    Each row records both players' cards, the action sequence in words, the
    payoff, and the strategic pattern the hand exhibits, so a reader can see
    what the learned solution actually does rather than only how it scores.
    """
    import pandas as pd

    from game import MixturePolicy

    learned = MixturePolicy(population, weights, name="learned")
    rng = np.random.default_rng([seed, 77])
    outcomes = game.chance_outcomes()
    probabilities = np.fromiter((p for _c, p in outcomes), dtype=float,
                                count=len(outcomes))
    rows = []
    for _ in range(hands):
        for policy in (learned, opponent):
            policy.begin_episode(rng)
        chance = outcomes[rng.choice(len(outcomes), p=probabilities)][0]
        history, log = game.initial_history(), []
        while not game.is_terminal(history):
            acting = game.current_player(history)
            legal = game.legal_actions(history)
            infoset = game.infoset(chance, history, acting)
            policy = learned if acting == 0 else opponent
            action = policy.act(infoset, legal, rng)
            log.append(f"P{acting} {ACTION_LABELS[action]}")
            history = history + (action,)
        rows.append({
            "P0 card": CARD_LABELS[chance[0]],
            "P1 card": CARD_LABELS[chance[1]],
            "sequence": " -> ".join(log),
            "payoff to P0": game.terminal_value(chance, history),
            "behaviour": classify_hand(chance, history),
        })
    return pd.DataFrame(rows)


# ================================================================ population
# Three diversity measures taken directly from the open-ended-learning
# literature, so the population can also be scored the way that literature
# scores it and not only by the bespoke measures above.

def unique_policy_count(game, player: int, policies: Sequence) -> dict:
    """How many *behaviourally distinct* policies the population contains.

    The simplest possible diversity measure, and the one Muller et al. (ICLR
    2020) chose for Kuhn Poker: two policies are the same exactly when they act
    identically at every information set. In a finite game that is decidable,
    exact, and free of any tuning constant, and they report it tracking
    exploitability closely.

    It is the right primary measure here for a further reason: it is the
    measure that detects the specific failure mode of PSRO_rN, which is not
    that the population is badly spread but that it stops growing at all.
    """
    signatures, first_seen = [], {}
    for index, policy in enumerate(policies):
        probs = policy.bet_probabilities(game, player)
        key = tuple(round(probs[i], 9) for i in sorted(probs))
        signatures.append(key)
        first_seen.setdefault(key, index)
    unique = len(set(signatures))
    return {
        "unique": unique,
        "total": len(policies),
        "fraction": unique / max(1, len(policies)),
        "duplicates": [i for i, k in enumerate(signatures) if first_seen[k] != i],
    }


def effective_diversity(head_to_head: np.ndarray, weights: Sequence[float]) -> float:
    """Effective diversity of Balduzzi et al. (ICML 2019), d(P) = p' max(A, 0) p.

    ``A`` must be the antisymmetric head-to-head matrix and ``p`` a Nash
    mixture over the population. The rectifier is not cosmetic: for an
    antisymmetric ``A`` the plain quadratic form ``p @ A @ p`` is identically
    zero, so without it the measure would carry no information at all. What
    survives is the Nash-weighted margin by which the *effective* agents, the
    ones with support, beat one another. A population with a single dominant
    policy scores zero.

    One caveat is worth stating, because the source is not self-consistent
    about it: their Definition 4 says "a Nash equilibrium", while the sentence
    immediately after it says "the maximum entropy Nash". Nash mixtures need
    not be unique, so the measure is well defined only once a selection rule is
    fixed. The weights passed in here come from this project's linear program,
    and the number returned is relative to that choice.
    """
    matrix = np.asarray(head_to_head, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("effective_diversity expects a square head-to-head matrix")
    if matrix.shape[0] != weights.size:
        raise ValueError("weights must have one entry per policy")
    return float(weights @ np.maximum(matrix, 0.0) @ weights)


def dpp_expected_cardinality(payoff_matrix: np.ndarray) -> float:
    """Expected cardinality of a determinantal point process over the population.

    Perez-Nieves et al. (ICML 2021) treat the payoff matrix ``M`` as a feature
    table, row ``i`` being policy ``i``'s response profile, build the Gram
    kernel ``L = M @ M.T``, and score the population by
    ``E|Y| = trace(I - inv(L + I)) = sum_i lam_i / (1 + lam_i)``.

    Their argument for it over the determinant is redundancy: ``det(L)``
    collapses to zero the moment two policies coincide, whereas the expected
    cardinality stays finite and simply stops growing. It also differs from
    effective diversity in what it counts, since every policy contributes and
    not only those inside the Nash support, which is why the two measures can
    rank the same pair of populations differently.
    """
    matrix = np.asarray(payoff_matrix, dtype=float)
    eigenvalues = np.clip(np.linalg.eigvalsh(matrix @ matrix.T), 0.0, None)
    return float(np.sum(eigenvalues / (1.0 + eigenvalues)))


# ================================================================ statistics
# Interval estimates in the style recommended by Agarwal et al. (NeurIPS 2021).
# Of the PSRO papers surveyed for this project, the ones that report a spread
# at all report a standard deviation over two to five seeds; none reports a
# bootstrap interval or a trimmed mean. Both are cheap on a game this small.

def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle half of the runs.

    Preferred to the mean, which a single bad seed can dominate, and to the
    median, which discards most of an already small sample. With few runs it
    detects a given improvement using considerably fewer seeds than either.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return float("nan")
    low = int(np.floor(ordered.size * 0.25))
    high = int(np.ceil(ordered.size * 0.75))
    middle = ordered[low:high]
    return float(np.mean(middle if middle.size else ordered))


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for any statistic of a set of runs.

    Resamples the seeds with replacement, recomputes the statistic on each
    resample and reads the interval off the resulting distribution. Unlike a
    Student-t interval it assumes nothing about the shape of the sampling
    distribution, which matters here because several of the measured
    quantities, population size and support size among them, are small
    integers and plainly not normal.

    Returns ``(statistic, low, high)``.
    """
    sample = np.asarray(values, dtype=float)
    point = float(statistic(sample))
    if sample.size < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sample.size, size=(resamples, sample.size))
    estimates = np.array([statistic(sample[row]) for row in draws])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return point, float(low), float(high)


def probability_of_improvement(better: Sequence[float], worse: Sequence[float]) -> float:
    """P(a run of ``better`` scores lower than a run of ``worse``).

    Reported alongside, and preferred to, a p-value. Agarwal et al. argue
    against significance tests in small-sample reinforcement learning because a
    dichotomous verdict hides the size of the effect. This is the same
    comparison expressed as an effect size, and it is exactly the statistic
    underlying the Mann-Whitney test. Ties count as half, so two identical
    distributions give ``0.5`` and a clean separation gives ``1.0``.

    Lower is better for exploitability, which is why the comparison is ``<``.
    """
    x = np.asarray(better, dtype=float)[:, None]
    y = np.asarray(worse, dtype=float)[None, :]
    return float((np.sum(x < y) + 0.5 * np.sum(x == y)) / (x.size * y.size))
