# -*- coding: utf-8 -*-
"""Meta-strategy solvers.

A meta-solver reads the empirical payoff matrix ``M`` of the meta-game (entry
``M[j, k]`` is the expected return of player 0 when its policy ``j`` meets
player 1's policy ``k``) and returns a distribution over each player's policy
population. That distribution is what the response oracle trains against, and
it is what turns PSRO into a family of algorithms rather than a single one:

============================  =========================================
Solver                        Resulting algorithm
============================  =========================================
``self_play``                 iterated best response / ordinary self-play
``uniform``                   fictitious-play-like averaging
``fictitious_play``           fictitious play in policy space
``nash``                      the double-oracle algorithm
``replicator``                evolutionary population dynamics
``projected_replicator``      PSRO with forced exploration (PSRO-rN)
``regret_matching``           no-regret population dynamics
============================  =========================================

All solvers share the signature ``solver(matrix, **kwargs) -> (sigma0, sigma1)``
and return probability vectors of length ``M.shape[0]`` and ``M.shape[1]``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "uniform",
    "self_play",
    "fictitious_play",
    "nash",
    "replicator",
    "projected_replicator",
    "regret_matching",
    "solve_zero_sum_nash",
    "SOLVERS",
    "get_solver",
]

_EPS = 1e-12


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    total = v.sum()
    if total <= _EPS:
        return np.full(len(v), 1.0 / len(v))
    return v / total


# ------------------------------------------------------------------- trivial
def uniform(matrix: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
    """Every policy in the population is played equally often."""
    n0, n1 = matrix.shape
    return np.full(n0, 1.0 / n0), np.full(n1, 1.0 / n1)


def self_play(matrix: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
    """All probability mass on the most recently added policy.

    This makes PSRO degenerate into ordinary self-play: the oracle only ever
    trains against the newest opponent, so the population is kept but never
    used. It is the control condition for the project's central question.
    """
    n0, n1 = matrix.shape
    s0, s1 = np.zeros(n0), np.zeros(n1)
    s0[-1] = 1.0
    s1[-1] = 1.0
    return s0, s1


# ----------------------------------------------------------- fictitious play
def fictitious_play(
    matrix: np.ndarray, iterations: int = 2000, **_
) -> tuple[np.ndarray, np.ndarray]:
    """Fictitious play on the meta-game.

    Each player repeatedly best-responds to the opponent's empirical
    distribution of past choices; the returned meta-strategies are those
    empirical distributions. In a zero-sum game these converge to a Nash
    equilibrium, but slowly, which is exactly the behaviour we want to contrast
    with the linear-programming solver.
    """
    n0, n1 = matrix.shape
    counts0 = np.zeros(n0)
    counts1 = np.zeros(n1)
    # Seed with one arbitrary but deterministic choice each.
    counts0[0] += 1.0
    counts1[0] += 1.0
    for _i in range(iterations):
        sigma0 = counts0 / counts0.sum()
        sigma1 = counts1 / counts1.sum()
        best0 = int(np.argmax(matrix @ sigma1))       # player 0 maximises
        best1 = int(np.argmin(sigma0 @ matrix))       # player 1 minimises player 0's payoff
        counts0[best0] += 1.0
        counts1[best1] += 1.0
    return counts0 / counts0.sum(), counts1 / counts1.sum()


# --------------------------------------------------------------- Nash via LP
def solve_zero_sum_nash(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a zero-sum matrix game exactly by linear programming.

    The maximin problem for the row player is

    ``max_{sigma, v} v   subject to   sigma^T M[:, k] >= v for all k,
    sum(sigma) = 1, sigma >= 0``

    which is fed to HiGHS in standard form. The column player's strategy is
    obtained by solving the same problem for the transposed, negated game, and
    the two optimal values must agree by the minimax theorem.

    Returns
    -------
    (sigma0, sigma1, value)
    """
    sigma0, value0 = _maximin(matrix)
    sigma1, value1 = _maximin(-matrix.T)
    # value0 is player 0's guaranteed payoff; value1 is player 1's, in its own
    # units. In a zero-sum game they must sum to zero.
    if not np.isclose(value0, -value1, atol=1e-6):
        raise RuntimeError(
            f"minimax mismatch: row value {value0:.6g} vs column value {-value1:.6g}"
        )
    return sigma0, sigma1, float(value0)


def _maximin(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Maximin strategy and value for the row player of ``matrix``."""
    n_rows, n_cols = matrix.shape
    # Variables: [sigma (n_rows), v]; objective: maximise v == minimise -v.
    c = np.zeros(n_rows + 1)
    c[-1] = -1.0
    # For every column k:  v - sum_j sigma_j M[j, k] <= 0
    a_ub = np.hstack([-matrix.T, np.ones((n_cols, 1))])
    b_ub = np.zeros(n_cols)
    a_eq = np.zeros((1, n_rows + 1))
    a_eq[0, :n_rows] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n_rows + [(None, None)]
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"linear program failed: {result.message}")
    return _normalise(result.x[:n_rows]), float(result.x[-1])


def nash(matrix: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
    """Nash equilibrium of the empirical meta-game (the double-oracle solver)."""
    sigma0, sigma1, _value = solve_zero_sum_nash(matrix)
    return sigma0, sigma1


# ------------------------------------------------------- replicator dynamics
def replicator(
    matrix: np.ndarray,
    iterations: int = 5000,
    step_size: float = 0.05,
    gamma: float = 0.0,
    **_,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time replicator dynamics on the meta-game.

    Each policy's share grows in proportion to how much better it does than the
    population average::

        sigma_j <- sigma_j * (1 + eta * (f_j - f_bar))

    With ``gamma > 0`` the iterate is projected back onto the subset of the
    simplex where every policy keeps at least ``gamma / n`` probability, which
    is the *projected* variant used by :func:`projected_replicator`.
    """
    n0, n1 = matrix.shape
    sigma0 = np.full(n0, 1.0 / n0)
    sigma1 = np.full(n1, 1.0 / n1)
    for _i in range(iterations):
        fitness0 = matrix @ sigma1              # player 0 maximises
        fitness1 = -(sigma0 @ matrix)           # player 1 maximises its own payoff
        sigma0 = _normalise(sigma0 * (1.0 + step_size * (fitness0 - sigma0 @ fitness0)))
        sigma1 = _normalise(sigma1 * (1.0 + step_size * (fitness1 - sigma1 @ fitness1)))
        if gamma > 0.0:
            sigma0 = _project_min_probability(sigma0, gamma)
            sigma1 = _project_min_probability(sigma1, gamma)
    return sigma0, sigma1


def _project_min_probability(sigma: np.ndarray, gamma: float) -> np.ndarray:
    """Project onto ``{sigma : sigma_j >= gamma / n}`` keeping the total at 1."""
    n = len(sigma)
    floor = gamma / n
    if floor * n >= 1.0:
        return np.full(n, 1.0 / n)
    free_mass = 1.0 - floor * n
    return floor + free_mass * _normalise(sigma - np.minimum(sigma, floor))


def projected_replicator(
    matrix: np.ndarray,
    iterations: int = 5000,
    step_size: float = 0.05,
    gamma: float = 0.1,
    **_,
) -> tuple[np.ndarray, np.ndarray]:
    """Projected replicator dynamics: replicator with guaranteed exploration.

    Forcing a floor on every policy's probability stops the meta-strategy from
    collapsing onto a single policy, so the oracle keeps seeing the whole
    population. This is the mechanism the ablation study isolates.
    """
    return replicator(matrix, iterations=iterations, step_size=step_size, gamma=gamma)


# -------------------------------------------------------------- no regret
def regret_matching(
    matrix: np.ndarray, iterations: int = 5000, **_
) -> tuple[np.ndarray, np.ndarray]:
    """Regret matching: probabilities proportional to accumulated positive regret.

    Both players run the classical no-regret update against each other, and the
    returned meta-strategies are the time-averaged play, which converges to a
    Nash equilibrium of the zero-sum meta-game.
    """
    n0, n1 = matrix.shape
    regret0 = np.zeros(n0)
    regret1 = np.zeros(n1)
    sum0 = np.zeros(n0)
    sum1 = np.zeros(n1)
    for _i in range(iterations):
        sigma0 = _positive_part_or_uniform(regret0)
        sigma1 = _positive_part_or_uniform(regret1)
        sum0 += sigma0
        sum1 += sigma1
        payoff0 = matrix @ sigma1
        payoff1 = -(sigma0 @ matrix)
        regret0 += payoff0 - sigma0 @ payoff0
        regret1 += payoff1 - sigma1 @ payoff1
    return _normalise(sum0), _normalise(sum1)


def _positive_part_or_uniform(regret: np.ndarray) -> np.ndarray:
    positive = np.clip(regret, 0.0, None)
    total = positive.sum()
    if total <= _EPS:
        return np.full(len(regret), 1.0 / len(regret))
    return positive / total


# ------------------------------------------------------------------ registry
def rectified_nash(matrix: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
    """Nash, with every opponent the incumbent loses to removed from the mixture.

    This is the meta-solver behind PSRO_rN (Balduzzi et al., ICML 2019). Their
    slogan is that agents should "amplify their strengths and ignore their
    weaknesses": each learner trains only against the Nash-weighted mixture of
    opponents it already beats or ties, which is meant to carve out
    game-theoretic niches instead of chasing one global best response.

    Two adaptations are needed to run it here, and both are worth stating
    because the original is defined only for symmetric games - a limitation
    the authors are explicit about, and which Muller et al. (ICLR 2020) invoke
    when reporting that it fails on Kuhn Poker.

    First, "beats or ties" has no meaning in a game whose value is not zero.
    Kuhn Poker is worth -1/18 to the player who moves first, so a policy that
    scores exactly the game value has done neither well nor badly. The
    comparison is therefore made against the value of the current meta-game
    rather than against zero.

    Second, the rectifier in the original is relative to *one particular
    agent*, since every Nash-supported agent is updated separately. This
    single-add variant rectifies against the incumbent, meaning the
    Nash-supported policy carrying the most weight, which keeps the population
    growing at the same rate as every other solver and so keeps the comparison
    in experiment 2 on equal terms.

    It is included because it is the canonical diversity-driven meta-solver
    *and* because it is known to fail: McAleer et al. (NeurIPS 2020, Prop. 3.1)
    exhibit a game where it provably stops short of Nash, and Muller et al.
    report it looping on Kuhn specifically. Reproducing a documented failure is
    more informative than omitting the method.
    """
    sigma0, sigma1 = nash(matrix)
    value = float(sigma0 @ matrix @ sigma1)
    incumbent0, incumbent1 = int(np.argmax(sigma0)), int(np.argmax(sigma1))

    # Opponents the incumbent does not lose to, from each side's point of view.
    keep1 = sigma1 * (matrix[incumbent0, :] >= value - 1e-12)
    keep0 = sigma0 * (matrix[:, incumbent1] <= value + 1e-12)
    # An agent that loses to everything has no niche to defend, so it keeps the
    # unrectified mixture rather than training against nothing at all.
    sigma1 = _normalise(keep1) if keep1.sum() > 1e-12 else sigma1
    sigma0 = _normalise(keep0) if keep0.sum() > 1e-12 else sigma0
    return sigma0, sigma1


SOLVERS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "uniform": uniform,
    "self_play": self_play,
    "fictitious_play": fictitious_play,
    "nash": nash,
    "replicator": replicator,
    "projected_replicator": projected_replicator,
    "regret_matching": regret_matching,
    "rectified_nash": rectified_nash,
}

#: Solvers that can produce a genuinely non-uniform stochastic mixture, as
#: required by the project specification.
STOCHASTIC_SOLVERS = (
    "fictitious_play",
    "nash",
    "replicator",
    "projected_replicator",
    "regret_matching",
    "rectified_nash",
)


def get_solver(name: str) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    try:
        return SOLVERS[name]
    except KeyError:
        raise KeyError(
            f"unknown meta-solver {name!r}; available: {sorted(SOLVERS)}"
        ) from None


def strategy_entropy(sigma: np.ndarray) -> float:
    """Shannon entropy (nats) of a meta-strategy, ``H = -sum p log p``."""
    p = np.asarray(sigma, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
