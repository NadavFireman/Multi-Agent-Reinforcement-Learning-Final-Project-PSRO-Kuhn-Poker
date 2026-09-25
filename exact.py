# -*- coding: utf-8 -*-
"""Exact game-theoretic quantities: matchup values, best responses, exploitability.

Everything here is computed by full enumeration of the game tree, with no
sampling. For a game as small as Kuhn Poker this is both cheap and exact, which
means the project can report a true exploitability rather than an estimate.

Two facts make the exact best response easy:

1. A best response can always be taken **deterministic**, so it suffices to
   maximise over the finitely many pure strategies of a player (64 in Kuhn
   Poker: six information sets, two actions each).
2. Against a *mixture* of opponent policies the value is linear in the
   meta-strategy, so
   ``BR_i(sigma) = max_pure sum_k sigma_k * V(pure, pi_k)``.
   The per-opponent values ``V(pure, pi_k)`` are cached and reused as the
   population grows, which is what makes the calculator cheap inside the PSRO
   loop.
"""

from __future__ import annotations

from itertools import product
from typing import Iterator, Sequence

import numpy as np

from game import Policy, TabularPolicy, ZeroSumGame

#: Refuse to enumerate pure strategies beyond this many (guards against
#: accidentally calling the exact machinery on a game that is too large).
MAX_PURE_STRATEGIES = 100_000


# --------------------------------------------------------------- exact value
def exact_value(game: ZeroSumGame, policy0: Policy, policy1: Policy) -> float:
    """Exact expected payoff to player 0 when ``policy0`` faces ``policy1``.

    The whole tree is integrated over, so the result carries no sampling error.
    """
    policies = (policy0, policy1)
    total = 0.0
    for chance, prob in game.chance_outcomes():
        total += prob * _node_value(game, chance, game.initial_history(), policies)
    return total


def _node_value(game: ZeroSumGame, chance, history: tuple, policies) -> float:
    if game.is_terminal(history):
        return game.terminal_value(chance, history)
    player = game.current_player(history)
    actions = game.legal_actions(history)
    probs = policies[player].action_probs(game.infoset(chance, history, player), actions)
    value = 0.0
    for action, p in zip(actions, probs):
        if p > 0.0:
            value += p * _node_value(game, chance, history + (action,), policies)
    return value


def exact_payoff_matrix(
    game: ZeroSumGame, population0: Sequence[Policy], population1: Sequence[Policy]
) -> np.ndarray:
    """The payoff matrix of player 0 for every pair of population policies."""
    matrix = np.zeros((len(population0), len(population1)))
    for j, p0 in enumerate(population0):
        for k, p1 in enumerate(population1):
            matrix[j, k] = exact_value(game, p0, p1)
    return matrix


def mixture_value(
    matrix: np.ndarray, weights0: Sequence[float], weights1: Sequence[float]
) -> float:
    """``V_0(sigma_0, sigma_1) = sigma_0^T M sigma_1``, the meta-game value."""
    return float(np.asarray(weights0) @ matrix @ np.asarray(weights1))


# ---------------------------------------------------------- pure strategies
def enumerate_pure_strategies(game: ZeroSumGame, player: int) -> Iterator[TabularPolicy]:
    """Yield every deterministic strategy of ``player``."""
    infosets = game.all_infosets(player)
    action_sets = [game.legal_actions_at_infoset(i) for i in infosets]
    count = 1
    for actions in action_sets:
        count *= len(actions)
    if count > MAX_PURE_STRATEGIES:
        raise ValueError(
            f"{game.name} has {count} pure strategies for player {player}, "
            f"above the limit of {MAX_PURE_STRATEGIES}; use an approximate "
            f"best response instead"
        )
    for index, choice in enumerate(product(*action_sets)):
        probs = {}
        for infoset, actions, chosen in zip(infosets, action_sets, choice):
            row = [1.0 if a == chosen else 0.0 for a in actions]
            probs[infoset] = row
        yield TabularPolicy(
            probs, action_sets[0], name=f"pure{player}_{index}"
        )


class ExactBestResponse:
    """Exact best-response calculator for one player, with a growing cache.

    Parameters
    ----------
    game:
        The game to solve.
    player:
        The player computing the best response.

    Notes
    -----
    ``values`` are always expressed from ``player``'s own point of view, so a
    larger number is better for that player regardless of which seat it holds.
    """

    def __init__(self, game: ZeroSumGame, player: int):
        self.game = game
        self.player = player
        self.pure_strategies: list[TabularPolicy] = list(
            enumerate_pure_strategies(game, player)
        )
        # id(opponent policy) -> (policy, value of every pure strategy against it)
        self._cache: dict[int, tuple[Policy, np.ndarray]] = {}

    # ------------------------------------------------------------- internals
    def _values_against(self, opponent: Policy) -> np.ndarray:
        """Value of every pure strategy against a single opponent policy."""
        key = id(opponent)
        hit = self._cache.get(key)
        if hit is not None:
            return hit[1]
        values = np.empty(len(self.pure_strategies))
        for i, pure in enumerate(self.pure_strategies):
            if self.player == 0:
                values[i] = exact_value(self.game, pure, opponent)
            else:
                values[i] = -exact_value(self.game, opponent, pure)
        # Keep a reference to the policy so its id cannot be recycled.
        self._cache[key] = (opponent, values)
        return values

    # ---------------------------------------------------------------- public
    def best_response(
        self, opponents: Sequence[Policy], weights: Sequence[float]
    ) -> tuple[float, TabularPolicy]:
        """Best value attainable against the opponent mixture, and a strategy achieving it.

        Returns
        -------
        (value, policy)
            ``value`` is in ``player``'s own payoff units.
        """
        if len(opponents) != len(weights):
            raise ValueError("opponents and weights must have the same length")
        w = np.asarray(weights, dtype=float)
        if not np.isclose(w.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got {w.sum()}")
        totals = np.zeros(len(self.pure_strategies))
        for opponent, weight in zip(opponents, w):
            if weight > 0.0:
                totals += weight * self._values_against(opponent)
        best = int(np.argmax(totals))
        return float(totals[best]), self.pure_strategies[best]

    def best_response_value(
        self, opponents: Sequence[Policy], weights: Sequence[float]
    ) -> float:
        return self.best_response(opponents, weights)[0]

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# -------------------------------------------------------------- exploitability
def exploitability(
    game: ZeroSumGame,
    population0: Sequence[Policy],
    weights0: Sequence[float],
    population1: Sequence[Policy],
    weights1: Sequence[float],
    br0: ExactBestResponse | None = None,
    br1: ExactBestResponse | None = None,
    matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """Exact exploitability of a pair of meta-strategies.

    Implements the definition from the project handout,

    ``Exploitability(sigma) = [BR_0(sigma_1) - V_0(sigma)] + [BR_1(sigma_0) - V_1(sigma)]``

    which in a zero-sum game reduces to ``BR_0(sigma_1) + BR_1(sigma_0)`` because
    ``V_0 + V_1 = 0``. Both forms are returned so the identity can be checked.

    Returns
    -------
    dict with keys ``exploitability``, ``br0``, ``br1``, ``value0``,
    ``gap0`` and ``gap1`` (the per-player incentives to deviate).
    """
    br0 = br0 or ExactBestResponse(game, 0)
    br1 = br1 or ExactBestResponse(game, 1)
    if matrix is None:
        matrix = exact_payoff_matrix(game, population0, population1)

    value0 = mixture_value(matrix, weights0, weights1)
    value1 = -value0
    br0_value = br0.best_response_value(population1, weights1)
    br1_value = br1.best_response_value(population0, weights0)
    gap0 = br0_value - value0
    gap1 = br1_value - value1
    return {
        "exploitability": gap0 + gap1,
        "br0": br0_value,
        "br1": br1_value,
        "value0": value0,
        "gap0": gap0,
        "gap1": gap1,
    }


# ------------------------------------------- population-level exploitability
def population_exploitability(
    game: ZeroSumGame,
    population0: Sequence[Policy],
    population1: Sequence[Policy],
) -> dict[str, float]:
    """Exploitability of the *best* mixture each population can support.

    ``exploitability`` above scores one particular pair of meta-strategies, so
    its value depends on the meta-solver that produced them. That makes it the
    wrong yardstick for comparing meta-solvers against one another: each solver
    is then graded by its own notion of a solution. Wang, Ma and Wellman
    (AAMAS 2022) call the requirement to avoid this the *consistency criterion*,
    and report that much of the PSRO literature - OpenSpiel included - violates
    it.

    The solver-independent alternative is the minimum-regret constrained
    profile (Jordan, Schvartzman and Wellman 2010), also called *population
    exploitability* (Yao et al., NeurIPS 2023): the least exploitable pair of
    mixtures the two populations can express.

    It is cheap here because the two halves separate. Exploitability is
    ``BR_0(sigma_1) + BR_1(sigma_0)``, and the first term does not involve
    ``sigma_0`` at all, so

    ``min_{sigma_0, sigma_1} [BR_0(sigma_1) + BR_1(sigma_0)]
       = min_{sigma_1} BR_0(sigma_1) + min_{sigma_0} BR_1(sigma_0)``

    and each half is the value of a matrix game between one player's *complete*
    pure-strategy set and the other player's population - two linear programs.

    The two halves have a second reading. ``min_{sigma_1} BR_0(sigma_1)`` is an
    upper bound on the value of the full game and ``max_{sigma_0} min_b`` is a
    lower bound, so this quantity is exactly the ``v_upper - v_lower`` gap that
    the double-oracle algorithm of McMahan, Gordon and Blum (2003) uses as its
    own stopping rule. Reaching zero here is therefore not a proxy for
    convergence; it *is* the convergence certificate.

    Returns
    -------
    dict with ``population_exploitability``, the two bounds ``value_upper`` and
    ``value_lower`` that bracket the true game value, and the per-player halves.
    """
    from metasolvers import solve_zero_sum_nash

    pure0 = list(enumerate_pure_strategies(game, 0))
    pure1 = list(enumerate_pure_strategies(game, 1))

    # Player 0 free to use any pure strategy, player 1 confined to its population.
    upper = exact_payoff_matrix(game, pure0, list(population1))
    _, _, value_upper = solve_zero_sum_nash(upper)
    # Player 0 confined to its population, player 1 free.
    lower = exact_payoff_matrix(game, list(population0), pure1)
    _, _, value_lower = solve_zero_sum_nash(lower)

    return {
        "population_exploitability": float(value_upper - value_lower),
        "value_upper": float(value_upper),
        "value_lower": float(value_lower),
    }


# ------------------------------------------------ Kuhn behaviour parameters
#: Information sets at which player 0 has already acted once, and the earlier
#: information set whose *check* is what leads there.
_KUHN_PARENT = {"0pb": "0", "1pb": "1", "2pb": "2"}


def kuhn_behaviour_parameters(
    game: ZeroSumGame,
    population: Sequence[Policy],
    weights: Sequence[float],
    player: int,
) -> dict[str, float]:
    """Kuhn's own equilibrium parameters, read off a meta-strategy.

    Kuhn (1950) reduced Simplified Poker to five behaviour parameters and
    showed the equilibria collapse onto a one-parameter family. For player 0,
    with ``alpha`` the chance of bluffing the jack, ``beta`` the chance of
    calling with the queen and ``gamma`` the chance of betting the king,

    ``alpha = gamma / 3``   and   ``beta = gamma / 3 + 1 / 3``.

    Player 1's equilibrium is unique: bluff the jack after a check with
    probability ``1/3`` and call a bet with the queen with probability ``1/3``.

    Checking these is a validation the exploitability curve cannot provide.
    Exploitability near zero says the meta-strategy is hard to beat; matching
    Kuhn's line says it is hard to beat *for the reason Kuhn identified*.

    A mixture over policies is not the average of their behaviours: a component
    only influences play at an information set it actually reaches. The
    behavioural probabilities below are therefore weighted by each component's
    own reach probability, which is what makes them realisation-equivalent to
    the mixture.
    """
    weights = np.asarray(weights, dtype=float)
    per_policy = [p.bet_probabilities(game, player) for p in population]

    def behavioural(infoset: str) -> float:
        parent = _KUHN_PARENT.get(infoset)
        if parent is None:                       # reached regardless of own play
            reach = weights
        else:                                    # reached only after checking
            reach = weights * np.array([1.0 - q[parent] for q in per_policy])
        total = reach.sum()
        if total <= 1e-12:
            return float("nan")
        return float(np.dot(reach, [q[infoset] for q in per_policy]) / total)

    if player == 0:
        alpha, beta, gamma = behavioural("0"), behavioural("1pb"), behavioural("2")
        return {
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "alpha_from_gamma": gamma / 3.0,
            "beta_from_gamma": gamma / 3.0 + 1.0 / 3.0,
            "residual": abs(alpha - gamma / 3.0) + abs(beta - gamma / 3.0 - 1.0 / 3.0),
            "bet_queen_first": behavioural("1"),   # dominated: should be 0
            "fold_jack": 1.0 - behavioural("0pb"),  # dominated: should be 1
            "call_king": behavioural("2pb"),        # dominated: should be 1
        }
    xi, eta = behavioural("0p"), behavioural("1b")
    return {
        "xi": xi, "eta": eta,
        "residual": abs(xi - 1.0 / 3.0) + abs(eta - 1.0 / 3.0),
        "bet_king_after_check": behavioural("2p"),  # dominated: should be 1
        "call_jack": behavioural("0b"),             # dominated: should be 0
        "call_king": behavioural("2b"),             # dominated: should be 1
    }
