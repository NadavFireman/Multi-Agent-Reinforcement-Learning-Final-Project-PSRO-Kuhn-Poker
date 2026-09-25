# -*- coding: utf-8 -*-
"""The game, the policies that play it, and episode simulation (Part A).

Three layers live here because they only make sense together:

* :class:`ZeroSumGame` - the abstract description of a finite two-player
  zero-sum game with imperfect information: a chance node, a tree of decisions,
  and information sets that hide what a player cannot see.
* :class:`Policy` and its concrete forms - what populates a PSRO population.
  :class:`MixturePolicy` is the important one: it draws a member at the start of
  each episode, which is exactly the opponent the response oracle must face.
* :func:`play_episode` and friends - sampling actual play from those objects.

The concrete game is :class:`KuhnPoker`, together with its analytic equilibrium
and the three baseline agents the specification asks for.

Conventions
-----------
* Two players, indexed 0 and 1.
* Zero-sum: ``terminal_value`` returns the payoff of **player 0**; player 1
  receives its negation.
* ``chance`` is an opaque hashable produced by :meth:`chance_outcomes` (the
  dealt cards); ``history`` is a tuple of actions, so it is hashable and cheap
  to extend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import permutations
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


class ZeroSumGame(ABC):
    """A finite two-player zero-sum game with imperfect information."""

    name: str = "game"
    num_players: int = 2

    # ------------------------------------------------------------------ tree
    @abstractmethod
    def chance_outcomes(self) -> Sequence[tuple[Hashable, float]]:
        """All outcomes of the initial chance node with their probabilities.

        Returns a sequence of ``(chance, probability)`` pairs summing to 1.
        """

    def initial_history(self) -> tuple:
        """The empty action history at the root of the decision tree."""
        return ()

    @abstractmethod
    def is_terminal(self, history: tuple) -> bool:
        """Whether ``history`` ends the episode."""

    @abstractmethod
    def terminal_value(self, chance: Hashable, history: tuple) -> float:
        """Payoff of **player 0** at a terminal history."""

    @abstractmethod
    def current_player(self, history: tuple) -> int:
        """Index of the player to act at a non-terminal ``history``."""

    @abstractmethod
    def legal_actions(self, history: tuple) -> tuple:
        """Actions available at a non-terminal ``history``."""

    @abstractmethod
    def infoset(self, chance: Hashable, history: tuple, player: int) -> str:
        """The information-set key observed by ``player``.

        Two states that a player cannot distinguish must map to the same key.
        """

    @abstractmethod
    def all_infosets(self, player: int) -> tuple[str, ...]:
        """Every information-set key belonging to ``player``, in a fixed order.

        The order is part of the game's contract: pure strategies are
        enumerated against it, so it must be deterministic across runs.
        """

    # ------------------------------------------------------------- utilities
    def num_pure_strategies(self, player: int) -> int:
        """Number of deterministic strategies available to ``player``."""
        n = 1
        for infoset in self.all_infosets(player):
            n *= len(self.legal_actions_at_infoset(infoset))
        return n

    @abstractmethod
    def legal_actions_at_infoset(self, infoset: str) -> tuple:
        """Actions available at an information set, identified by its key."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.name}>"


class Policy(ABC):
    """A (possibly stochastic) behavioural strategy for one player."""

    #: Short human-readable label used in plots, tables and logs.
    name: str = "policy"

    @abstractmethod
    def action_probs(self, infoset: str, legal_actions: Sequence) -> np.ndarray:
        """Probability of each action in ``legal_actions`` at ``infoset``."""

    def act(self, infoset: str, legal_actions: Sequence, rng: np.random.Generator):
        """Sample a single action.

        The binary case is special-cased on purpose: ``rng.choice`` with an
        explicit probability vector is far slower than one uniform draw, and
        this is the hottest line in the whole project - the response oracle
        calls it once per decision for millions of training episodes.
        """
        probs = self.action_probs(infoset, legal_actions)
        if len(legal_actions) == 2:
            return legal_actions[1] if rng.random() < probs[1] else legal_actions[0]
        return legal_actions[rng.choice(len(legal_actions), p=probs)]

    def bet_probabilities(self, game, player: int) -> dict[str, float]:
        """Probability of the *second* action at each of ``player``'s infosets.

        For poker the two actions are ``('p', 'b')``, so this is the chance of
        betting or calling. It gives every policy - tabular, learned or
        analytic - a common behavioural signature that can be compared,
        averaged or checked against a known equilibrium.
        """
        return {
            infoset: float(
                self.action_probs(infoset, game.legal_actions_at_infoset(infoset))[1]
            )
            for infoset in game.all_infosets(player)
        }

    def begin_episode(self, rng: np.random.Generator) -> None:
        """Hook called once per episode.

        Only stateful policies need this; :class:`MixturePolicy` uses it to
        draw the member policy that will play the whole episode.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.name}>"


class RandomPolicy(Policy):
    """Uniformly random over the legal actions."""

    def __init__(self, name: str = "random"):
        self.name = name

    def action_probs(self, infoset: str, legal_actions: Sequence) -> np.ndarray:
        n = len(legal_actions)
        return np.full(n, 1.0 / n)


class TabularPolicy(Policy):
    """An explicit table mapping each information set to action probabilities.

    This is the representation produced by the tabular response oracle and by
    every hand-written baseline, and it is what gets checkpointed to disk.
    """

    def __init__(
        self,
        probs: Mapping[str, Sequence[float]],
        actions: Sequence,
        name: str = "tabular",
    ):
        self.actions = tuple(actions)
        self.name = name
        self._probs: dict[str, np.ndarray] = {}
        for infoset, row in probs.items():
            arr = np.asarray(row, dtype=float)
            if arr.shape != (len(self.actions),):
                raise ValueError(
                    f"infoset {infoset!r}: expected {len(self.actions)} "
                    f"probabilities, got {arr.shape}"
                )
            total = arr.sum()
            if not np.isclose(total, 1.0):
                raise ValueError(
                    f"infoset {infoset!r}: probabilities sum to {total}, not 1"
                )
            self._probs[infoset] = arr

    # ---------------------------------------------------------- construction
    @classmethod
    def from_bet_probabilities(
        cls, game, player: int, bet_prob: Mapping[str, float], name: str = "tabular"
    ) -> "TabularPolicy":
        """Build a binary-action policy from the probability of the *second* action.

        For poker games the two actions are ``('p', 'b')``, so ``bet_prob``
        gives the chance of betting or calling at each information set.
        """
        infosets = game.all_infosets(player)
        actions = game.legal_actions_at_infoset(infosets[0])
        if len(actions) != 2:
            raise ValueError("from_bet_probabilities requires exactly two actions")
        missing = set(infosets) - set(bet_prob)
        if missing:
            raise ValueError(f"missing probabilities for information sets: {sorted(missing)}")
        probs = {i: (1.0 - float(bet_prob[i]), float(bet_prob[i])) for i in infosets}
        return cls(probs, actions, name=name)

    @classmethod
    def uniform(cls, game, player: int, name: str = "uniform") -> "TabularPolicy":
        """A uniform-random policy in explicit tabular form."""
        infosets = game.all_infosets(player)
        actions = game.legal_actions_at_infoset(infosets[0])
        n = len(actions)
        return cls({i: [1.0 / n] * n for i in infosets}, actions, name=name)

    # ------------------------------------------------------------ behaviour
    def action_probs(self, infoset: str, legal_actions: Sequence) -> np.ndarray:
        try:
            return self._probs[infoset]
        except KeyError:
            # An unseen information set: fall back to uniform rather than crash,
            # so a partially trained oracle is still a valid policy.
            n = len(legal_actions)
            return np.full(n, 1.0 / n)

    @property
    def infosets(self) -> tuple[str, ...]:
        return tuple(self._probs)

    def is_deterministic(self) -> bool:
        return all(np.isclose(p.max(), 1.0) for p in self._probs.values())

    def entropy(self) -> float:
        """Mean Shannon entropy (nats) of the action distributions."""
        vals = []
        for p in self._probs.values():
            nz = p[p > 0]
            vals.append(float(-(nz * np.log(nz)).sum()))
        return float(np.mean(vals)) if vals else 0.0

    # -------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "type": "TabularPolicy",
            "name": self.name,
            "actions": list(self.actions),
            "probs": {k: v.tolist() for k, v in self._probs.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TabularPolicy":
        return cls(data["probs"], data["actions"], name=data.get("name", "tabular"))


def policy_to_dict(policy: Policy) -> dict:
    """Serialise a population policy so a run can be checkpointed."""
    if isinstance(policy, TabularPolicy):
        return policy.to_dict()
    if isinstance(policy, RandomPolicy):
        return {"type": "RandomPolicy", "name": policy.name}
    raise TypeError(
        f"cannot serialise {type(policy).__name__}; population policies must be "
        f"TabularPolicy or RandomPolicy"
    )


def policy_from_dict(data: dict) -> Policy:
    """Inverse of :func:`policy_to_dict`."""
    kind = data["type"]
    if kind == "TabularPolicy":
        return TabularPolicy.from_dict(data)
    if kind == "RandomPolicy":
        return RandomPolicy(data.get("name", "random"))
    raise TypeError(f"unknown policy type {kind!r}")


class MixturePolicy(Policy):
    """Plays one member of a population, resampled at the start of each episode.

    This is exactly the opponent the response oracle faces: the meta-strategy
    ``weights`` selects an opponent policy, and that opponent then plays the
    whole episode, as required by the PSRO specification.

    Note that a mixture over policies is *not* the same object as the
    behavioural strategy obtained by averaging the tables: the opponent's card
    and its policy identity stay correlated across the episode. The exact
    machinery in :mod:`psro.exact` respects that distinction.
    """

    def __init__(
        self,
        policies: Sequence[Policy],
        weights: Sequence[float],
        name: str = "mixture",
    ):
        if len(policies) != len(weights):
            raise ValueError("policies and weights must have the same length")
        if len(policies) == 0:
            raise ValueError("a mixture needs at least one policy")
        w = np.asarray(weights, dtype=float)
        if np.any(w < -1e-9):
            raise ValueError("meta-strategy weights must be non-negative")
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total <= 0:
            raise ValueError("meta-strategy weights must not be all zero")
        self.policies = list(policies)
        self.weights = w / total
        self.name = name
        self._cumulative = np.cumsum(self.weights)
        self._current = self.policies[int(np.argmax(self.weights))]

    def begin_episode(self, rng: np.random.Generator) -> None:
        # searchsorted on the cumulative weights is markedly faster than
        # rng.choice with a probability vector, and this runs once per episode.
        index = int(np.searchsorted(self._cumulative, rng.random(), side="right"))
        self._current = self.policies[min(index, len(self.policies) - 1)]

    def action_probs(self, infoset: str, legal_actions: Sequence) -> np.ndarray:
        return self._current.action_probs(infoset, legal_actions)

    @property
    def support(self) -> list[int]:
        """Indices of the policies with strictly positive probability."""
        return [i for i, w in enumerate(self.weights) if w > 0]

    def entropy(self) -> float:
        """Shannon entropy (nats) of the meta-strategy itself."""
        w = self.weights[self.weights > 0]
        return float(-(w * np.log(w)).sum())


PASS = "p"
BET = "b"
ACTIONS: tuple[str, str] = (PASS, BET)

CARD_NAMES = {0: "J", 1: "Q", 2: "K"}

#: Terminal histories mapped to (pot swing, whether a showdown decides it).
#: For fold endings the sign is fixed; for showdowns the higher card wins.
_TERMINALS: dict[tuple[str, ...], tuple[float, bool]] = {
    (PASS, PASS): (1.0, True),
    (PASS, BET, PASS): (-1.0, False),
    (PASS, BET, BET): (2.0, True),
    (BET, PASS): (1.0, False),
    (BET, BET): (2.0, True),
}


class KuhnPoker(ZeroSumGame):
    """Three-card Kuhn Poker."""

    name = "kuhn_poker"

    def chance_outcomes(self) -> Sequence[tuple[Hashable, float]]:
        deals = list(permutations(range(3), 2))
        p = 1.0 / len(deals)
        return [(deal, p) for deal in deals]

    def is_terminal(self, history: tuple) -> bool:
        return history in _TERMINALS

    def terminal_value(self, chance: Hashable, history: tuple) -> float:
        amount, showdown = _TERMINALS[history]
        if not showdown:
            return amount
        # Higher card wins the pot swing.
        return amount if chance[0] > chance[1] else -amount

    def current_player(self, history: tuple) -> int:
        return len(history) % 2

    def legal_actions(self, history: tuple) -> tuple:
        return ACTIONS

    def legal_actions_at_infoset(self, infoset: str) -> tuple:
        return ACTIONS

    def infoset(self, chance: Hashable, history: tuple, player: int) -> str:
        return f"{chance[player]}{''.join(history)}"

    def all_infosets(self, player: int) -> tuple[str, ...]:
        if player == 0:
            suffixes = ("", "pb")
        else:
            suffixes = ("p", "b")
        return tuple(f"{card}{s}" for s in suffixes for card in range(3))

    # ------------------------------------------------------------- rendering
    @staticmethod
    def describe_infoset(infoset: str) -> str:
        """Human-readable description, e.g. ``'2pb'`` -> ``'K after check-bet'``."""
        card = CARD_NAMES[int(infoset[0])]
        history = infoset[1:]
        context = {
            "": "to act first",
            "p": "after opponent checked",
            "b": "facing a bet",
            "pb": "facing a bet after checking",
        }[history]
        return f"{card} {context}"


# --------------------------------------------------------------- equilibrium
def kuhn_nash_policies(alpha: float = 1.0 / 6.0):
    """The analytic Nash equilibrium of Kuhn Poker, parameterised by ``alpha``.

    Player 0's equilibrium strategies form a one-parameter family for
    ``alpha`` in ``[0, 1/3]``; player 1 has a unique equilibrium strategy.
    The value of the game to player 0 is ``-1/18`` for every member.

    Returns
    -------
    (TabularPolicy, TabularPolicy)
        Equilibrium strategies for player 0 and player 1.
    """

    if not 0.0 <= alpha <= 1.0 / 3.0 + 1e-12:
        raise ValueError(f"alpha must lie in [0, 1/3], got {alpha}")

    # Probability of betting / calling at each information set.
    p0_bet = {
        "0": alpha,           # bluff with the jack
        "1": 0.0,             # never bet the queen first
        "2": 3.0 * alpha,     # value-bet the king at three times the bluff rate
        "0pb": 0.0,           # always fold the jack
        "1pb": alpha + 1.0 / 3.0,
        "2pb": 1.0,           # always call with the king
    }
    p1_bet = {
        "0p": 1.0 / 3.0,      # bluff the jack after a check
        "1p": 0.0,
        "2p": 1.0,
        "0b": 0.0,            # always fold the jack
        "1b": 1.0 / 3.0,
        "2b": 1.0,
    }
    game = KuhnPoker()
    return (
        TabularPolicy.from_bet_probabilities(game, 0, p0_bet, name=f"nash(a={alpha:.3f})"),
        TabularPolicy.from_bet_probabilities(game, 1, p1_bet, name="nash"),
    )


#: Value of Kuhn Poker to player 0 under optimal play.
KUHN_GAME_VALUE = -1.0 / 18.0


# ----------------------------------------------------------------- baselines
def kuhn_heuristic(player: int, name: str = "heuristic"):
    """A simple, plausible rule-of-thumb strategy ("only play the king").

    The agent bets and calls exclusively with the king and otherwise checks or
    folds. It is deterministic, never bluffs, and is therefore easy to exploit
    by betting relentlessly - which makes it a useful baseline.
    """

    game = KuhnPoker()
    bet_prob = {
        infoset: (1.0 if infoset[0] == "2" else 0.0)
        for infoset in game.all_infosets(player)
    }
    return TabularPolicy.from_bet_probabilities(game, player, bet_prob, name=name)


def kuhn_always_bet(player: int, name: str = "always_bet"):
    """A maximally aggressive baseline that bets and calls with every card."""

    game = KuhnPoker()
    bet_prob = {infoset: 1.0 for infoset in game.all_infosets(player)}
    return TabularPolicy.from_bet_probabilities(game, player, bet_prob, name=name)


def _chance_table(game: ZeroSumGame):
    """Cache the chance node's outcomes, and whether it is uniform.

    The chance distribution never changes, so deriving it once per game object
    rather than once per episode removes what profiling showed to be the single
    largest cost in the whole training loop.
    """
    table = getattr(game, "_chance_cache", None)
    if table is None:
        outcomes = game.chance_outcomes()
        values = [c for c, _p in outcomes]
        probabilities = np.fromiter((p for _c, p in outcomes), dtype=float,
                                    count=len(outcomes))
        uniform = bool(np.allclose(probabilities, probabilities[0]))
        table = (values, probabilities, uniform)
        game._chance_cache = table
    return table


def _sample_chance(game: ZeroSumGame, rng: np.random.Generator):
    """Draw one outcome of the initial chance node.

    A uniform chance node - which every poker deal is - is drawn with a single
    integer, avoiding the much slower probability-weighted path.
    """
    values, probabilities, uniform = _chance_table(game)
    if uniform:
        return values[int(rng.integers(len(values)))]
    return values[rng.choice(len(values), p=probabilities)]


def play_episode(
    game: ZeroSumGame, policies: Sequence[Policy], rng: np.random.Generator
) -> float:
    """Play one episode and return the realised payoff of player 0.

    ``begin_episode`` is called on both policies first, which is what lets a
    :class:`~psro.policies.MixturePolicy` draw the single opponent that stays
    fixed for the whole episode, as the PSRO specification requires.
    """
    for policy in policies:
        policy.begin_episode(rng)

    chance = _sample_chance(game, rng)

    history = game.initial_history()
    while not game.is_terminal(history):
        player = game.current_player(history)
        actions = game.legal_actions(history)
        infoset = game.infoset(chance, history, player)
        history = history + (policies[player].act(infoset, actions, rng),)
    return game.terminal_value(chance, history)


def play_episode_trajectory(
    game: ZeroSumGame,
    policies: Sequence[Policy],
    rng: np.random.Generator,
    learner: int,
) -> tuple[list[tuple[str, object]], float]:
    """Play one episode, recording the learner's decisions.

    Returns the learner's ``(infoset, action)`` pairs in order together with the
    episode return **from the learner's point of view**. This is the data the
    response oracle learns from.
    """
    for policy in policies:
        policy.begin_episode(rng)

    chance = _sample_chance(game, rng)

    history = game.initial_history()
    trajectory: list[tuple[str, object]] = []
    while not game.is_terminal(history):
        player = game.current_player(history)
        actions = game.legal_actions(history)
        infoset = game.infoset(chance, history, player)
        action = policies[player].act(infoset, actions, rng)
        if player == learner:
            trajectory.append((infoset, action))
        history = history + (action,)

    value = game.terminal_value(chance, history)
    return trajectory, (value if learner == 0 else -value)


def transpose_policy(
    game: ZeroSumGame, policy: Policy, from_player: int, to_player: int
) -> "TabularPolicy":
    """Move a policy into the other seat, matching decision situations by type.

    The two seats of Kuhn Poker have different information sets, but they pair
    up one-to-one by the situation they describe: "act first" and "opponent
    checked" are both *no bet on the table*, and "facing a bet after checking"
    and "facing a bet" are both *a bet to answer*. The information sets are
    listed in that order by :meth:`all_infosets`, so the transfer is positional.

    This is what makes a head-to-head comparison between two policies of the
    same player possible at all: without it they would never meet.
    """
    source = game.all_infosets(from_player)
    target = game.all_infosets(to_player)
    actions = game.legal_actions_at_infoset(source[0])
    return TabularPolicy(
        {t: policy.action_probs(s, actions) for s, t in zip(source, target)},
        actions, name=f"{policy.name}@{to_player}",
    )


def head_to_head_matrix(
    game: ZeroSumGame, policies: Sequence[Policy], player: int = 0
) -> np.ndarray:
    """Antisymmetric matrix of who beats whom, with the seat advantage removed.

    Kuhn Poker is not symmetric - the player who acts first is worth ``-1/18``
    under optimal play - so simply seating one policy opposite another would
    measure the seat as much as the policy. Each pair is therefore played
    *both ways* and averaged::

        H[i, j] = (V(i first, j second) - V(j first, i second)) / 2

    which is exactly the two-sided evaluation used to neutralise a first-player
    advantage. The result is antisymmetric with a zero diagonal, which is the
    precondition for reading cycles out of it.
    """
    from exact import exact_value

    n = len(policies)
    other = 1 - player
    seated = [transpose_policy(game, p, player, other) for p in policies]
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if player == 0:
                i_first = exact_value(game, policies[i], seated[j])
                j_first = exact_value(game, policies[j], seated[i])
            else:
                i_first = -exact_value(game, seated[j], policies[i])
                j_first = -exact_value(game, seated[i], policies[j])
            matrix[i, j] = 0.5 * (i_first - j_first)
            matrix[j, i] = -matrix[i, j]
    return matrix


def payoff_distribution(
    game: ZeroSumGame, policy0: Policy, policy1: Policy
) -> tuple[np.ndarray, np.ndarray]:
    """Exact distribution of one episode's payoff to player 0.

    A hand of Kuhn Poker has only a few dozen possible (deal, ending) pairs, so
    the whole distribution can be integrated out of the tree instead of being
    discovered by simulation. Returns the distinct payoff values and their
    probabilities.

    Both policies must be stateless across an episode, which every population
    policy is; a :class:`MixturePolicy` resamples its member per episode and is
    rejected, because its payoff distribution is a weighted mixture of the
    distributions of its members rather than a single one.
    """
    for policy in (policy0, policy1):
        if isinstance(policy, MixturePolicy):
            raise TypeError(
                "payoff_distribution needs stateless policies; expand the "
                "mixture over its members and weight the results instead"
            )
    weights: dict[float, float] = {}
    for chance, chance_probability in game.chance_outcomes():
        _accumulate_payoffs(game, chance, game.initial_history(),
                            chance_probability, (policy0, policy1), weights)
    values = np.fromiter(weights.keys(), dtype=float, count=len(weights))
    probabilities = np.fromiter(weights.values(), dtype=float, count=len(weights))
    return values, probabilities / probabilities.sum()


def _accumulate_payoffs(game, chance, history, probability, policies, weights):
    if probability <= 0.0:
        return
    if game.is_terminal(history):
        value = game.terminal_value(chance, history)
        weights[value] = weights.get(value, 0.0) + probability
        return
    player = game.current_player(history)
    actions = game.legal_actions(history)
    action_probabilities = policies[player].action_probs(
        game.infoset(chance, history, player), actions)
    for action, p in zip(actions, action_probabilities):
        if p > 0.0:
            _accumulate_payoffs(game, chance, history + (action,),
                                probability * p, policies, weights)


def sample_payoffs(
    game: ZeroSumGame,
    policy0: Policy,
    policy1: Policy,
    episodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``episodes`` payoffs, equivalent to playing that many hands.

    Sampling from the exact payoff distribution is *distributionally identical*
    to simulating episode by episode, because the joint law of (deal, ending) is
    exactly what the traversal integrates. The estimate therefore carries the
    same sampling noise and the same standard error as a simulation would, and
    costs a single categorical draw instead of thousands of Python loops.
    """
    values, probabilities = payoff_distribution(game, policy0, policy1)
    return values[rng.choice(len(values), size=episodes, p=probabilities)]


def evaluate_pair(
    game: ZeroSumGame,
    policy0: Policy,
    policy1: Policy,
    episodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Play ``episodes`` episodes and return the per-episode payoffs of player 0."""
    return np.fromiter(
        (play_episode(game, (policy0, policy1), rng) for _ in range(episodes)),
        dtype=float,
        count=episodes,
    )
