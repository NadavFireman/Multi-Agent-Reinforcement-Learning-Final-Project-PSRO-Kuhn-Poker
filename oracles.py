# -*- coding: utf-8 -*-
"""Response oracles (Part C of the project specification).

An oracle answers the central PSRO subroutine: given the opponent's current
meta-strategy ``sigma``, produce an approximate best response

``pi_BR ~= argmax_pi E_{pi' ~ sigma} [ G(pi, pi') ]``.

Every oracle here follows the rule the specification lays down: at the start of
each training episode one opponent policy is **sampled** from the meta-strategy
and then stays fixed for that whole episode. That is what makes the learner face
the mixture rather than an artificial average opponent.

Independent implementation
--------------------------
No reinforcement-learning library is used. The tabular Q-learning update, the
epsilon schedule, the REINFORCE gradient estimator with its baseline, the
episode loop and the exact enumeration oracle are all written from scratch in
this file; only NumPy (array arithmetic and random numbers) and SciPy (used
elsewhere for linear programming) are external.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from exact import ExactBestResponse
from game import MixturePolicy, Policy, TabularPolicy, ZeroSumGame, play_episode_trajectory


@dataclass
class OracleResult:
    """What an oracle returns: the new policy plus training diagnostics."""

    policy: Policy
    episodes: int
    mean_return: float
    final_return: float
    wall_time: float
    extra: dict = field(default_factory=dict)


class Oracle(ABC):
    """Base class for approximate best-response oracles."""

    name: str = "oracle"

    @abstractmethod
    def train(
        self,
        game: ZeroSumGame,
        player: int,
        opponents: Sequence[Policy],
        weights: Sequence[float],
        budget: int,
        seed: int,
    ) -> OracleResult:
        """Train a best response to the opponent mixture ``(opponents, weights)``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.name}>"


def _make_opponent(opponents: Sequence[Policy], weights: Sequence[float]) -> MixturePolicy:
    return MixturePolicy(opponents, weights, name="meta_opponent")


def _running_mean(values: list[float], window: int = 500) -> float:
    if not values:
        return 0.0
    return float(np.mean(values[-window:]))


# ------------------------------------------------------------ Q-learning
class TabularQLearningOracle(Oracle):
    """Best response by tabular Q-learning over information sets.

    The learner keeps ``Q[infoset][action]`` and follows an epsilon-greedy
    behaviour policy whose exploration decays linearly, as in the tabular
    assignment earlier in the course. Rewards arrive only at the end of an
    episode, so within one episode the update is

    * for the learner's last decision:   ``target = G``
    * for any earlier decision:          ``target = max_a Q(next_infoset, a)``

    which is the standard off-policy Q-learning target specialised to a game
    with a single terminal reward. The returned policy is greedy with respect to
    the learned values, and a best response can always be taken deterministic,
    so this loses nothing in principle.
    """

    name = "q_learning"

    def __init__(
        self,
        learning_rate: float | None = None,
        lr_power: float = 0.7,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.10,
        epsilon_decay_fraction: float = 0.8,
        optimistic_init: float = 0.0,
    ):
        """
        Parameters
        ----------
        learning_rate:
            A constant step size, or ``None`` (the default) for the count-based
            schedule ``alpha = 1 / N(I, a) ** lr_power``.

            The count-based schedule is the default for a concrete reason. With a
            constant step size the estimate of a *noisy* action keeps a
            permanent random wobble, while a deterministic action's estimate is
            exact. In Kuhn Poker checking the jack is worth exactly -1 with no
            variance, whereas betting it is worth -0.5 but pays -2 or +1 at
            random. A run of unlucky bets pushes the noisy estimate below the
            noiseless one, the greedy policy stops betting, exploration of that
            action collapses to ``epsilon / 2``, and the value never recovers.
            Averaging the samples removes the wobble and the failure with it.
        epsilon_end:
            Floor of the exploration schedule. Kept well above zero for the same
            reason: every action must keep receiving samples.
        """
        self.learning_rate = learning_rate
        self.lr_power = lr_power
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_fraction = epsilon_decay_fraction
        self.optimistic_init = optimistic_init

    def _epsilon(self, step: int, budget: int) -> float:
        span = max(1.0, self.epsilon_decay_fraction * budget)
        frac = min(1.0, step / span)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def train(self, game, player, opponents, weights, budget, seed) -> OracleResult:
        start = time.perf_counter()
        rng = np.random.default_rng([seed, player, budget])
        opponent = _make_opponent(opponents, weights)
        infosets = game.all_infosets(player)
        actions = game.legal_actions_at_infoset(infosets[0])
        n_actions = len(actions)
        action_index = {a: i for i, a in enumerate(actions)}
        q = {i: np.full(n_actions, self.optimistic_init) for i in infosets}
        counts = {i: np.zeros(n_actions) for i in infosets}

        returns: list[float] = []
        for episode in range(budget):
            epsilon = self._epsilon(episode, budget)
            behaviour = _EpsilonGreedy(q, actions, epsilon)
            policies = (behaviour, opponent) if player == 0 else (opponent, behaviour)
            trajectory, value = play_episode_trajectory(game, policies, rng, player)
            returns.append(value)

            # Backwards Q-learning updates along the learner's own decisions.
            for t in reversed(range(len(trajectory))):
                infoset, action = trajectory[t]
                a = action_index[action]
                if t == len(trajectory) - 1:
                    target = value
                else:
                    target = float(q[trajectory[t + 1][0]].max())
                counts[infoset][a] += 1.0
                if self.learning_rate is None:
                    alpha = 1.0 / counts[infoset][a] ** self.lr_power
                else:
                    alpha = self.learning_rate
                q[infoset][a] += alpha * (target - q[infoset][a])

        policy = _greedy_policy(q, actions, infosets, name=f"br_q{budget}")
        return OracleResult(
            policy=policy,
            episodes=budget,
            mean_return=float(np.mean(returns)) if returns else 0.0,
            final_return=_running_mean(returns),
            wall_time=time.perf_counter() - start,
            extra={"q_values": {k: v.tolist() for k, v in q.items()},
                   "visit_counts": {k: v.tolist() for k, v in counts.items()}},
        )


class _EpsilonGreedy(Policy):
    """Behaviour policy used during Q-learning; not part of the population."""

    name = "epsilon_greedy"

    def __init__(self, q: dict, actions: Sequence, epsilon: float):
        self._q = q
        self._actions = tuple(actions)
        self._epsilon = epsilon
        self._n = len(actions)

    def action_probs(self, infoset: str, legal_actions: Sequence) -> np.ndarray:
        values = self._q.get(infoset)
        probs = np.full(self._n, self._epsilon / self._n)
        if values is None:
            return np.full(self._n, 1.0 / self._n)
        best = np.flatnonzero(values == values.max())
        probs[best] += (1.0 - self._epsilon) / len(best)
        return probs


def _greedy_policy(
    q: dict, actions: Sequence, infosets: Sequence[str], name: str
) -> TabularPolicy:
    """Deterministic policy that is greedy with respect to ``q``.

    Ties are broken towards the first action so the result is reproducible; an
    information set that was never visited keeps a uniform distribution.
    """
    probs = {}
    n = len(actions)
    for infoset in infosets:
        values = q.get(infoset)
        if values is None or not np.any(np.isfinite(values)) or np.allclose(values, values[0]):
            probs[infoset] = [1.0 / n] * n
        else:
            row = [0.0] * n
            row[int(np.argmax(values))] = 1.0
            probs[infoset] = row
    return TabularPolicy(probs, actions, name=name)


# --------------------------------------------------------------- REINFORCE
class ReinforceOracle(Oracle):
    """Best response by REINFORCE (policy gradient) with a moving baseline.

    Each information set carries its own softmax logits. After every episode the
    logits are pushed along

    ``grad log pi(a | I) * (G - b)``

    where ``b`` is an exponentially weighted average of recent returns, used as a
    variance-reducing baseline. Unlike the Q-learning oracle this produces a
    genuinely stochastic policy, which matters in a game whose equilibrium
    strategies are mixed.
    """

    name = "reinforce"

    def __init__(
        self,
        learning_rate: float = 0.05,
        baseline_decay: float = 0.99,
        entropy_bonus: float = 0.0,
        temperature: float = 1.0,
    ):
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        self.entropy_bonus = entropy_bonus
        self.temperature = temperature

    def train(self, game, player, opponents, weights, budget, seed) -> OracleResult:
        start = time.perf_counter()
        rng = np.random.default_rng([seed, player, budget, 7])
        opponent = _make_opponent(opponents, weights)
        infosets = game.all_infosets(player)
        actions = game.legal_actions_at_infoset(infosets[0])
        n_actions = len(actions)
        action_index = {a: i for i, a in enumerate(actions)}
        logits = {i: np.zeros(n_actions) for i in infosets}

        baseline = 0.0
        returns: list[float] = []
        for _episode in range(budget):
            learner = _SoftmaxPolicy(logits, actions, self.temperature)
            policies = (learner, opponent) if player == 0 else (opponent, learner)
            trajectory, value = play_episode_trajectory(game, policies, rng, player)
            returns.append(value)
            advantage = value - baseline
            baseline = self.baseline_decay * baseline + (1.0 - self.baseline_decay) * value

            for infoset, action in trajectory:
                probs = _softmax(logits[infoset] / self.temperature)
                grad = -probs
                grad[action_index[action]] += 1.0
                update = advantage * grad
                if self.entropy_bonus:
                    with np.errstate(divide="ignore"):
                        log_p = np.where(probs > 0, np.log(probs), 0.0)
                    update += self.entropy_bonus * (-probs * (log_p + 1.0))
                logits[infoset] += self.learning_rate * update

        probs = {i: _softmax(logits[i] / self.temperature) for i in infosets}
        policy = TabularPolicy(probs, actions, name=f"br_pg{budget}")
        return OracleResult(
            policy=policy,
            episodes=budget,
            mean_return=float(np.mean(returns)) if returns else 0.0,
            final_return=_running_mean(returns),
            wall_time=time.perf_counter() - start,
            extra={"logits": {k: v.tolist() for k, v in logits.items()}},
        )


class _SoftmaxPolicy(Policy):
    name = "softmax"

    def __init__(self, logits: dict, actions: Sequence, temperature: float):
        self._logits = logits
        self._actions = tuple(actions)
        self._temperature = temperature
        self._n = len(actions)

    def action_probs(self, infoset: str, legal_actions: Sequence) -> np.ndarray:
        row = self._logits.get(infoset)
        if row is None:
            return np.full(self._n, 1.0 / self._n)
        return _softmax(row / self._temperature)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


# ------------------------------------------------------------ exact oracle
class ExactOracle(Oracle):
    """An *exact* best response, found by enumerating deterministic strategies.

    This is not a learning algorithm; it is the perfect-oracle limit. PSRO with
    this oracle is precisely the double-oracle algorithm, so it serves two
    purposes: it upper-bounds what any learned oracle can achieve, and it is the
    reference point for the oracle-quality experiment.
    """

    name = "exact"

    def __init__(self, calculators: dict[int, ExactBestResponse] | None = None):
        self._calculators = calculators if calculators is not None else {}

    def _calculator(self, game: ZeroSumGame, player: int) -> ExactBestResponse:
        if player not in self._calculators:
            self._calculators[player] = ExactBestResponse(game, player)
        return self._calculators[player]

    def train(self, game, player, opponents, weights, budget, seed) -> OracleResult:
        start = time.perf_counter()
        value, policy = self._calculator(game, player).best_response(opponents, weights)
        policy.name = f"br_exact{len(opponents)}"
        return OracleResult(
            policy=policy,
            episodes=0,
            mean_return=value,
            final_return=value,
            wall_time=time.perf_counter() - start,
            extra={"exact": True},
        )


# ------------------------------------------------------------------ registry
ORACLES: dict[str, type[Oracle]] = {
    "q_learning": TabularQLearningOracle,
    "reinforce": ReinforceOracle,
    "exact": ExactOracle,
}


def get_oracle(name: str, **kwargs) -> Oracle:
    try:
        return ORACLES[name](**kwargs)
    except KeyError:
        raise KeyError(f"unknown oracle {name!r}; available: {sorted(ORACLES)}") from None


# ------------------------------------------------- single-policy self-play
def self_play_baseline(
    game: ZeroSumGame,
    player: int,
    rounds: int = 8,
    budget: int = 5_000,
    seed: int = 0,
    oracle: Oracle | None = None,
) -> OracleResult:
    """The third baseline the specification asks for: a self-play learner.

    A *single-policy* self-play learner keeps one policy and repeatedly replaces
    it with a best response to itself::

        pi <- BR(pi)

    Nothing is remembered. That is the whole point of the baseline: it is the
    control condition against which population-based learning is judged, and
    the reason PSRO exists. In a game with non-transitive structure this
    iteration cycles rather than converging, which the returned history makes
    visible - the exploitability of the successive policies does not decrease
    monotonically, and often returns close to where it started.

    Returns
    -------
    OracleResult
        ``policy`` is the final iterate. ``extra['history']`` holds the
        exploitability of every intermediate policy, and ``extra['returned']``
        whether the iteration revisited a policy it had already produced.
    """
    from exact import ExactBestResponse

    start = time.perf_counter()
    opponent_seat = 1 - player
    current = TabularPolicy.uniform(game, opponent_seat, name="self_play_seed")
    # The exploiter sits in the OTHER seat: we are asking how much an opponent
    # can extract from the policy we just learned.
    calculator = ExactBestResponse(game, opponent_seat)
    oracle = oracle or TabularQLearningOracle()

    history: list[float] = []
    signatures: list[tuple] = []
    returned = False
    actions = game.legal_actions_at_infoset(game.all_infosets(player)[0])

    for round_index in range(rounds):
        result = oracle.train(game, player, [current], [1.0], budget,
                              seed=seed * 100 + round_index)
        learned = result.policy
        # How exploitable is the policy this round produced, on its own?
        history.append(calculator.best_response_value([learned], [1.0]))
        signature = tuple(
            tuple(np.round(learned.action_probs(i, actions), 6))
            for i in game.all_infosets(player)
        )
        if signature in signatures:
            returned = True
        signatures.append(signature)
        # The learner now faces itself: the same table, worn by the other seat.
        current = TabularPolicy(
            {infoset_other: learned.action_probs(infoset_self, actions)
             for infoset_self, infoset_other in zip(game.all_infosets(player),
                                                    game.all_infosets(opponent_seat))},
            actions, name=f"self_play_r{round_index}",
        )

    final = TabularPolicy(
        {i: learned.action_probs(i, actions) for i in game.all_infosets(player)},
        actions, name="self_play_learner",
    )
    return OracleResult(
        policy=final,
        episodes=rounds * budget,
        mean_return=float(np.mean(history)) if history else 0.0,
        final_return=history[-1] if history else 0.0,
        wall_time=time.perf_counter() - start,
        extra={"history": history, "returned": returned, "rounds": rounds},
    )
