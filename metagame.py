# -*- coding: utf-8 -*-
"""The empirical meta-game (Part B of the project specification).

The meta-game is the matrix game whose *actions* are whole policies. Its entry
``M[j, k]`` is the expected return of player 0 when it plays its policy ``j``
against player 1's policy ``k``, estimated by simulation.

Design requirements this class satisfies:

* stores the policy population of every player;
* runs repeated matches for every required joint policy combination;
* estimates the expected return **and its standard error**;
* builds the payoff matrix;
* adds policies **without recomputing existing entries**;
* uses multiple episodes *and* multiple random seeds per estimate.

Each estimate is a mean over ``num_seeds`` independent blocks of
``episodes_per_seed`` episodes. Reporting the standard error *across seed
means* rather than across individual episodes is deliberate: it is the seed
variability that matters when comparing algorithms, and it is what the
confidence intervals in the report are built from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from exact import exact_value
from game import Policy, ZeroSumGame, evaluate_pair, sample_payoffs


class EmpiricalMetaGame:
    """Incrementally estimated payoff matrix over two policy populations.

    Parameters
    ----------
    game:
        The underlying two-player zero-sum game.
    episodes_per_seed:
        Episodes played per (pair, seed) block.
    num_seeds:
        Independent seed blocks per pair; the standard error is computed across
        their means.
    base_seed:
        Root of the reproducible seed hierarchy. The stream for a given pair and
        seed index is derived from ``(base_seed, j, k, seed_index)``, so an
        entry's value does not depend on the order in which entries were filled.
    """

    def __init__(
        self,
        game: ZeroSumGame,
        episodes_per_seed: int = 200,
        num_seeds: int = 5,
        base_seed: int = 0,
        exact: bool = False,
    ):
        if episodes_per_seed < 1 or num_seeds < 1:
            raise ValueError("need at least one episode and one seed per estimate")
        self.game = game
        # When exact, every entry is integrated out of the game tree instead of
        # being simulated. That removes the third and last approximation in
        # PSRO - McMahan's double-oracle theorem assumes exact payoffs, an
        # exact best response and enumerable strategies, and only with this
        # flag set does the implementation satisfy all three.
        self.exact = exact
        self.episodes_per_seed = episodes_per_seed
        self.num_seeds = num_seeds
        self.base_seed = base_seed
        self.populations: tuple[list[Policy], list[Policy]] = ([], [])
        # (j, k) -> array of per-seed mean returns for player 0
        self._seed_means: dict[tuple[int, int], np.ndarray] = {}
        self.episodes_simulated = 0

    # ------------------------------------------------------------ population
    def add_policy(self, player: int, policy: Policy) -> int:
        """Append a policy to a player's population and return its index."""
        self.populations[player].append(policy)
        return len(self.populations[player]) - 1

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.populations[0]), len(self.populations[1])

    def missing_entries(self) -> list[tuple[int, int]]:
        """Population pairs that have not been evaluated yet."""
        n0, n1 = self.shape
        return [
            (j, k)
            for j in range(n0)
            for k in range(n1)
            if (j, k) not in self._seed_means
        ]

    # ------------------------------------------------------------ estimation
    def update(self, verbose: bool = False) -> int:
        """Evaluate every missing pair. Returns the number of pairs simulated."""
        missing = self.missing_entries()
        for j, k in missing:
            self._seed_means[(j, k)] = self._estimate(j, k)
        if verbose and missing:
            print(
                f"    meta-game: filled {len(missing)} new entries "
                f"({len(missing) * self.episodes_per_seed * self.num_seeds:,} episodes)"
            )
        return len(missing)

    def _estimate(self, j: int, k: int) -> np.ndarray:
        policy0 = self.populations[0][j]
        policy1 = self.populations[1][k]
        if self.exact:
            return np.full(self.num_seeds, exact_value(self.game, policy0, policy1))
        means = np.empty(self.num_seeds)
        for s in range(self.num_seeds):
            rng = np.random.default_rng([self.base_seed, j, k, s])
            payoffs = self._play(policy0, policy1, rng)
            means[s] = payoffs.mean()
            self.episodes_simulated += self.episodes_per_seed
        return means

    def _play(self, policy0, policy1, rng) -> np.ndarray:
        """Draw one block of episode payoffs.

        Falls back to hand-by-hand simulation only for policies whose payoff
        distribution cannot be integrated out (a mixture, say); every population
        policy takes the fast path.
        """
        try:
            return sample_payoffs(self.game, policy0, policy1,
                                  self.episodes_per_seed, rng)
        except TypeError:
            return evaluate_pair(self.game, policy0, policy1,
                                 self.episodes_per_seed, rng)

    # --------------------------------------------------------------- readout
    @property
    def matrix(self) -> np.ndarray:
        """Estimated payoff matrix of player 0."""
        n0, n1 = self.shape
        out = np.full((n0, n1), np.nan)
        for (j, k), means in self._seed_means.items():
            if j < n0 and k < n1:
                out[j, k] = means.mean()
        if np.isnan(out).any():
            raise RuntimeError("payoff matrix has unevaluated entries; call update() first")
        return out

    @property
    def standard_error(self) -> np.ndarray:
        """Standard error of each entry, computed across seed means."""
        n0, n1 = self.shape
        out = np.full((n0, n1), np.nan)
        for (j, k), means in self._seed_means.items():
            if j < n0 and k < n1:
                out[j, k] = (
                    means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else 0.0
                )
        return out

    def confidence_interval(self, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Confidence band around the payoff matrix.

        The standard error is estimated from only ``num_seeds`` observations, so
        the half-width uses the Student-t quantile with ``num_seeds - 1`` degrees
        of freedom rather than the normal 1.96. With five seeds the correct
        multiplier is 2.78, and using 1.96 instead visibly under-covers.
        """
        from scipy import stats

        m, se = self.matrix, self.standard_error
        if self.num_seeds < 2:
            return m.copy(), m.copy()
        half_width = stats.t.ppf(0.5 + confidence / 2.0, df=self.num_seeds - 1) * se
        return m - half_width, m + half_width

    def entry_summary(self, j: int, k: int) -> dict:
        means = self._seed_means[(j, k)]
        return {
            "policy0": self.populations[0][j].name,
            "policy1": self.populations[1][k].name,
            "mean": float(means.mean()),
            "stderr": float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0,
            "seed_means": means.tolist(),
            "episodes": self.episodes_per_seed * self.num_seeds,
        }

    # -------------------------------------------------------- serialisation
    def state_dict(self) -> dict:
        """Everything needed to resume, except the policies themselves."""
        return {
            "episodes_per_seed": self.episodes_per_seed,
            "num_seeds": self.num_seeds,
            "base_seed": self.base_seed,
            "episodes_simulated": self.episodes_simulated,
            "shape": list(self.shape),
            "seed_means": {
                f"{j},{k}": means.tolist() for (j, k), means in self._seed_means.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.episodes_per_seed = state["episodes_per_seed"]
        self.num_seeds = state["num_seeds"]
        self.base_seed = state["base_seed"]
        self.episodes_simulated = state.get("episodes_simulated", 0)
        self._seed_means = {
            tuple(int(x) for x in key.split(",")): np.asarray(value, dtype=float)
            for key, value in state["seed_means"].items()
        }

    def save_matrix(self, path: str | Path) -> None:
        """Write the payoff matrix, its standard errors and the labels to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "row_policies": [p.name for p in self.populations[0]],
            "column_policies": [p.name for p in self.populations[1]],
            "payoff_matrix": self.matrix.tolist(),
            "standard_error": self.standard_error.tolist(),
            "episodes_per_entry": self.episodes_per_seed * self.num_seeds,
            "num_seeds": self.num_seeds,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------- diagnostics
def dominated_policies(matrix: np.ndarray, player: int, tolerance: float = 1e-9) -> list[int]:
    """Indices of policies that are (weakly) dominated by another single policy.

    For player 0 a row ``j`` is dominated when some other row does at least as
    well against **every** opponent policy and strictly better against one.
    """
    m = matrix if player == 0 else -matrix.T
    n = m.shape[0]
    dominated = []
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            if np.all(m[i] >= m[j] - tolerance) and np.any(m[i] > m[j] + tolerance):
                dominated.append(j)
                break
    return dominated


def cycle_strength(matrix: np.ndarray, tolerance: float = 1e-9) -> dict:
    """Measure non-transitivity of a head-to-head matrix.

    Builds the "who beats whom" relation and counts three-cycles ``a > b > c >
    a``. A transitive population - a strict pecking order - has none; a
    rock-paper-scissors structure has many, and a pure one scores 1.0.

    The input must be a genuine head-to-head matrix: antisymmetric with a zero
    diagonal, as produced by :func:`game.head_to_head_matrix`. Passing the
    bipartite meta-game matrix instead would compare four different policies at
    a time and the cycles read out of it would be meaningless, so the shape is
    checked rather than assumed.
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("cycle_strength expects a square head-to-head matrix")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-9):
        raise ValueError(
            "cycle_strength expects a head-to-head matrix with a zero diagonal; "
            "the bipartite meta-game matrix is not one - use "
            "game.head_to_head_matrix to build it"
        )
    if not np.allclose(matrix, -matrix.T, atol=1e-9):
        raise ValueError(
            "cycle_strength expects an antisymmetric head-to-head matrix"
        )

    beats = (matrix - matrix.T) > tolerance
    cycles = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                # Each unordered triple is inspected once, in both orientations.
                if (beats[a, b] and beats[b, c] and beats[c, a]) or (
                        beats[a, c] and beats[c, b] and beats[b, a]):
                    cycles.append((a, b, c))
    total = n * (n - 1) * (n - 2) // 6 if n >= 3 else 0
    return {
        "num_3cycles": len(cycles),
        "cycles": cycles,
        "cyclic_triple_fraction": (len(cycles) / total) if total else 0.0,
    }
