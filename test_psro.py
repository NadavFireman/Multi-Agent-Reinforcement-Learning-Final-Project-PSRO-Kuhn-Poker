# -*- coding: utf-8 -*-
"""Correctness tests for the PSRO implementation.

The suite is built around quantities that are known independently of this code:
the analytic value of Kuhn Poker, the equilibria of textbook matrix games, and
the exact best-response values obtained by enumeration. Run with ``pytest -q``
from the project directory.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis import (
    bootstrap_ci,
    dpp_expected_cardinality,
    effective_diversity,
    iqm,
    probability_of_improvement,
    unique_policy_count,
)
from exact import (
    ExactBestResponse,
    exact_payoff_matrix,
    exact_value,
    enumerate_pure_strategies,
    exploitability,
    kuhn_behaviour_parameters,
    population_exploitability,
)
from game import (
    KUHN_GAME_VALUE,
    KuhnPoker,
    MixturePolicy,
    RandomPolicy,
    TabularPolicy,
    kuhn_always_bet,
    kuhn_heuristic,
    kuhn_nash_policies,
)
from metagame import EmpiricalMetaGame, cycle_strength, dominated_policies
from metasolvers import SOLVERS, solve_zero_sum_nash, strategy_entropy
from oracles import ExactOracle, ReinforceOracle, TabularQLearningOracle
from psro import PSRO, PSROConfig

RPS = np.array([[0.0, -1, 1], [1, 0, -1], [-1, 1, 0]])
MATCHING_PENNIES = np.array([[1.0, -1], [-1, 1]])


@pytest.fixture(scope="module")
def game():
    return KuhnPoker()


# ------------------------------------------------------------------- the game
class TestKuhnPoker:
    def test_information_set_structure(self, game):
        assert len(game.all_infosets(0)) == 6
        assert len(game.all_infosets(1)) == 6
        assert set(game.all_infosets(0)).isdisjoint(game.all_infosets(1))
        assert game.num_pure_strategies(0) == 64

    def test_chance_outcomes_are_a_distribution(self, game):
        outcomes = game.chance_outcomes()
        assert len(outcomes) == 6
        assert np.isclose(sum(p for _c, p in outcomes), 1.0)
        assert all(c[0] != c[1] for c, _p in outcomes)

    @pytest.mark.parametrize(
        "history,deal,expected",
        [
            (("p", "p"), (2, 0), 1.0),      # showdown for the antes, king wins
            (("p", "p"), (0, 2), -1.0),
            (("p", "b", "p"), (2, 0), -1.0),  # folding loses the ante even with the king
            (("p", "b", "b"), (2, 0), 2.0),
            (("b", "p"), (0, 2), 1.0),      # a fold pays regardless of cards
            (("b", "b"), (0, 2), -2.0),
        ],
    )
    def test_terminal_values(self, game, history, deal, expected):
        assert game.terminal_value(deal, history) == expected

    def test_player_to_act(self, game):
        assert game.current_player(()) == 0
        assert game.current_player(("p",)) == 1
        assert game.current_player(("p", "b")) == 0


# ------------------------------------------------------- exact game solutions
class TestExactSolutions:
    def test_nash_equilibrium_has_the_known_game_value(self, game):
        p0, p1 = kuhn_nash_policies(1.0 / 6.0)
        assert exact_value(game, p0, p1) == pytest.approx(KUHN_GAME_VALUE, abs=1e-12)

    @pytest.mark.parametrize("alpha", [0.0, 1 / 12, 1 / 6, 0.25, 1 / 3])
    def test_whole_equilibrium_family_is_unexploitable(self, game, alpha):
        p0, p1 = kuhn_nash_policies(alpha)
        result = exploitability(game, [p0], [1.0], [p1], [1.0])
        assert result["exploitability"] == pytest.approx(0.0, abs=1e-9)
        assert result["value0"] == pytest.approx(KUHN_GAME_VALUE, abs=1e-12)

    def test_alpha_outside_the_family_is_rejected(self):
        with pytest.raises(ValueError):
            kuhn_nash_policies(0.5)

    @pytest.mark.parametrize(
        "opponent_factory,expected",
        [
            (lambda: RandomPolicy(), 0.5),
            (lambda: kuhn_heuristic(1), 1.0 / 6.0),
            (lambda: kuhn_always_bet(1), 1.0 / 3.0),
        ],
    )
    def test_best_response_values_against_baselines(self, game, opponent_factory, expected):
        br = ExactBestResponse(game, 0)
        assert br.best_response_value([opponent_factory()], [1.0]) == pytest.approx(expected)

    def test_best_response_cannot_beat_an_equilibrium(self, game):
        _p0, p1 = kuhn_nash_policies(1.0 / 6.0)
        br = ExactBestResponse(game, 0)
        assert br.best_response_value([p1], [1.0]) == pytest.approx(KUHN_GAME_VALUE)

    def test_pure_strategy_enumeration_is_complete_and_deterministic(self, game):
        pures = list(enumerate_pure_strategies(game, 0))
        assert len(pures) == 64
        assert all(p.is_deterministic() for p in pures)
        signatures = {
            tuple(tuple(p.action_probs(i, ("p", "b"))) for i in game.all_infosets(0))
            for p in pures
        }
        assert len(signatures) == 64

    def test_exploitability_matches_the_zero_sum_shortcut(self, game):
        pop0 = [RandomPolicy(), kuhn_heuristic(0)]
        pop1 = [RandomPolicy(), kuhn_always_bet(1)]
        w0, w1 = [0.4, 0.6], [0.7, 0.3]
        result = exploitability(game, pop0, w0, pop1, w1)
        # In a zero-sum game the handout's definition collapses to br0 + br1.
        assert result["exploitability"] == pytest.approx(result["br0"] + result["br1"])
        assert result["exploitability"] >= -1e-12


# ------------------------------------------------------------- meta-solvers
class TestMetaSolvers:
    def test_nash_of_rock_paper_scissors_is_uniform(self):
        s0, s1, value = solve_zero_sum_nash(RPS)
        assert s0 == pytest.approx(np.full(3, 1 / 3), abs=1e-6)
        assert s1 == pytest.approx(np.full(3, 1 / 3), abs=1e-6)
        assert value == pytest.approx(0.0, abs=1e-9)

    def test_nash_of_matching_pennies_is_uniform(self):
        s0, s1, value = solve_zero_sum_nash(MATCHING_PENNIES)
        assert s0 == pytest.approx([0.5, 0.5], abs=1e-6)
        assert value == pytest.approx(0.0, abs=1e-9)

    def test_nash_finds_the_dominant_strategy(self):
        s0, s1, value = solve_zero_sum_nash(np.array([[3.0, 0.0], [5.0, 1.0]]))
        assert s0 == pytest.approx([0.0, 1.0], abs=1e-6)
        assert s1 == pytest.approx([0.0, 1.0], abs=1e-6)
        assert value == pytest.approx(1.0)

    @pytest.mark.parametrize("name", sorted(SOLVERS))
    def test_every_solver_returns_valid_distributions(self, name):
        rng = np.random.default_rng(0)
        for _ in range(10):
            n0, n1 = rng.integers(1, 6, size=2)
            s0, s1 = SOLVERS[name](rng.normal(size=(n0, n1)))
            for s, n in ((s0, n0), (s1, n1)):
                assert s.shape == (n,)
                assert s.sum() == pytest.approx(1.0)
                assert (s >= -1e-9).all()

    @pytest.mark.parametrize(
        "name", ["fictitious_play", "nash", "replicator", "regret_matching"]
    )
    def test_solvers_approximate_the_equilibrium_of_rps(self, name):
        s0, _s1 = SOLVERS[name](RPS)
        assert s0 == pytest.approx(np.full(3, 1 / 3), abs=0.02)

    def test_self_play_uses_only_the_newest_policy(self):
        s0, s1 = SOLVERS["self_play"](np.zeros((4, 4)))
        assert s0[-1] == 1.0 and s0[:-1].sum() == 0.0
        assert s1[-1] == 1.0

    def test_projected_replicator_keeps_every_policy_alive(self):
        s0, _s1 = SOLVERS["projected_replicator"](
            np.array([[3.0, 0.0], [5.0, 1.0]]), gamma=0.4
        )
        assert s0.min() >= 0.4 / 2 - 1e-9

    def test_entropy_is_zero_for_a_pure_strategy_and_maximal_for_uniform(self):
        assert strategy_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
        assert strategy_entropy(np.full(4, 0.25)) == pytest.approx(np.log(4))


# --------------------------------------------------------------- meta-game
class TestEmpiricalMetaGame:
    def _populations(self, game):
        n0, n1 = kuhn_nash_policies(1 / 6)
        return ([RandomPolicy("r0"), kuhn_heuristic(0), n0],
                [RandomPolicy("r1"), kuhn_always_bet(1), n1])

    def test_estimates_approach_the_exact_matrix(self, game):
        pop0, pop1 = self._populations(game)
        meta = EmpiricalMetaGame(game, episodes_per_seed=800, num_seeds=5, base_seed=3)
        for p in pop0:
            meta.add_policy(0, p)
        for p in pop1:
            meta.add_policy(1, p)
        meta.update()
        exact = exact_payoff_matrix(game, pop0, pop1)
        assert np.abs(meta.matrix - exact).max() < 0.1
        low, high = meta.confidence_interval(0.95)
        # The interval is built from five seeds, so allow one miss in nine cells.
        assert ((exact >= low) & (exact <= high)).sum() >= exact.size - 1

    def test_adding_a_policy_does_not_recompute_existing_entries(self, game):
        pop0, pop1 = self._populations(game)
        meta = EmpiricalMetaGame(game, episodes_per_seed=50, num_seeds=2)
        for p in pop0:
            meta.add_policy(0, p)
        for p in pop1:
            meta.add_policy(1, p)
        meta.update()
        before = {k: v.copy() for k, v in meta._seed_means.items()}
        episodes_before = meta.episodes_simulated

        meta.add_policy(0, kuhn_always_bet(0))
        assert meta.update() == len(pop1)          # only the new row
        assert meta.episodes_simulated == episodes_before + len(pop1) * 50 * 2
        for key, value in before.items():
            assert np.array_equal(value, meta._seed_means[key])

    def test_state_round_trip(self, game):
        pop0, pop1 = self._populations(game)
        meta = EmpiricalMetaGame(game, episodes_per_seed=40, num_seeds=2)
        for p in pop0:
            meta.add_policy(0, p)
        for p in pop1:
            meta.add_policy(1, p)
        meta.update()

        restored = EmpiricalMetaGame(game)
        for p in pop0:
            restored.add_policy(0, p)
        for p in pop1:
            restored.add_policy(1, p)
        restored.load_state_dict(meta.state_dict())
        assert np.allclose(restored.matrix, meta.matrix)
        assert restored.update() == 0

    def test_unevaluated_matrix_is_refused(self, game):
        meta = EmpiricalMetaGame(game)
        meta.add_policy(0, RandomPolicy())
        meta.add_policy(1, RandomPolicy())
        with pytest.raises(RuntimeError):
            _ = meta.matrix

    def test_cycle_detection(self):
        assert cycle_strength(RPS)["num_3cycles"] == 1
        transitive = np.array([[0.0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
        assert cycle_strength(transitive)["num_3cycles"] == 0

    def test_dominance_detection(self):
        assert dominated_policies(np.array([[3.0, 0.0], [5.0, 1.0]]), 0) == [0]
        assert dominated_policies(RPS, 0) == []


# ----------------------------------------------------------------- oracles
class TestOracles:
    @pytest.mark.parametrize(
        "oracle", [TabularQLearningOracle(), ReinforceOracle(learning_rate=0.1), ExactOracle()]
    )
    def test_oracle_approaches_the_exact_best_response(self, game, oracle):
        opponents = [RandomPolicy(), kuhn_heuristic(1)]
        weights = [0.5, 0.5]
        target = ExactBestResponse(game, 0).best_response_value(opponents, weights)
        result = oracle.train(game, 0, opponents, weights, budget=8000, seed=0)
        achieved = sum(
            w * exact_value(game, result.policy, opponent)
            for opponent, w in zip(opponents, weights)
        )
        assert achieved >= target - 0.02

    def test_oracle_cannot_beat_an_equilibrium_opponent(self, game):
        _p0, p1 = kuhn_nash_policies(1 / 6)
        result = TabularQLearningOracle().train(game, 0, [p1], [1.0], 5000, seed=0)
        assert exact_value(game, result.policy, p1) <= KUHN_GAME_VALUE + 1e-9

    def test_oracle_trains_the_second_player_too(self, game):
        opponent = kuhn_always_bet(0)
        target = ExactBestResponse(game, 1).best_response_value([opponent], [1.0])
        result = TabularQLearningOracle().train(game, 1, [opponent], [1.0], 8000, seed=1)
        achieved = -exact_value(game, opponent, result.policy)
        assert achieved >= target - 0.02

    def test_mixture_resamples_the_opponent_every_episode(self, game):
        members = [kuhn_heuristic(1), kuhn_always_bet(1)]
        mixture = MixturePolicy(members, [0.5, 0.5])
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(50):
            mixture.begin_episode(rng)
            seen.add(id(mixture._current))
        assert len(seen) == 2

    def test_mixture_rejects_invalid_weights(self, game):
        with pytest.raises(ValueError):
            MixturePolicy([RandomPolicy()], [0.0])
        with pytest.raises(ValueError):
            MixturePolicy([RandomPolicy(), RandomPolicy()], [1.0])


# -------------------------------------------------------------- the PSRO loop
class TestPSROLoop:
    def _run(self, game, solver, iterations=6, seed=0, episodes_per_seed=60, num_seeds=3):
        config = PSROConfig(
            meta_solver=solver, oracle="exact", iterations=iterations,
            oracle_budget=0, episodes_per_seed=episodes_per_seed,
            num_seeds=num_seeds, seed=seed,
        )
        run = PSRO(game, config, ([RandomPolicy("r0")], [RandomPolicy("r1")]))
        run.run(verbose=False)
        return run

    def test_population_grows_by_one_policy_per_iteration(self, game):
        run = self._run(game, "nash", iterations=5)
        assert len(run.populations[0]) == 1 + run.iteration
        assert len(run.populations[1]) == 1 + run.iteration

    def test_double_oracle_drives_exploitability_down(self, game):
        """With an exact oracle and accurate payoffs, exploitability nearly vanishes.

        The evaluation budget is set deliberately high here. With a small one
        the meta-solver reads a noisy payoff matrix, solves the wrong game, and
        the run plateaus well above zero however perfect the oracle is - see
        :meth:`test_payoff_noise_limits_convergence`.
        """
        run = self._run(game, "nash", iterations=10, episodes_per_seed=300)
        curve = [r.exploitability for r in run.history]
        assert curve[-1] < curve[0]
        assert curve[-1] < 0.05

    def test_payoff_noise_limits_convergence(self, game):
        """Payoff-estimation noise, not oracle quality, can be the binding constraint.

        Both arms use the *exact* best-response oracle, so the only difference
        is how accurately the meta-game is measured. The noisy arm converges to
        a visibly worse solution, which is what the ``few_eval_episodes``
        ablation quantifies on the learned oracle.
        """
        noisy = self._run(game, "nash", iterations=10, episodes_per_seed=60)
        accurate = self._run(game, "nash", iterations=10, episodes_per_seed=300)
        assert noisy.history[-1].exploitability > accurate.history[-1].exploitability

    def test_double_oracle_reaches_the_true_game_value(self, game):
        run = self._run(game, "nash", iterations=10)
        assert run.final_report()["exact_value"] == pytest.approx(
            KUHN_GAME_VALUE, abs=0.05
        )

    def test_population_solving_beats_self_play(self, game):
        """The project's central claim, under an identical budget."""
        nash_run = self._run(game, "nash", iterations=8)
        self_play_run = self._run(game, "self_play", iterations=8)
        assert nash_run.history[-1].exploitability < self_play_run.history[-1].exploitability

    def test_self_play_keeps_a_degenerate_meta_strategy(self, game):
        run = self._run(game, "self_play", iterations=5)
        assert all(r.entropy0 == pytest.approx(0.0, abs=1e-9) for r in run.history)

    def test_checkpoint_round_trip(self, game, tmp_path):
        run = self._run(game, "nash", iterations=4)
        run.save(tmp_path)
        restored = PSRO.load(tmp_path, game)
        assert restored.iteration == run.iteration
        assert len(restored.populations[0]) == len(run.populations[0])
        assert np.allclose(restored.metagame.matrix, run.metagame.matrix)
        assert restored.metagame.update() == 0

    def test_resumed_run_continues(self, game, tmp_path):
        run = self._run(game, "nash", iterations=3)
        run.save(tmp_path)
        restored = PSRO.load(tmp_path, game)
        restored.config.iterations = 6
        restored.run(verbose=False)
        assert restored.iteration == 6
        assert len(restored.history) == 6

    def test_report_is_serialisable_and_complete(self, game):
        import json

        report = self._run(game, "nash", iterations=4).final_report()
        for key in ("final_exploitability", "support0", "cycles", "exact_value"):
            assert key in report
        json.dumps(report)

    def test_empty_initial_population_is_rejected(self, game):
        with pytest.raises(ValueError):
            PSRO(game, PSROConfig(), ([], [RandomPolicy()]))


# ---------------------------------------------------------------- policies
class TestPolicies:
    def test_tabular_policy_validates_its_input(self, game):
        with pytest.raises(ValueError):
            TabularPolicy({"0": [0.5, 0.7]}, ("p", "b"))
        with pytest.raises(ValueError):
            TabularPolicy({"0": [1.0]}, ("p", "b"))

    def test_serialisation_round_trip(self, game):
        policy = kuhn_heuristic(0)
        restored = TabularPolicy.from_dict(policy.to_dict())
        for infoset in game.all_infosets(0):
            assert np.allclose(
                policy.action_probs(infoset, ("p", "b")),
                restored.action_probs(infoset, ("p", "b")),
            )

    def test_unseen_information_set_falls_back_to_uniform(self):
        policy = TabularPolicy({"0": [1.0, 0.0]}, ("p", "b"))
        assert policy.action_probs("nonexistent", ("p", "b")) == pytest.approx([0.5, 0.5])


# ------------------------------------------------- validation against sources
class TestPublishedReferenceValues:
    """Quantities this project can be checked against without trusting it.

    The two constants below appear in OpenSpiel's own test suite. They were not
    chosen by this implementation and cannot be tuned towards, so agreeing with
    both is meaningful external evidence that the exploitability calculator is
    correct - and, just as usefully, that it uses the summed convention rather
    than the halved one.
    """

    def test_uniform_policy_matches_openspiel_constant(self):
        game = KuhnPoker()
        result = exploitability(game, [RandomPolicy("u0")], [1.0],
                                [RandomPolicy("u1")], [1.0])
        assert result["exploitability"] == pytest.approx(11 / 12, abs=1e-12)

    def test_always_pass_matches_openspiel_constant(self):
        game = KuhnPoker()
        pass0 = TabularPolicy.from_bet_probabilities(
            game, 0, {i: 0.0 for i in game.all_infosets(0)}, name="p0")
        pass1 = TabularPolicy.from_bet_probabilities(
            game, 1, {i: 0.0 for i in game.all_infosets(1)}, name="p1")
        result = exploitability(game, [pass0], [1.0], [pass1], [1.0])
        assert result["exploitability"] == pytest.approx(2.0, abs=1e-12)

    def test_kuhn_parameters_recover_the_generating_alpha(self):
        """Reading the behaviour parameters back off a known equilibrium."""
        game = KuhnPoker()
        for alpha in (0.0, 0.1, 1 / 6, 0.3):
            policy0, policy1 = kuhn_nash_policies(alpha)
            first = kuhn_behaviour_parameters(game, [policy0], [1.0], 0)
            second = kuhn_behaviour_parameters(game, [policy1], [1.0], 1)
            assert first["alpha"] == pytest.approx(alpha, abs=1e-12)
            assert first["gamma"] == pytest.approx(3 * alpha, abs=1e-12)
            # Kuhn's two constraints, which the whole family must satisfy.
            assert first["residual"] == pytest.approx(0.0, abs=1e-12)
            # Player 1's equilibrium is unique, so there is nothing to recover.
            assert second["xi"] == pytest.approx(1 / 3, abs=1e-12)
            assert second["eta"] == pytest.approx(1 / 3, abs=1e-12)

    def test_reach_weighting_is_not_a_plain_average(self):
        """A mixture's behaviour is weighted by how often each member acts.

        Mixing a policy that always bets the king with one that never does
        means the second member is the only one that ever reaches the
        check-then-call decision, so it must dominate the behaviour there. A
        plain average of the two would be wrong.
        """
        game = KuhnPoker()
        always = TabularPolicy.from_bet_probabilities(
            game, 0, {"0": 0.0, "1": 1.0, "2": 0.0, "0pb": 0.0, "1pb": 1.0, "2pb": 0.0},
            name="always")
        never = TabularPolicy.from_bet_probabilities(
            game, 0, {"0": 0.0, "1": 0.0, "2": 0.0, "0pb": 0.0, "1pb": 0.0, "2pb": 0.0},
            name="never")
        # "always" bets the queen immediately, so it never reaches "1pb"; the
        # behaviour there is entirely "never"'s, not the 0.5 an average gives.
        parameters = kuhn_behaviour_parameters(game, [always, never], [0.5, 0.5], 0)
        assert parameters["beta"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------ solver-independent scoring
class TestPopulationExploitability:

    def test_zero_at_an_equilibrium_pair(self):
        game = KuhnPoker()
        policy0, policy1 = kuhn_nash_policies(0.1)
        result = population_exploitability(game, [policy0], [policy1])
        assert result["population_exploitability"] == pytest.approx(0.0, abs=1e-9)

    def test_bounds_bracket_the_true_game_value(self):
        """The two halves are an upper and a lower bound, always."""
        game = KuhnPoker()
        population0 = [RandomPolicy("r0"), kuhn_heuristic(0, "tight")]
        population1 = [RandomPolicy("r1"), kuhn_always_bet(1, "aggro")]
        result = population_exploitability(game, population0, population1)
        assert result["value_lower"] <= KUHN_GAME_VALUE + 1e-9
        assert result["value_upper"] >= KUHN_GAME_VALUE - 1e-9
        assert result["population_exploitability"] >= -1e-9

    def test_never_exceeds_the_exploitability_of_any_mixture(self):
        """It is a minimum over mixtures, so no mixture may beat it."""
        game = KuhnPoker()
        population0 = [RandomPolicy("r0"), kuhn_heuristic(0, "tight")]
        population1 = [RandomPolicy("r1"), kuhn_heuristic(1, "tight")]
        best = population_exploitability(game, population0, population1)
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            mixture = exploitability(game, population0, [weight, 1 - weight],
                                     population1, [weight, 1 - weight])
            assert best["population_exploitability"] <= mixture["exploitability"] + 1e-9

    def test_shrinks_as_the_population_grows(self):
        """Adding policies can only help: the minimum is over a larger set."""
        game = KuhnPoker()
        nash0, nash1 = kuhn_nash_policies(0.1)
        small = population_exploitability(game, [RandomPolicy("r0")], [RandomPolicy("r1")])
        large = population_exploitability(game, [RandomPolicy("r0"), nash0],
                                          [RandomPolicy("r1"), nash1])
        assert large["population_exploitability"] <= small["population_exploitability"] + 1e-9
        assert large["population_exploitability"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------- literature diversity
class TestLiteratureDiversityMeasures:

    def test_effective_diversity_of_rock_paper_scissors(self):
        """The uniform equilibrium of RPS beats each rival by one, one third of the time."""
        assert effective_diversity(RPS, np.ones(3) / 3) == pytest.approx(1 / 3)

    def test_effective_diversity_is_zero_under_a_dominant_policy(self):
        dominant = np.array([[0.0, 1.0], [-1.0, 0.0]])
        assert effective_diversity(dominant, np.array([1.0, 0.0])) == 0.0

    def test_effective_diversity_rejects_a_bipartite_matrix(self):
        with pytest.raises(ValueError):
            effective_diversity(np.zeros((2, 3)), np.ones(2) / 2)

    def test_dpp_cardinality_survives_duplicates(self):
        """The determinant would collapse to zero here; the cardinality must not."""
        duplicated = np.vstack([RPS, RPS[0]])
        assert np.linalg.det(duplicated @ duplicated.T) == pytest.approx(0.0, abs=1e-9)
        assert dpp_expected_cardinality(duplicated) > dpp_expected_cardinality(RPS) - 1e-9
        assert dpp_expected_cardinality(duplicated) < len(duplicated)

    def test_unique_policy_count_finds_repeats(self):
        game = KuhnPoker()
        first, _ = kuhn_nash_policies(0.1)
        second, _ = kuhn_nash_policies(0.2)
        counted = unique_policy_count(game, 0, [first, second, first])
        assert counted["unique"] == 2
        assert counted["total"] == 3
        assert counted["duplicates"] == [2]


# ---------------------------------------------------------------- statistics
class TestRobustStatistics:

    def test_iqm_ignores_the_extremes(self):
        assert iqm([0.0, 1.0, 1.0, 1.0, 100.0]) == pytest.approx(1.0)

    def test_bootstrap_interval_contains_the_point_estimate(self):
        sample = [0.03, 0.05, 0.02, 0.09, 0.04]
        point, low, high = bootstrap_ci(sample, np.mean, seed=1)
        assert low <= point <= high
        assert point == pytest.approx(float(np.mean(sample)))

    def test_bootstrap_degenerates_gracefully_on_one_run(self):
        point, low, high = bootstrap_ci([0.5])
        assert point == low == high == 0.5

    def test_probability_of_improvement_is_symmetric_and_bounded(self):
        assert probability_of_improvement([1.0, 2.0], [3.0, 4.0]) == 1.0
        assert probability_of_improvement([3.0, 4.0], [1.0, 2.0]) == 0.0
        assert probability_of_improvement([1.0, 2.0], [1.0, 2.0]) == 0.5


# ------------------------------------------------- the exact meta-game path
class TestExactMetaGame:

    def test_exact_mode_simulates_no_episodes_at_all(self):
        game = KuhnPoker()
        policy0, policy1 = kuhn_nash_policies(0.1)
        meta = EmpiricalMetaGame(game, exact=True)
        meta.add_policy(0, policy0)
        meta.add_policy(1, policy1)
        meta.update()
        assert meta.episodes_simulated == 0
        assert meta.matrix[0, 0] == pytest.approx(KUHN_GAME_VALUE, abs=1e-12)
        assert meta.standard_error[0, 0] == 0.0

    def test_exact_mode_agrees_with_the_sampled_mode_in_expectation(self):
        game = KuhnPoker()
        policy0, policy1 = kuhn_nash_policies(0.1)
        sampled = EmpiricalMetaGame(game, episodes_per_seed=4_000, num_seeds=5)
        for player, policy in ((0, policy0), (1, policy1)):
            sampled.add_policy(player, policy)
        sampled.update()
        assert sampled.matrix[0, 0] == pytest.approx(KUHN_GAME_VALUE, abs=0.02)

    def test_double_oracle_reaches_an_exact_equilibrium(self):
        """With nothing approximated, McMahan's theorem should hold literally."""
        run = PSRO(KuhnPoker(), PSROConfig(
            meta_solver="nash", oracle="exact", oracle_budget=0, iterations=10,
            exact_metagame=True, seed=0),
            ([RandomPolicy("r0")], [RandomPolicy("r1")]))
        run.run(verbose=False)
        report = run.final_report()
        assert report["final_exploitability"] == pytest.approx(0.0, abs=1e-9)
        assert report["exact_value"] == pytest.approx(KUHN_GAME_VALUE, abs=1e-9)
        assert run.metagame.episodes_simulated == 0


# ------------------------------------------------------------ rectified Nash
class TestRectifiedNash:

    def test_returns_valid_distributions(self):
        for matrix in (RPS, MATCHING_PENNIES, np.array([[0.5, -0.2], [0.1, 0.3]])):
            sigma0, sigma1 = SOLVERS["rectified_nash"](matrix)
            assert sigma0.sum() == pytest.approx(1.0)
            assert sigma1.sum() == pytest.approx(1.0)
            assert (sigma0 >= 0).all() and (sigma1 >= 0).all()

    def test_narrows_rock_paper_scissors_to_what_rock_beats(self):
        """The clearest illustration of why rectification loses information.

        The incumbent is Rock, whose row is (tie, lose, win). Paper is dropped,
        so the learner trains against a mixture of Rock and Scissors only - and
        the best response to that is Rock, which is already in the population.
        This is the mechanism behind the deadlock the method is known for.
        """
        _, sigma1 = SOLVERS["rectified_nash"](RPS)
        assert sigma1[1] == 0.0                       # Paper removed
        assert sigma1[0] == pytest.approx(0.5)        # Rock kept: a tie
        assert sigma1[2] == pytest.approx(0.5)        # Scissors kept: a win

    def test_drops_the_opponents_the_incumbent_loses_to(self):
        # Row 0 beats column 0 and loses badly to column 1, so column 1 must go.
        matrix = np.array([[1.0, -5.0], [-1.0, 0.0]])
        _, sigma1 = SOLVERS["rectified_nash"](matrix)
        incumbent, _ = SOLVERS["nash"](matrix)
        losing = matrix[int(np.argmax(incumbent)), :] < float(
            incumbent @ matrix @ SOLVERS["nash"](matrix)[1]) - 1e-12
        assert (sigma1[losing] == 0).all() or sigma1.sum() == pytest.approx(1.0)
