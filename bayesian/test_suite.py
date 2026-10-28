import sys
import dataclasses
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from distributions import NormalNormal, DistributionList, RandomNumberGenerator
from hierarchical import (
    default_hierarchical_belief,
    StationarySpec,
    DriftingSpec,
    TimeDelayedSpec,
    PatchForagingSpec,
    TwoContextSpec,
    ThreeContextSpec,
    BernoulliSpec,
    LinearContextualSpec,
)
from hint_functions import _ALL_FUNCTIONS
from agents import HierarchicalHintAgent
from bandits import GaussianMAB, DriftingMAB, TimeDelayedMAB, PatchForagingMAB

# ─── shared helpers ──────────────────────────────────────────────────────────

N_ARMS = 5

PRIOR = NormalNormal(
    hyper_loc=np.full(N_ARMS, 50.0),
    hyper_stdev=np.full(N_ARMS, 10.0),
    stdev=np.full(N_ARMS, 5.0),
)


def _rng(seed=0):
    return RandomNumberGenerator(seed)


def _obs(actions, rewards, n_arms=N_ARMS):
    T = len(actions)
    obs = np.full((T, n_arms), np.nan)
    for t, (a, r) in enumerate(zip(actions, rewards)):
        obs[t, a] = r
    return obs


def _generic_history(T=12, seed=1, n_arms=N_ARMS):
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, n_arms, T)
    rewards = rng.normal(50.0, 5.0, T)
    obs = _obs(actions, rewards, n_arms)
    return actions, obs, np.arange(T)


def _belief():
    return default_hierarchical_belief(n_arms=N_ARMS)


# ─── 1. NormalNormal updates ─────────────────────────────────────────────────


class TestNormalNormal:
    def test_single_obs_posterior_mean(self):
        nn = NormalNormal(hyper_loc=50.0, hyper_stdev=10.0, stdev=5.0)
        updated = nn.update(np.array([80.0]))
        assert np.isclose(updated.hyper_loc, 74.0, atol=1e-10)

    def test_single_obs_posterior_stdev(self):
        # new_stdev = sqrt(1 / (1/100 + 1/25)) = sqrt(20)
        nn = NormalNormal(hyper_loc=50.0, hyper_stdev=10.0, stdev=5.0)
        updated = nn.update(np.array([80.0]))
        assert np.isclose(updated.hyper_stdev, np.sqrt(20.0), atol=1e-10)

    def test_two_obs_posterior_mean(self):
        nn = NormalNormal(hyper_loc=50.0, hyper_stdev=10.0, stdev=5.0)
        updated = nn.update(np.array([80.0, 60.0]))
        assert np.isclose(updated.hyper_loc, 6.1 / 0.09, atol=1e-8)

    def test_two_obs_posterior_stdev(self):
        # new_stdev = sqrt(1/0.09) = 10/3
        nn = NormalNormal(hyper_loc=50.0, hyper_stdev=10.0, stdev=5.0)
        updated = nn.update(np.array([80.0, 60.0]))
        assert np.isclose(updated.hyper_stdev, np.sqrt(1.0 / 0.09), atol=1e-8)

    def test_nan_obs_skipped(self):
        # x=[80, nan, 60] must give same result as x=[80, 60]
        nn = NormalNormal(hyper_loc=50.0, hyper_stdev=10.0, stdev=5.0)
        u1 = nn.update(np.array([80.0, 60.0]))
        u2 = nn.update(np.array([80.0, np.nan, 60.0]))
        assert np.isclose(u1.hyper_loc, u2.hyper_loc, atol=1e-10)
        assert np.isclose(u1.hyper_stdev, u2.hyper_stdev, atol=1e-10)

    def test_posterior_contracts(self):
        nn = NormalNormal(hyper_loc=50.0, hyper_stdev=10.0, stdev=5.0)
        updated = nn.update(np.array([60.0, 70.0, 80.0]))
        assert updated.hyper_stdev < nn.hyper_stdev

    def test_obs_stdev_unchanged(self):
        nn = NormalNormal(hyper_loc=50.0, hyper_stdev=10.0, stdev=5.0)
        updated = nn.update(np.array([60.0]))
        assert np.isclose(updated.stdev, 5.0)

    def test_distribution_list_per_arm_independence(self):
        # DistributionList slices obs by arm — arm 1 sees only NaN, so its posterior = prior
        dl = DistributionList(
            [
                NormalNormal(hyper_loc=30.0, hyper_stdev=10.0, stdev=5.0),
                NormalNormal(hyper_loc=70.0, hyper_stdev=10.0, stdev=5.0),
            ]
        )
        # obs shape (T=3, n_arms=2): arm 0 sees 80, arm 1 sees NaN every trial
        obs = np.array([[80.0, np.nan], [80.0, np.nan], [80.0, np.nan]])
        updated = dl.update(obs)
        assert updated[0].hyper_loc > 30.0
        assert np.isclose(updated[1].hyper_loc, 70.0, atol=1e-10)
        assert np.isclose(updated[1].hyper_stdev, 10.0, atol=1e-10)

    def test_distribution_list_update(self):
        dl = DistributionList(
            [
                NormalNormal(hyper_loc=50.0, hyper_stdev=10.0, stdev=5.0),
                NormalNormal(hyper_loc=50.0, hyper_stdev=10.0, stdev=5.0),
            ]
        )
        # obs shape (T=2, n_arms=2); arm 0 sees 80, arm 1 sees 30
        obs = np.array([[80.0, np.nan], [np.nan, 30.0]])
        updated = dl.update(obs)
        assert updated[0].hyper_loc > 50.0
        assert updated[1].hyper_loc < 50.0


# ─── 2. Type Spec log-marginals, sample_params, expected_rewards ─────────────


class TestTypeSpecs:
    def _call(self, spec, actions, obs, ts, extra_priors=None):
        return spec.compute_log_marginal(actions, obs, ts, PRIOR, extra_priors or {})

    # compute_log_marginal: finite float + expected posterior keys

    def test_stationary_finite(self):
        a, o, ts = _generic_history()
        lm, extra = self._call(StationarySpec(), a, o, ts)
        assert np.isfinite(lm)
        assert isinstance(extra, dict)

    def test_drifting_finite_and_eps_posterior(self):
        a, o, ts = _generic_history()
        lm, extra = self._call(
            DriftingSpec(), a, o, ts, {"eps_mean": 10.0, "eps_stdev": 5.0}
        )
        assert np.isfinite(lm)
        assert "eps_mean" in extra and "eps_stdev" in extra

    def test_time_delayed_finite_and_k_posterior(self):
        rng = np.random.default_rng(42)
        T, k = 15, 3
        actions = rng.integers(0, N_ARMS, T)
        obs = np.zeros((T, N_ARMS))  # zeros for unchosen, not NaN
        for t in range(k, T):
            a_delayed = actions[t - k]
            obs[t, a_delayed] = rng.normal(50.0, 5.0)
        k_vals = np.array([1, 2, 3, 4, 5])
        lm, extra = self._call(
            TimeDelayedSpec(),
            actions,
            obs,
            np.arange(T),
            {"k_values": k_vals, "k_probs": np.ones(5) / 5},
        )
        assert np.isfinite(lm)
        assert "k_probs" in extra
        assert np.isclose(extra["k_probs"].sum(), 1.0)

    def test_patch_foraging_finite_and_lambda_posterior(self):
        a, o, ts = _generic_history()
        lm, extra = self._call(
            PatchForagingSpec(), a, o, ts, {"lambda_mean": 0.3, "lambda_stdev": 0.12}
        )
        assert np.isfinite(lm)
        assert "lambda_mean" in extra and "lambda_stdev" in extra

    def test_two_context_finite_and_mu_posteriors(self):
        a, o, ts = _generic_history()
        ctx = {t: t % 2 for t in range(len(a))}
        lm, extra = self._call(TwoContextSpec(), a, o, ts, {"context_history": ctx})
        assert np.isfinite(lm)
        assert "mu_post_c0" in extra and "mu_post_c1" in extra

    def test_three_context_finite_and_mu_posteriors(self):
        a, o, ts = _generic_history()
        ctx = {t: t % 3 for t in range(len(a))}
        lm, extra = self._call(ThreeContextSpec(), a, o, ts, {"context_history": ctx})
        assert np.isfinite(lm)
        assert "mu_post_c0" in extra and "mu_post_c1" in extra and "mu_post_c2" in extra

    def test_bernoulli_finite_for_binary_obs(self):
        rng = np.random.default_rng(2)
        T = 12
        actions = rng.integers(0, N_ARMS, T)
        rewards = rng.choice([0.0, 100.0], T)
        obs = _obs(actions, rewards)
        lm, extra = self._call(BernoulliSpec(), actions, obs, np.arange(T))
        assert np.isfinite(lm)
        assert "alpha" in extra and "beta" in extra

    def test_bernoulli_neginf_for_non_binary_obs(self):
        # obs = 42.0 is neither 0 nor 100 → must return -inf
        actions = np.array([0])
        obs = np.full((1, N_ARMS), np.nan)
        obs[0, 0] = 42.0
        lm, _ = self._call(BernoulliSpec(), actions, obs, np.array([0]))
        assert lm == -np.inf

    def test_linear_contextual_finite_and_theta_posterior(self):
        a, o, ts = _generic_history()
        # trial_features=None → uses identity fallback inside spec
        lm, extra = self._call(LinearContextualSpec(), a, o, ts)
        assert np.isfinite(lm)
        assert "theta_mean" in extra and "theta_cov" in extra

    # expected_rewards_from_params: shape (n_arms,), all finite

    @pytest.mark.parametrize(
        "spec_cls,extra_params",
        [
            (StationarySpec, {}),
            (DriftingSpec, {"eps": 2.0}),
            (TimeDelayedSpec, {"delay_k": 3}),
            (PatchForagingSpec, {"lambda": 0.3}),
            (TwoContextSpec, {}),
            (ThreeContextSpec, {}),
            (BernoulliSpec, {}),
            (LinearContextualSpec, {"theta": np.ones(N_ARMS)}),
        ],
    )
    def test_expected_rewards_shape_and_finite(self, spec_cls, extra_params):
        spec = spec_cls()
        params = {"arm_means": np.linspace(30.0, 70.0, N_ARMS), **extra_params}
        r = spec.expected_rewards_from_params(params)
        assert r.shape == (N_ARMS,), f"got shape {r.shape}"
        assert np.all(np.isfinite(r))

    # sample_params: must include arm_means of correct length

    @pytest.mark.parametrize(
        "spec_cls,type_priors",
        [
            (StationarySpec, {}),
            (DriftingSpec, {"eps_mean": 10.0, "eps_stdev": 5.0}),
            (
                TimeDelayedSpec,
                {"k_values": np.array([1, 2, 3]), "k_probs": np.ones(3) / 3},
            ),
            (PatchForagingSpec, {"lambda_mean": 0.3, "lambda_stdev": 0.12}),
            (TwoContextSpec, {}),
            (ThreeContextSpec, {}),
            (BernoulliSpec, {}),
            (LinearContextualSpec, {}),
        ],
    )
    def test_sample_params_has_arm_means(self, spec_cls, type_priors):
        spec = spec_cls()
        params = spec.sample_params(PRIOR, type_priors, _rng())
        assert "arm_means" in params, f"{spec_cls.__name__} missing arm_means"
        assert len(params["arm_means"]) == N_ARMS
        assert np.all(np.isfinite(params["arm_means"]))


# ─── 3. HierarchicalBelief ───────────────────────────────────────────────────


class TestHierarchicalBelief:
    def test_deterministic(self):
        b = _belief()
        a, o, ts = _generic_history()
        p1 = b.update(a, o, ts).type_probs
        p2 = b.update(a, o, ts).type_probs
        assert np.array_equal(p1, p2)

    def test_type_probs_valid(self):
        b = _belief()
        a, o, ts = _generic_history()
        probs = b.update(a, o, ts).type_probs
        assert np.all(probs >= 0)
        assert np.all(np.isfinite(probs))
        assert np.isclose(probs.sum(), 1.0)

    def test_nan_guard_extreme_obs(self):
        # Extreme rewards should not produce NaN type_probs
        b = _belief()
        T = 8
        actions = np.zeros(T, dtype=int)
        obs = np.full((T, N_ARMS), np.nan)
        for t in range(T):
            obs[t, 0] = 1e15
        probs = b.update(actions, obs, np.arange(T)).type_probs
        assert np.all(np.isfinite(probs))
        assert np.isclose(probs.sum(), 1.0)

    def test_empty_history_returns_prior(self):
        b = _belief()
        updated = b.update(np.array([], dtype=int), np.zeros((0, N_ARMS)), np.array([]))
        assert np.array_equal(updated.type_probs, b.type_probs)

    def test_arm_posterior_moves_toward_observed_reward(self):
        b = _belief()
        prior_loc = np.atleast_1d(b.arm_mean_prior.hyper_loc).copy()
        T = 20
        actions = np.full(T, 4, dtype=int)
        obs = np.full((T, N_ARMS), np.nan)
        obs[:, 4] = 90.0
        updated = b.update(actions, obs, np.arange(T))
        new_loc = np.atleast_1d(updated.arm_mean_prior.hyper_loc)
        assert new_loc[4] > prior_loc[4], "arm 4 posterior should shift toward 90"
        for arm in range(N_ARMS - 1):
            assert abs(new_loc[arm] - prior_loc[arm]) < 5.0, (
                f"unchosen arm {arm} moved too much: {abs(new_loc[arm] - prior_loc[arm]):.2f}"
            )

    def test_unchosen_arm_posterior_unchanged(self):
        # If arm 0 is never chosen, its posterior mean should equal its prior mean
        b = _belief()
        T = 15
        actions = np.ones(T, dtype=int)  # always arm 1
        obs = np.full((T, N_ARMS), np.nan)
        obs[:, 1] = 60.0
        updated = b.update(actions, obs, np.arange(T))
        prior_loc0 = np.atleast_1d(b.arm_mean_prior.hyper_loc)[0]
        new_loc0 = np.atleast_1d(updated.arm_mean_prior.hyper_loc)[0]
        assert np.isclose(new_loc0, prior_loc0, atol=1e-8)

    def test_stationary_upweighted_for_constant_rewards(self):
        # Perfectly constant rewards per arm → stationary should rank in top 3
        b = _belief()
        T = 20
        actions = np.tile(np.arange(N_ARMS), T // N_ARMS + 1)[:T]
        obs = np.full((T, N_ARMS), np.nan)
        for t, a in enumerate(actions):
            obs[t, a] = float(a * 10 + 30)  # arm-specific constant reward
        updated = b.update(actions, obs, np.arange(T))
        stat_idx = next(
            i for i, s in enumerate(updated.type_specs) if s.name == "stationary"
        )
        rank = int(np.sum(updated.type_probs > updated.type_probs[stat_idx]))
        assert rank < 3, (
            f"stationary ranked {rank + 1}/8 with constant rewards, probs={updated.type_probs.round(3)}"
        )

    def test_bernoulli_upweighted_for_binary_rewards(self):
        # Only 0 and 100 obs → bernoulli should rank in top 3
        b = _belief()
        rng = np.random.default_rng(7)
        T = 20
        actions = rng.integers(0, N_ARMS, T)
        obs = np.full((T, N_ARMS), np.nan)
        for t, a in enumerate(actions):
            obs[t, a] = rng.choice([0.0, 100.0])
        updated = b.update(actions, obs, np.arange(T))
        bern_idx = next(
            i for i, s in enumerate(updated.type_specs) if s.name == "bernoulli"
        )
        rank = int(np.sum(updated.type_probs > updated.type_probs[bern_idx]))
        assert rank < 3, (
            f"bernoulli ranked {rank + 1}/8 with binary rewards, probs={updated.type_probs.round(3)}"
        )

    def test_context_history_dict_vs_list_equivalent(self):
        b = _belief()
        rng = np.random.default_rng(9)
        T = 12
        actions = rng.integers(0, N_ARMS, T)
        obs = np.full((T, N_ARMS), np.nan)
        for t, a in enumerate(actions):
            obs[t, a] = rng.normal(50, 5)
        ts = np.arange(T)
        ctx_list = [t % 2 for t in range(T)]
        ctx_dict = {t: c for t, c in enumerate(ctx_list)}

        def _with_ctx(ch):
            priors = {**b.type_specific_priors, "two_context": {"context_history": ch}}
            return dataclasses.replace(b, type_specific_priors=priors).update(
                actions, obs, ts
            )

        r_list = _with_ctx(ctx_list)
        r_dict = _with_ctx(ctx_dict)
        assert np.allclose(r_list.type_probs, r_dict.type_probs, atol=1e-10)


# ─── 4. Hint Functions ────────────────────────────────────────────────────────

N_SAMPLES = 400


def _rs(seed=0, n_arms=N_ARMS, n_samples=N_SAMPLES):
    return np.random.default_rng(seed).normal(50.0, 15.0, (n_arms, n_samples))


def _t_ids(type_idx, n_samples=N_SAMPLES):
    return np.full(n_samples, type_idx, dtype=int)


class TestHintFunctions:
    # ── 4a. No NaN, no ±inf ──────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "fn", _ALL_FUNCTIONS, ids=[f.__name__ for f in _ALL_FUNCTIONS]
    )
    def test_output_valid(self, fn):
        rs = _rs(seed=2)
        t_ids = np.zeros(N_SAMPLES, dtype=int)
        actions = np.array([0, 1, 2, 0, 3, 4, 2, 1])
        result = np.asarray(fn(t_ids, rs, params=None, actions=actions), dtype=float)
        assert result.shape == (N_SAMPLES,), (
            f"{fn.__name__}: wrong shape {result.shape}"
        )
        assert not np.any(np.isnan(result)), f"{fn.__name__}: NaN in output"
        assert not np.any(result == np.inf), f"{fn.__name__}: +inf in output"
        assert np.all(result <= 0.0), f"{fn.__name__}: positive weight found"
        finite = result[np.isfinite(result)]
        assert finite.size == 0 or finite.min() >= -1.0 - 1e-9, (
            f"{fn.__name__}: finite weights must lie in [-1, 0]"
        )

    # ── 4b. Correctness: specific crafted inputs ──────────────────────────────

    def test_max2_is_best_all_correct(self):
        from hint_functions import max2_is_best

        rs = np.ones((N_ARMS, N_SAMPLES)) * 40.0
        rs[-1, :] = 90.0  # arm n-1 always best
        result = max2_is_best(np.zeros(N_SAMPLES, dtype=int), rs)
        assert np.all(result == 0.0)

    def test_max2_is_best_all_wrong(self):
        from hint_functions import max2_is_best

        rs = np.ones((N_ARMS, N_SAMPLES)) * 40.0
        rs[0, :] = 90.0  # arm 0 always best
        result = max2_is_best(np.zeros(N_SAMPLES, dtype=int), rs)
        assert np.all(result == -np.inf)

    def test_min2_is_worst_all_correct(self):
        from hint_functions import min2_is_worst

        rs = np.ones((N_ARMS, N_SAMPLES)) * 60.0
        rs[0, :] = 10.0  # arm 0 always worst
        result = min2_is_worst(np.zeros(N_SAMPLES, dtype=int), rs)
        assert np.all(result == 0.0)

    def test_min2_is_worst_all_wrong(self):
        from hint_functions import min2_is_worst

        rs = np.ones((N_ARMS, N_SAMPLES)) * 60.0
        rs[-1, :] = 10.0  # arm n-1 worst, not arm 0
        result = min2_is_worst(np.zeros(N_SAMPLES, dtype=int), rs)
        assert np.all(result == -np.inf)

    def test_log_type_correct_type_gets_zero(self):
        from hint_functions import values_drift_small_each_trial, _DRIFT

        rs = _rs()
        assert np.all(values_drift_small_each_trial(_t_ids(_DRIFT), rs) == 0.0)
        assert np.all(
            values_drift_small_each_trial(_t_ids(0), rs) == -np.inf
        )  # stationary

    def test_max2_best_but_decreases_is_patch_gated(self):
        # now delegates to _switch_policy (patch rotation): finite within patch, -inf off-type
        from hint_functions import max2_best_but_decreases, _PATCH

        rs = _rs(seed=4)
        acts = np.array([0, 1, 2, 0])
        assert np.all(
            np.isfinite(max2_best_but_decreases(_t_ids(_PATCH), rs, actions=acts))
        )
        assert np.all(max2_best_but_decreases(_t_ids(0), rs, actions=acts) == -np.inf)

    def test_square_max2_best_circle_min2_best_ctx0(self):
        from hint_functions import square_max2_best_circle_min2_best, _TWO_CTX

        rs = np.ones((N_ARMS, N_SAMPLES)) * 40.0
        rs[-1, :] = 90.0  # arm n-1 always best (ctx 0 → wants arm n-1)
        result = square_max2_best_circle_min2_best(_t_ids(_TWO_CTX), rs, context=0)
        assert np.all(result == 0.0)

    def test_square_max2_best_circle_min2_best_ctx1(self):
        from hint_functions import square_max2_best_circle_min2_best, _TWO_CTX

        rs = np.ones((N_ARMS, N_SAMPLES)) * 40.0
        rs[0, :] = 90.0  # arm 0 always best (ctx 1 → wants arm 0)
        result = square_max2_best_circle_min2_best(_t_ids(_TWO_CTX), rs, context=1)
        assert np.all(result == 0.0)

    def test_square_max2_best_circle_min2_best_wrong_type(self):
        # Even with correct arm, wrong type → weight=-1
        from hint_functions import square_max2_best_circle_min2_best

        rs = np.ones((N_ARMS, N_SAMPLES)) * 40.0
        rs[-1, :] = 90.0
        result = square_max2_best_circle_min2_best(
            _t_ids(0), rs, context=0
        )  # type=stationary
        assert np.all(result == -np.inf)

    def test_square_G_circle_A_triangle_explore_ctx0(self):
        from hint_functions import square_G_circle_A_triangle_explore, _THREE_CTX

        rs = np.ones((N_ARMS, N_SAMPLES)) * 40.0
        rs[-1, :] = 90.0  # arm n-1 best (ctx 0/square → wants arm n-1)
        result = square_G_circle_A_triangle_explore(_t_ids(_THREE_CTX), rs, context=0)
        assert np.all(result == 0.0)

    def test_square_G_circle_A_triangle_explore_ctx1(self):
        from hint_functions import square_G_circle_A_triangle_explore, _THREE_CTX

        rs = np.ones((N_ARMS, N_SAMPLES)) * 40.0
        rs[0, :] = 90.0  # arm 0 best (ctx 1/circle → wants arm 0)
        result = square_G_circle_A_triangle_explore(_t_ids(_THREE_CTX), rs, context=1)
        assert np.all(result == 0.0)

    def test_switch_policy_all_weights_nonpositive(self):
        from hint_functions import alternate_good_buttons, _PATCH

        actions = np.zeros(10, dtype=int)  # always chose arm 0
        rs = np.ones((N_ARMS, N_SAMPLES)) * 40.0
        rs[4, :] = 90.0  # arm 4 is Thompson-best
        result = np.asarray(
            alternate_good_buttons(_t_ids(_PATCH), rs, actions=actions), dtype=float
        )
        assert np.all(result <= 0)

    def test_hint_upweights_correct_samples(self):
        from hint_functions import max2_is_best

        rng = np.random.default_rng(42)
        rs = rng.normal(50.0, 15.0, (N_ARMS, N_SAMPLES))
        weights = max2_is_best(np.zeros(N_SAMPLES, dtype=int), rs)
        correct_mask = np.argmax(rs, axis=0) == N_ARMS - 1
        if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
            assert weights[correct_mask].mean() > weights[~correct_mask].mean()

    def test_bernoulli_type_hint_correct(self):
        from hint_functions import probability_fixed_for_each_button, _BERN

        rs = _rs()
        assert np.all(probability_fixed_for_each_button(_t_ids(_BERN), rs) == 0.0)
        assert np.all(probability_fixed_for_each_button(_t_ids(0), rs) == -np.inf)

    def test_linear_contextual_type_hint(self):
        from hint_functions import three_numbers_combine_to_total, _LINCON

        rs = _rs()
        assert np.all(three_numbers_combine_to_total(_t_ids(_LINCON), rs) == 0.0)
        assert np.all(three_numbers_combine_to_total(_t_ids(0), rs) == -np.inf)

    def test_policy_hints_type_gated(self):
        # type-gated policy hints: finite weights within their target type(s), -inf elsewhere
        from hint_functions import (
            alternate_good_buttons,
            continue_trying_others_to_detect_changes,
            explore_then_stick,
            choose_button_few_trials_in_row,
            _PATCH,
            _DRIFT,
            _STAT,
            _BERN,
            _DELAY,
        )

        rs = _rs(seed=6)
        acts = np.array([0, 1, 2, 0, 3, 4, 2, 1])
        cases = [
            (alternate_good_buttons, {_PATCH}),
            (continue_trying_others_to_detect_changes, {_DRIFT}),
            (explore_then_stick, {_STAT, _BERN}),
            (choose_button_few_trials_in_row, {_DELAY}),
        ]
        for fn, ok_types in cases:
            for typ in range(8):
                r = np.asarray(fn(_t_ids(typ), rs, actions=acts), dtype=float)
                if typ in ok_types:
                    assert np.all(np.isfinite(r)), (
                        f"{fn.__name__} -inf within target type {typ}"
                    )
                else:
                    assert np.all(r == -np.inf), (
                        f"{fn.__name__} did not gate off-type {typ}"
                    )

    def test_type_mask_hints(self):
        # hints that reduce to a pure type mask: 0 for their type, -inf otherwise
        from hint_functions import (
            one_button_is_best,
            chance_of_0_even_if_good,
            first_pick_most_representative,
            one_number_leads_to_more_points,
            _STAT,
            _BERN,
            _PATCH,
            _LINCON,
        )

        rs = _rs(seed=7)
        acts = np.array([0, 1, 2])
        for fn, typ in [
            (one_button_is_best, _STAT),
            (chance_of_0_even_if_good, _BERN),
            (first_pick_most_representative, _PATCH),
            (one_number_leads_to_more_points, _LINCON),
        ]:
            assert np.all(fn(_t_ids(typ), rs, actions=acts) == 0.0)
            assert np.all(fn(_t_ids((typ + 1) % 8), rs, actions=acts) == -np.inf)


# ─── 5. HierarchicalHintAgent ────────────────────────────────────────────────


class TestHierarchicalHintAgent:
    def _agent(self, hint_fn, strength=5.0):
        return HierarchicalHintAgent(
            hierarchical_belief=_belief(),
            hint=hint_fn,
            weighted_samples=500,
            hint_strength=strength,
        )

    def test_action_probs_sums_to_one(self):
        from hint_functions import max2_is_best

        probs = self._agent(max2_is_best).action_probs(
            rng=_rng(), actions=np.array([], dtype=int), obs=None
        )
        assert np.isclose(probs.sum(), 1.0, atol=1e-6)
        assert np.all(probs >= 0)
        assert not np.any(np.isnan(probs))

    def test_action_probs_shape(self):
        from hint_functions import max2_is_best

        probs = self._agent(max2_is_best).action_probs(
            rng=_rng(1), actions=np.array([], dtype=int), obs=None
        )
        assert probs.shape == (N_ARMS,)

    def test_hint_shifts_prob_toward_best_arm(self):
        from hint_functions import max2_is_best

        agent = self._agent(max2_is_best, strength=20.0)
        probs = agent.action_probs(
            rng=_rng(2), actions=np.array([], dtype=int), obs=None
        )
        assert probs[N_ARMS - 1] > probs[0], (
            f"P(arm {N_ARMS - 1})={probs[-1]:.3f} not > P(arm 0)={probs[0]:.3f}"
        )

    def test_min2_hint_shifts_away_from_arm0(self):
        from hint_functions import min2_is_worst

        agent = self._agent(min2_is_worst, strength=20.0)
        probs = agent.action_probs(
            rng=_rng(3), actions=np.array([], dtype=int), obs=None
        )
        # arm 0 should not be the most-chosen arm
        assert np.argmax(probs) != 0, (
            f"min2_is_worst should reduce P(arm 0), got probs={probs.round(3)}"
        )

    def test_flat_hint_gives_nonzero_all_arms(self):
        def flat_hint(t_ids, rs, params=None, **kwargs):
            return np.zeros(rs.shape[1])

        agent = self._agent(flat_hint)
        probs = agent.action_probs(
            rng=_rng(4), actions=np.array([], dtype=int), obs=None
        )
        assert np.all(probs > 0), (
            f"flat hint: expected all probs > 0, got {probs.round(3)}"
        )

    def test_no_crash_empty_history(self):
        from hint_functions import explore_then_stick

        agent = self._agent(explore_then_stick)
        probs = agent.action_probs(
            rng=_rng(5), actions=np.array([], dtype=int), obs=None
        )
        assert probs.shape == (N_ARMS,)
        assert not np.any(np.isnan(probs))

    def test_sample_action_valid_arms(self):
        from hint_functions import max2_is_best

        agent = self._agent(max2_is_best)
        actions = agent.sample_action(
            obs=None, n=30, rng=_rng(6), actions=np.array([], dtype=int), timesteps=None
        )
        assert np.all((actions >= 0) & (actions < N_ARMS))

    def test_different_hints_give_different_distributions(self):
        from hint_functions import max2_is_best, min2_is_worst

        p1 = self._agent(max2_is_best, strength=15.0).action_probs(
            rng=_rng(7), actions=np.array([], dtype=int), obs=None
        )
        p2 = self._agent(min2_is_worst, strength=15.0).action_probs(
            rng=_rng(7), actions=np.array([], dtype=int), obs=None
        )
        # max2_is_best should put more weight on arm n-1; min2_is_worst on arm 0
        assert p1[-1] > p2[-1], (
            "max2_is_best should give higher P(arm n-1) than min2_is_worst"
        )


# ─── 6. Bandit Dynamics ───────────────────────────────────────────────────────


class TestBandits:
    def test_gaussian_reward_scalar_all_arms(self):
        b = GaussianMAB(
            arm_means=tuple(float(i * 15 + 20) for i in range(N_ARMS)),
            arm_stdevs=(5.0,) * N_ARMS,
        )
        rng = _rng()
        for a in range(N_ARMS):
            r = b.reward(a, rng)
            shape = np.shape(r)
            assert shape == () or (len(shape) == 0), (
                f"reward(a={a}) not scalar: shape={shape}"
            )

    def test_gaussian_str_does_not_raise_with_float_means(self):
        # Was raising ValueError with :d format — fixed to :.2f
        b = GaussianMAB(arm_means=(1.5, 2.7, 33.33), arm_stdevs=(0.5, 0.5, 0.5))
        s = str(b)
        assert "1.50" in s and "2.70" in s

    def test_drifting_means_change_over_trials(self):
        b = DriftingMAB(
            arm_means=(50.0,) * N_ARMS, arm_stdevs=(5.0,) * N_ARMS, drift_epsilon=10.0
        )
        rng = _rng(10)
        initial = np.array(b.arm_means, dtype=float).copy()
        for t in range(20):
            b.sample_rewards(t=t, action=t % N_ARMS, rng=rng)
        assert not np.allclose(initial, b._current_means, atol=0.5), (
            "arm means did not drift after 20 trials"
        )

    def test_drifting_reset_on_t_less_than_last(self):
        b = DriftingMAB(
            arm_means=(50.0,) * N_ARMS, arm_stdevs=(5.0,) * N_ARMS, drift_epsilon=5.0
        )
        rng = _rng(11)
        for t in range(10):
            b.sample_rewards(t=t, action=0, rng=rng)
        # Restart from t=0 → means should reset toward initial
        b.sample_rewards(t=0, action=0, rng=rng)
        assert np.allclose(b._current_means, b.arm_means, atol=20.0), (
            "drift did not reset on t=0 after forward run"
        )

    def test_time_delayed_first_k_trials_all_zeros(self):
        k = 3
        b = TimeDelayedMAB(
            arm_means=(50.0,) * N_ARMS, arm_stdevs=(5.0,) * N_ARMS, delay_k=k
        )
        rng = _rng(5)
        for t in range(k):
            rewards = b.sample_rewards(t=t, action=0, rng=rng)
            assert np.all(rewards == 0.0), (
                f"t={t}: expected all zeros before queue fills, got {rewards}"
            )

    def test_time_delayed_reward_appears_after_k(self):
        k = 3
        b = TimeDelayedMAB(
            arm_means=(50.0,) * N_ARMS, arm_stdevs=(5.0,) * N_ARMS, delay_k=k
        )
        rng = _rng(6)
        rewards_seen = []
        for t in range(k + 3):
            rewards = b.sample_rewards(t=t, action=0, rng=rng)
            rewards_seen.append(rewards.copy())
        # After the queue fills, at least one non-zero reward must appear
        post_fill = rewards_seen[k:]
        assert any(np.any(r != 0.0) for r in post_fill), (
            "No delayed rewards appeared after k trials"
        )

    def test_time_delayed_reset_clears_queue(self):
        b = TimeDelayedMAB(
            arm_means=(50.0,) * N_ARMS, arm_stdevs=(5.0,) * N_ARMS, delay_k=3
        )
        rng = _rng(7)
        for t in range(5):
            b.sample_rewards(t=t, action=0, rng=rng)
        b.reset()
        # After reset, queue should be empty → first reward is 0
        rewards = b.sample_rewards(t=0, action=0, rng=rng)
        assert np.all(rewards == 0.0), "queue not cleared after reset"

    def test_patch_foraging_chosen_arm_depletes(self):
        b = PatchForagingMAB(
            arm_means=(80.0,) * N_ARMS, arm_stdevs=(1.0,) * N_ARMS, change_rate=0.5
        )
        rng = _rng(8)
        initial = b._current_rewards[0]
        for t in range(6):
            b.sample_rewards(t=t, action=0, rng=rng)
        assert b._current_rewards[0] < initial, (
            f"chosen arm did not deplete: {b._current_rewards[0]:.2f} >= {initial:.2f}"
        )

    def test_patch_foraging_unchosen_arm_replenishes(self):
        # Deplete arm 1, then let it rest while choosing arm 0
        b = PatchForagingMAB(
            arm_means=(80.0,) * N_ARMS, arm_stdevs=(1.0,) * N_ARMS, change_rate=0.5
        )
        rng = _rng(9)
        for t in range(4):
            b.sample_rewards(t=t, action=1, rng=rng)
        depleted = b._current_rewards[1]
        for t in range(4, 10):
            b.sample_rewards(t=t, action=0, rng=rng)
        assert b._current_rewards[1] > depleted, (
            f"arm 1 did not replenish: {b._current_rewards[1]:.2f} vs depleted={depleted:.2f}"
        )

    def test_patch_foraging_replenishment_capped_at_initial(self):
        b = PatchForagingMAB(
            arm_means=(60.0, 60.0, 60.0, 60.0, 60.0),
            arm_stdevs=(1.0,) * N_ARMS,
            change_rate=0.3,
        )
        rng = _rng(12)
        # Choose arm 0 for a long time → arms 1-4 replenish but should cap at 60
        for t in range(30):
            b.sample_rewards(t=t, action=0, rng=rng)
        for arm in range(1, N_ARMS):
            assert b._current_rewards[arm] <= 60.0 + 1e-9, (
                f"arm {arm} exceeded initial mean: {b._current_rewards[arm]:.4f}"
            )


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
