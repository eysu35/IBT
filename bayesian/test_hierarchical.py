import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hierarchical import default_hierarchical_belief
from distributions import RandomNumberGenerator


def _small_history(n_arms=5, T=8, seed=0):
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, n_arms, T)
    obs = np.full((T, n_arms), np.nan)
    for t, a in enumerate(actions):
        obs[t, a] = rng.normal(50, 10)
    return actions, obs, np.arange(T)


def _make_belief():
    return default_hierarchical_belief(n_arms=5)


# --- Item 1: determinism ---


def test_update_deterministic():
    b = _make_belief()
    actions, obs, ts = _small_history()
    p1 = b.update(actions, obs, ts).type_probs
    p2 = b.update(actions, obs, ts).type_probs
    assert np.array_equal(p1, p2), "update() is non-deterministic"


def test_type_probs_finite_and_valid():
    b = _make_belief()
    actions, obs, ts = _small_history()
    probs = b.update(actions, obs, ts).type_probs
    assert np.all(np.isfinite(probs)), "type_probs contains non-finite values"
    assert np.all(probs >= 0), "type_probs contains negative values"
    assert np.isclose(probs.sum(), 1.0), f"type_probs sums to {probs.sum()}, not 1"


# --- Item 2: NaN guard ---


def test_all_inf_likelihoods_give_neginf_not_nan():
    b = _make_belief()
    n_arms, T = 5, 6
    actions = np.zeros(T, dtype=int)
    # Rewards wildly inconsistent with any Gaussian prior (e.g. 1e15)
    obs = np.full((T, n_arms), np.nan)
    for t in range(T):
        obs[t, 0] = 1e15
    ts = np.arange(T)
    new_b = b.update(actions, obs, ts)
    assert np.all(np.isfinite(new_b.type_probs)), (
        "type_probs has non-finite after extreme obs"
    )
    assert np.isclose(new_b.type_probs.sum(), 1.0)


# --- Item 3: context_history dict vs list ---


def _two_context_belief():
    return default_hierarchical_belief(n_arms=5)


def _run_two_context(context_history_format, seed=7):
    b = _two_context_belief()
    rng = np.random.default_rng(seed)
    n_arms, T = 5, 10
    contexts_list = [t % 2 for t in range(T)]
    contexts_dict = {t: c for t, c in enumerate(contexts_list)}

    actions = rng.integers(0, n_arms, T)
    obs = np.full((T, n_arms), np.nan)
    for t, a in enumerate(actions):
        obs[t, a] = rng.normal(50, 10)
    ts = np.arange(T)

    ch = contexts_dict if context_history_format == "dict" else contexts_list

    import dataclasses

    priors = {
        **b.type_specific_priors,
        "two_context": {"context_history": ch},
        "three_context": {"context_history": ch},
    }
    b2 = dataclasses.replace(b, type_specific_priors=priors)
    return b2.update(actions, obs, ts).type_specific_posteriors.get("two_context", {})


def test_context_dict_vs_list_equivalent():
    post_dict = _run_two_context("dict")
    post_list = _run_two_context("list")
    for key in ("mu_post_c0", "mu_post_c1"):
        assert np.allclose(post_dict[key], post_list[key], atol=1e-10), (
            f"{key} differs between dict and list context_history"
        )


# --- Item 5: arm posterior moves toward observed rewards ---


def test_arm_posterior_direction():
    b = _make_belief()
    prior_loc = np.atleast_1d(b.arm_mean_prior.hyper_loc).copy()
    n_arms, T = 5, 20
    target_arm = 4
    # Always choose arm 4 with reward 80 (well above any prior mean)
    actions = np.full(T, target_arm, dtype=int)
    obs = np.full((T, n_arms), np.nan)
    for t in range(T):
        obs[t, target_arm] = 80.0
    ts = np.arange(T)
    new_b = b.update(actions, obs, ts)
    new_loc = np.atleast_1d(new_b.arm_mean_prior.hyper_loc)
    assert new_loc[target_arm] > prior_loc[target_arm], (
        "Arm posterior mean did not move toward observed high reward"
    )
    # Other arms (NaN obs) should not move much
    for a in range(n_arms):
        if a != target_arm:
            assert abs(new_loc[a] - prior_loc[a]) < 5.0, (
                f"Arm {a} posterior moved unexpectedly (unchosen arm)"
            )


# --- Item 4 (sampling smoke test) ---


def test_sample_type_and_params_smoke():
    b = _make_belief()
    actions, obs, ts = _small_history(T=5)
    updated = b.update(actions, obs, ts)
    rng = RandomNumberGenerator(99)
    samples = updated.sample_type_and_params(rng=rng, n=50)
    assert len(samples) == 50
    for type_idx, params in samples:
        arm_means = updated.type_specs[type_idx].expected_rewards_from_params(params)
        assert arm_means.shape == (5,), f"Wrong arm_means shape: {arm_means.shape}"
        assert np.all(np.isfinite(arm_means)), "Non-finite arm_means in sample"


def test_sample_reproducible_with_seed():
    b = _make_belief()
    actions, obs, ts = _small_history(T=5)
    updated = b.update(actions, obs, ts)
    s1 = updated.sample_type_and_params(rng=RandomNumberGenerator(42), n=10)
    s2 = updated.sample_type_and_params(rng=RandomNumberGenerator(42), n=10)
    types1 = [i for i, _ in s1]
    types2 = [i for i, _ in s2]
    assert types1 == types2, "sample_type_and_params not reproducible with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
