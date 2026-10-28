from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import dataclasses
import scipy.stats as stats

from distributions import (
    NormalNormal,
    DistributionList,
    default_rng,
    RandomNumberGenerator,
)


def _context_at(context_history, t: int, n_contexts: int) -> int:
    if context_history is None:
        return t % n_contexts
    if isinstance(context_history, dict):
        return int(context_history[t]) if t in context_history else t % n_contexts
    if 0 <= t < len(context_history):
        return int(context_history[t])
    return t % n_contexts


def _reward_logpdf(r: float, mu: float, pred_var: float) -> float:
    pv = max(float(pred_var), 1e-20)
    pred_std = np.sqrt(pv)
    return float(stats.norm.logpdf(r, loc=mu, scale=pred_std))


# Bandit type specification: defines likelihood and param sampling per type


class BanditTypeSpec(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute_log_marginal(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        timesteps: np.ndarray,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        pass

    @abstractmethod
    def sample_params(
        self,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
        rng: RandomNumberGenerator,
        type_specific_posteriors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def expected_rewards_from_params(self, params: Dict[str, Any]) -> np.ndarray:
        pass


class StationarySpec(BanditTypeSpec):
    @property
    def name(self) -> str:
        return "stationary"

    def compute_log_marginal(
        self,
        actions: np.ndarray,  # history of actions
        obs: np.ndarray,  # history of rewards
        timesteps: np.ndarray,  # history of timesteps
        arm_mean_prior: NormalNormal,  # prior distribution over arm means
        type_specific_priors: Dict[str, Any],  # prior over type parameters
    ) -> Tuple[float, Dict[str, Any]]:

        # initialize starting values for the prior
        stdev = np.atleast_1d(arm_mean_prior.stdev)  # std of the gaussian arms
        mu0 = np.atleast_1d(arm_mean_prior.hyper_loc)  # prior belief of arm means
        sigma0 = np.atleast_1d(
            arm_mean_prior.hyper_stdev
        )  # std of the arm mean beliefs (how uncertain in the means)

        # initialize posterior values for the arm means
        mu_post = np.array(mu0, dtype=float)
        sigma_sq = np.array(sigma0, dtype=float) ** 2

        # compute the log marginal likelihood w Normal-Normal conjugate update
        log_marg = 0.0
        for t in range(len(actions)):
            a_t = actions[t]
            r = obs[t, a_t]

            # predicted reward variance (sigma_sq is variance of mean, so pred_var = var(mean) + var(obs))
            pred_var = sigma_sq[a_t] + stdev[a_t] ** 2
            log_marg += _reward_logpdf(r, mu_post[a_t], pred_var)

            # Posterior update for arm a_t (conjugate Normal-Normal)
            prec_post = 1.0 / sigma_sq[a_t] + 1.0 / (stdev[a_t] ** 2)
            mu_post[a_t] = (
                mu_post[a_t] / sigma_sq[a_t] + r / (stdev[a_t] ** 2)
            ) / prec_post
            sigma_sq[a_t] = 1.0 / prec_post

        return float(log_marg), {}

    def sample_params(
        self,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
        rng: RandomNumberGenerator,
        type_specific_posteriors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Sample arm means from prior (in practice, use posterior given obs)
        mus = stats.norm.rvs(
            loc=arm_mean_prior.hyper_loc,
            scale=arm_mean_prior.hyper_stdev,
            random_state=rng.np,
        )
        return {"arm_means": np.atleast_1d(mus)}

    def expected_rewards_from_params(self, params: Dict[str, Any]) -> np.ndarray:
        return np.asarray(params["arm_means"], dtype=float)


class DriftingSpec(BanditTypeSpec):
    @property
    def name(self) -> str:
        return "drifting"

    def compute_log_marginal(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        timesteps: np.ndarray,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        eps_mean = float(type_specific_priors.get("eps_mean", 10.0))
        eps_stdev = float(type_specific_priors.get("eps_stdev", 5.0))
        n_eps = 10
        quantiles = (np.arange(n_eps) + 0.5) / n_eps
        eps_samples = np.maximum(
            stats.norm.ppf(quantiles, loc=eps_mean, scale=eps_stdev), 1e-6
        )

        stdev = np.atleast_1d(arm_mean_prior.stdev)
        mu0 = np.atleast_1d(arm_mean_prior.hyper_loc)
        sigma0 = np.atleast_1d(arm_mean_prior.hyper_stdev)

        e_lls = np.full(n_eps, -np.inf)
        for i, eps in enumerate(eps_samples):
            mu_post = np.array(mu0, dtype=float)
            sigma_sq = np.array(sigma0, dtype=float) ** 2
            ll = 0.0
            for t in range(len(actions)):
                a_t = actions[t]
                r = obs[t, a_t]

                pred_var = sigma_sq[a_t] + stdev[a_t] ** 2
                ll += _reward_logpdf(r, mu_post[a_t], pred_var)

                prec_post = 1.0 / sigma_sq[a_t] + 1.0 / (stdev[a_t] ** 2)
                mu_post[a_t] = (
                    mu_post[a_t] / sigma_sq[a_t] + r / (stdev[a_t] ** 2)
                ) / prec_post
                sigma_sq[a_t] = 1.0 / prec_post

                sigma_sq = sigma_sq + (eps**2)
            e_lls[i] = ll

        # So non-finite log-likelihoods don't break logaddexp / exp
        ll_per_eps = np.where(np.isfinite(e_lls), e_lls, -np.inf)

        max_ll = np.max(ll_per_eps)
        if not np.isfinite(max_ll):
            log_marg = -np.inf
        else:
            log_marg = np.log(np.mean(np.exp(ll_per_eps - max_ll))) + max_ll

        # Posterior over eps: importance-weighted moment match on samples; return updated belief
        if np.isfinite(max_ll):
            weights = np.exp(ll_per_eps - max_ll)
            weights = weights / (weights.sum() + 1e-300)
            new_mean = max(float(np.sum(weights * eps_samples)), 1e-6)
            new_var = float(np.sum(weights * (eps_samples - new_mean) ** 2))
            new_stdev = float(np.sqrt(max(new_var, 1e-20)))
        else:
            new_mean = eps_mean
            new_stdev = eps_stdev

        extra = {
            "eps_mean": new_mean,
            "eps_stdev": new_stdev,
        }
        return (float(log_marg), extra)

    def sample_params(
        self,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
        rng: RandomNumberGenerator,
        type_specific_posteriors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mus = stats.norm.rvs(
            loc=arm_mean_prior.hyper_loc,
            scale=arm_mean_prior.hyper_stdev,
            random_state=rng.np,
        )
        # Sample epsilon from Gaussian: use posterior (updated in extra) if available, else prior
        if type_specific_posteriors and "eps_mean" in type_specific_posteriors:
            eps_mean = float(type_specific_posteriors["eps_mean"])
            eps_stdev = float(type_specific_posteriors["eps_stdev"])
        else:
            eps_mean = float(type_specific_priors.get("eps_mean", 10.0))
            eps_stdev = float(type_specific_priors.get("eps_stdev", 5.0))
        eps = np.maximum(
            stats.norm.rvs(loc=eps_mean, scale=eps_stdev, random_state=rng.np), 1e-6
        )
        return {"arm_means": np.atleast_1d(mus), "eps": eps}

    def expected_rewards_from_params(self, params: Dict[str, Any]) -> np.ndarray:
        return np.asarray(params["arm_means"], dtype=float)


class TimeDelayedSpec(BanditTypeSpec):
    @property
    def name(self) -> str:
        return "time_delayed"

    def compute_log_marginal(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        timesteps: np.ndarray,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        # Same structure as stationary/drifting: running (mu_post, sigma_sq) per arm per k.
        k_vals = type_specific_priors.get("k_values", np.array([1, 2, 3, 4, 5]))
        k_probs = type_specific_priors.get(
            "k_probs", np.ones(len(k_vals)) / len(k_vals)
        )

        stdev = np.atleast_1d(arm_mean_prior.stdev)
        mu0 = np.atleast_1d(arm_mean_prior.hyper_loc)
        sigma0 = np.atleast_1d(arm_mean_prior.hyper_stdev)

        k_lls = np.full(len(k_vals), -np.inf)
        total_ll = -np.inf

        for i, k in enumerate(k_vals):
            if k > len(actions) - 1:
                continue
            log_marg = 0.0
            # Observation stream for this k: (t, a_delayed, r) where r is a delivered reward.
            obs_list = []
            for t in range(k, len(actions)):
                a_delayed = actions[t - k]  # pairs action t-k
                r = obs[t, a_delayed]  # reward delivered at t (for pull at t-k)
                obs_list.append((t, a_delayed, r))

            mu_post = np.array(mu0, dtype=float)
            sigma_sq = np.array(sigma0, dtype=float) ** 2

            for t, a_t, r in obs_list:
                # predicted reward variance
                pred_var = sigma_sq[a_t] + stdev[a_t] ** 2
                log_marg += _reward_logpdf(r, mu_post[a_t], pred_var)

                # Posterior update for arm a_t (conjugate Normal-Normal)
                prec_post = 1.0 / sigma_sq[a_t] + 1.0 / (stdev[a_t] ** 2)
                mu_post[a_t] = (
                    mu_post[a_t] / sigma_sq[a_t] + r / (stdev[a_t] ** 2)
                ) / prec_post
                sigma_sq[a_t] = 1.0 / prec_post

            # keep track of the log likelihood for each k value
            k_lls[i] = log_marg
            term = np.log(k_probs[i] + 1e-300) + log_marg
            term = np.where(np.isfinite(term), term, -np.inf)
            total_ll = np.logaddexp(total_ll, term)

        # Posterior over k: P(k|data) ∝ P(data|k) P(k)
        ll_per_k = np.where(np.isfinite(k_lls), k_lls, -np.inf)
        log_post = np.log(k_probs + 1e-300) + ll_per_k
        max_ll = np.max(log_post)
        if np.isfinite(max_ll):
            log_post = log_post - max_ll
            k_probs_post = np.exp(log_post) / np.exp(log_post).sum()
        else:
            k_probs_post = np.array(k_probs, dtype=float)

        extra = {
            "k_values": k_vals,
            "k_probs": k_probs_post,
        }
        return (float(total_ll), extra)

    def sample_params(
        self,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
        rng: RandomNumberGenerator,
        type_specific_posteriors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mus = stats.norm.rvs(
            loc=arm_mean_prior.hyper_loc,
            scale=arm_mean_prior.hyper_stdev,
            random_state=rng.np,
        )
        if type_specific_posteriors and "k_probs" in type_specific_posteriors:
            k_values = type_specific_posteriors["k_values"]
            k_probs = type_specific_posteriors["k_probs"]
        else:
            k_values = np.atleast_1d(
                type_specific_priors.get("k_values", np.array([1, 2, 3, 4, 5]))
            )
            k_probs = type_specific_priors.get(
                "k_probs", np.ones(len(k_values)) / len(k_values)
            )
        ki = rng.np.choice(len(k_values), p=np.atleast_1d(k_probs))
        k = int(k_values[ki])

        return {"arm_means": np.atleast_1d(mus), "delay_k": k}

    def expected_rewards_from_params(self, params: Dict[str, Any]) -> np.ndarray:
        return np.asarray(params["arm_means"], dtype=float)


class PatchForagingSpec(BanditTypeSpec):
    @property
    def name(self) -> str:
        return "patch_foraging"

    def compute_log_marginal(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        timesteps: np.ndarray,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        lam_mean = type_specific_priors.get("lambda_mean", 0.3)
        lam_stdev = type_specific_priors.get("lambda_stdev", 0.12)
        n_lam_samples = 40
        quantiles = (np.arange(n_lam_samples) + 0.5) / n_lam_samples
        lam_samples = np.maximum(
            stats.norm.ppf(quantiles, loc=lam_mean, scale=lam_stdev), 1e-6
        )

        stdev = np.atleast_1d(arm_mean_prior.stdev)
        mu0 = np.atleast_1d(arm_mean_prior.hyper_loc)
        sigma0 = np.atleast_1d(arm_mean_prior.hyper_stdev)
        l_lls = np.full(len(lam_samples), -np.inf)
        for i, lam in enumerate(lam_samples):
            mu_post = np.array(mu0, dtype=float)
            sigma_sq = np.array(sigma0, dtype=float) ** 2
            gt_means_est = np.array(mu0, dtype=float)

            ll = 0.0
            for t in range(len(actions)):
                a_t = int(actions[t])
                r = obs[t, a_t]
                # Score under current state
                pred_var = sigma_sq[a_t] + stdev[a_t] ** 2
                ll += _reward_logpdf(r, mu_post[a_t], pred_var)

                prec_post = 1.0 / sigma_sq[a_t] + 1.0 / (stdev[a_t] ** 2)
                mu_post[a_t] = (
                    mu_post[a_t] / sigma_sq[a_t] + r / (stdev[a_t] ** 2)
                ) / prec_post
                sigma_sq[a_t] = 1.0 / prec_post

                # Update equilibrium estimate for chosen arm (post-update mean ≈ true mean).
                gt_means_est[a_t] = mu_post[a_t]

                mu_post[a_t] = max(0.0, mu_post[a_t] * np.exp(-lam))
                sigma_sq[a_t] = sigma_sq[a_t] * (np.exp(-2 * lam))
                other = np.ones(len(mu_post), dtype=bool)
                other[a_t] = False
                mu_post[other] = np.minimum(
                    gt_means_est[other], mu_post[other] * np.exp(lam)
                )
                sigma_sq[other] = np.minimum(
                    sigma_sq[other] * np.exp(2 * lam), sigma0[other] ** 2
                )

            l_lls[i] = ll

        ll_per_lam = np.where(np.isfinite(l_lls), l_lls, -np.inf)

        max_ll = np.max(ll_per_lam)
        if not np.isfinite(max_ll):
            log_marg = -np.inf
        else:
            log_marg = np.log(np.mean(np.exp(ll_per_lam - max_ll))) + max_ll

        # Posterior over lambda: importance-weighted moment match; return updated belief
        if np.isfinite(max_ll):
            weights = np.exp(ll_per_lam - max_ll)
            weights = weights / (weights.sum() + 1e-300)
            lam_mean_post = max(float(np.sum(weights * lam_samples)), 1e-6)
            lam_var_post = float(np.sum(weights * (lam_samples - lam_mean_post) ** 2))
            lam_stdev_post = float(np.sqrt(max(lam_var_post, 1e-20)))
        else:
            lam_mean_post = lam_mean
            lam_stdev_post = lam_stdev

        extra = {
            "lambda_mean": lam_mean_post,
            "lambda_stdev": lam_stdev_post,
        }
        return (float(log_marg), extra)

    def sample_params(
        self,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
        rng: RandomNumberGenerator,
        type_specific_posteriors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mus = stats.norm.rvs(
            loc=arm_mean_prior.hyper_loc,
            scale=arm_mean_prior.hyper_stdev,
            random_state=rng.np,
        )
        if type_specific_posteriors and "lambda_mean" in type_specific_posteriors:
            lam_mean = float(type_specific_posteriors["lambda_mean"])
            lam_stdev = float(type_specific_posteriors["lambda_stdev"])
        else:
            lam_mean = float(type_specific_priors.get("lambda_mean", 0.3))
            lam_stdev = float(type_specific_priors.get("lambda_stdev", 0.12))
        lam = np.maximum(
            stats.norm.rvs(loc=lam_mean, scale=lam_stdev, random_state=rng.np), 1e-6
        )
        return {"arm_means": np.atleast_1d(mus), "lambda": lam}

    def expected_rewards_from_params(self, params: Dict[str, Any]) -> np.ndarray:
        return np.asarray(params["arm_means"], dtype=float)


class TwoContextSpec(BanditTypeSpec):
    @property
    def name(self) -> str:
        return "two_context"

    def compute_log_marginal(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        timesteps: np.ndarray,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        stdev = np.atleast_1d(arm_mean_prior.stdev)
        mu0 = np.atleast_1d(arm_mean_prior.hyper_loc)
        sigma0 = np.atleast_1d(arm_mean_prior.hyper_stdev)

        context_history = type_specific_priors.get("context_history")

        # Separate beliefs per context (2 contexts)
        mu_post = {0: np.array(mu0, dtype=float), 1: np.array(mu0, dtype=float)}
        sigma_sq = {
            0: np.array(sigma0, dtype=float) ** 2,
            1: np.array(sigma0, dtype=float) ** 2,
        }

        log_marg = 0.0
        n_contexts = 2
        for t in range(len(actions)):
            c_t = _context_at(context_history, t, n_contexts)

            a_t = actions[t]
            r = obs[t, a_t]

            pred_var = sigma_sq[c_t][a_t] + stdev[a_t] ** 2
            log_marg += _reward_logpdf(r, mu_post[c_t][a_t], pred_var)
            prec_post = 1.0 / sigma_sq[c_t][a_t] + 1.0 / (stdev[a_t] ** 2)
            mu_post[c_t][a_t] = (
                mu_post[c_t][a_t] / sigma_sq[c_t][a_t] + r / (stdev[a_t] ** 2)
            ) / prec_post
            sigma_sq[c_t][a_t] = 1.0 / prec_post

        extra = {
            "mu_post_c0": mu_post[0],
            "sigma_sq_c0": sigma_sq[0],
            "mu_post_c1": mu_post[1],
            "sigma_sq_c1": sigma_sq[1],
        }
        return float(log_marg), extra

    def sample_params(
        self,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
        rng: RandomNumberGenerator,
        type_specific_posteriors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Determine which context the next trial is in
        if type_specific_posteriors and "next_context" in type_specific_posteriors:
            ctx = int(type_specific_posteriors["next_context"])
        else:
            ctx = 0

        # Sample arm means from the current context's posterior (or prior)
        key_mu = f"mu_post_c{ctx}"
        key_sig = f"sigma_sq_c{ctx}"
        if type_specific_posteriors and key_mu in type_specific_posteriors:
            mu = np.atleast_1d(type_specific_posteriors[key_mu])
            sig = np.sqrt(np.atleast_1d(type_specific_posteriors[key_sig]))
            arm_means = stats.norm.rvs(loc=mu, scale=sig, random_state=rng.np)
        else:
            arm_means = np.atleast_1d(
                stats.norm.rvs(
                    loc=arm_mean_prior.hyper_loc,
                    scale=arm_mean_prior.hyper_stdev,
                    random_state=rng.np,
                )
            )

        return {"arm_means": arm_means}

    def expected_rewards_from_params(self, params: Dict[str, Any]) -> np.ndarray:
        return np.asarray(params["arm_means"], dtype=float)


class ThreeContextSpec(BanditTypeSpec):
    @property
    def name(self) -> str:
        return "three_context"

    def compute_log_marginal(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        timesteps: np.ndarray,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        stdev = np.atleast_1d(arm_mean_prior.stdev)
        mu0 = np.atleast_1d(arm_mean_prior.hyper_loc)
        sigma0 = np.atleast_1d(arm_mean_prior.hyper_stdev)

        context_history = type_specific_priors.get("context_history")

        # Separate beliefs per context (3 contexts)
        n_contexts = 3
        mu_post = {c: np.array(mu0, dtype=float) for c in range(n_contexts)}
        sigma_sq = {c: np.array(sigma0, dtype=float) ** 2 for c in range(n_contexts)}

        log_marg = 0.0
        for t in range(len(actions)):
            c_t = _context_at(context_history, t, n_contexts)

            a_t = actions[t]
            r = obs[t, a_t]

            pred_var = sigma_sq[c_t][a_t] + stdev[a_t] ** 2
            log_marg += _reward_logpdf(r, mu_post[c_t][a_t], pred_var)
            prec_post = 1.0 / sigma_sq[c_t][a_t] + 1.0 / (stdev[a_t] ** 2)
            mu_post[c_t][a_t] = (
                mu_post[c_t][a_t] / sigma_sq[c_t][a_t] + r / (stdev[a_t] ** 2)
            ) / prec_post
            sigma_sq[c_t][a_t] = 1.0 / prec_post

        extra = {}
        for c in range(n_contexts):
            extra[f"mu_post_c{c}"] = mu_post[c]
            extra[f"sigma_sq_c{c}"] = sigma_sq[c]
        return float(log_marg), extra

    def sample_params(
        self,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
        rng: RandomNumberGenerator,
        type_specific_posteriors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Determine which context the next trial is in
        if type_specific_posteriors and "next_context" in type_specific_posteriors:
            ctx = int(type_specific_posteriors["next_context"])
        else:
            ctx = 0

        # Sample arm means from the current context's posterior (or prior)
        key_mu = f"mu_post_c{ctx}"
        key_sig = f"sigma_sq_c{ctx}"
        if type_specific_posteriors and key_mu in type_specific_posteriors:
            mu = np.atleast_1d(type_specific_posteriors[key_mu])
            sig = np.sqrt(np.atleast_1d(type_specific_posteriors[key_sig]))
            arm_means = stats.norm.rvs(loc=mu, scale=sig, random_state=rng.np)
        else:
            arm_means = np.atleast_1d(
                stats.norm.rvs(
                    loc=arm_mean_prior.hyper_loc,
                    scale=arm_mean_prior.hyper_stdev,
                    random_state=rng.np,
                )
            )

        return {"arm_means": arm_means}

    def expected_rewards_from_params(self, params: Dict[str, Any]) -> np.ndarray:
        return np.asarray(params["arm_means"], dtype=float)


class BernoulliSpec(BanditTypeSpec):
    WIN = 100.0
    LOSE = 0.0

    @property
    def name(self) -> str:
        return "bernoulli"

    def compute_log_marginal(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        timesteps: np.ndarray,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        n_arms = obs.shape[1]
        # Beta priors
        alpha = np.full(n_arms, type_specific_priors.get("alpha_0", 1.0))
        beta_ = np.full(n_arms, type_specific_priors.get("beta_0", 1.0))

        log_marg = 0.0
        for t in range(len(actions)):
            a_t = actions[t]
            r = obs[t, a_t]

            if r != self.LOSE and r != self.WIN:
                return (-np.inf, {"alpha": alpha, "beta": beta_})

            # P(success) under current posterior
            p = alpha[a_t] / (alpha[a_t] + beta_[a_t])
            if r == self.WIN:
                log_marg += np.log(p)
                alpha[a_t] += 1.0
            else:
                log_marg += np.log(1.0 - p)
                beta_[a_t] += 1.0

        extra = {
            "alpha": alpha,
            "beta": beta_,
        }
        return float(log_marg), extra

    def sample_params(
        self,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
        rng: RandomNumberGenerator,
        type_specific_posteriors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Sample p per arm from Beta posterior (or prior)
        if type_specific_posteriors and "alpha" in type_specific_posteriors:
            alpha = np.atleast_1d(type_specific_posteriors["alpha"])
            beta_ = np.atleast_1d(type_specific_posteriors["beta"])
        else:
            n_arms = len(np.atleast_1d(arm_mean_prior.hyper_loc))
            alpha = np.full(n_arms, type_specific_priors.get("alpha_0", 1.0))
            beta_ = np.full(n_arms, type_specific_priors.get("beta_0", 1.0))

        probs = stats.beta.rvs(alpha, beta_, random_state=rng.np)
        arm_means = np.atleast_1d(probs) * 100.0
        return {"arm_means": arm_means}

    def expected_rewards_from_params(self, params: Dict[str, Any]) -> np.ndarray:
        return np.asarray(params["arm_means"], dtype=float)


class LinearContextualSpec(BanditTypeSpec):
    @property
    def name(self) -> str:
        return "linear_contextual"

    def compute_log_marginal(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        timesteps: np.ndarray,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        n_arms = obs.shape[1]
        n_features = int(type_specific_priors.get("n_features", n_arms))

        trial_features = type_specific_priors.get("trial_features")
        identity_fallback = np.eye(n_arms, n_features)

        # Noise variance (scalar)
        noise_std = float(
            type_specific_priors.get(
                "noise_std", np.atleast_1d(arm_mean_prior.stdev)[0]
            )
        )
        noise_var = noise_std**2

        # Prior over theta: N(mu_0, Sigma_0)
        _theta_mean = type_specific_priors.get("theta_mean")
        theta_mean_prior = (
            np.zeros(n_features)
            if _theta_mean is None
            else np.atleast_1d(np.asarray(_theta_mean, dtype=float))
        )
        _theta_cov = type_specific_priors.get("theta_cov")
        theta_cov_prior = (
            np.eye(n_features) * 100.0
            if _theta_cov is None
            else np.atleast_2d(np.asarray(_theta_cov, dtype=float))
        )

        # Initialize precision form: Lambda = Sigma^{-1}, b = Lambda @ mu
        Lambda = np.linalg.inv(theta_cov_prior)
        b = Lambda @ theta_mean_prior

        log_marg = 0.0
        for t in range(len(actions)):
            a_t = actions[t]
            r = obs[t, a_t]
            if np.isnan(r):
                continue

            if trial_features is not None and t in trial_features:
                x = trial_features[t][a_t].astype(float)
            else:
                x = identity_fallback[a_t].astype(float)

            # Current posterior: mu = Sigma @ b, Sigma = Lambda^{-1}
            Sigma = np.linalg.inv(Lambda)
            mu = Sigma @ b

            # Predictive: r ~ N(mu · x, x^T Sigma x + noise_var)
            pred_mean = float(mu @ x)
            pred_var = float(x @ Sigma @ x) + noise_var
            log_marg += _reward_logpdf(r, pred_mean, pred_var)

            # Bayesian linear regression update
            Lambda = Lambda + np.outer(x, x) / noise_var
            b = b + r * x / noise_var

        # Final posterior
        Sigma_post = np.linalg.inv(Lambda)
        mu_post = Sigma_post @ b

        extra = {
            "theta_mean": mu_post,
            "theta_cov": Sigma_post,
        }
        return float(log_marg), extra

    def sample_params(
        self,
        arm_mean_prior: NormalNormal,
        type_specific_priors: Dict[str, Any],
        rng: RandomNumberGenerator,
        type_specific_posteriors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        n_arms = len(np.atleast_1d(arm_mean_prior.hyper_loc))
        n_features = int(type_specific_priors.get("n_features", n_arms))

        if "current_features" in type_specific_priors:
            current_features = np.atleast_2d(
                np.asarray(type_specific_priors["current_features"], dtype=float)
            )
        else:
            current_features = np.eye(n_arms, n_features)

        # Sample theta from posterior (or prior)
        if type_specific_posteriors and "theta_mean" in type_specific_posteriors:
            theta_mean = np.atleast_1d(type_specific_posteriors["theta_mean"])
            theta_cov = np.atleast_2d(type_specific_posteriors["theta_cov"])
        else:
            theta_mean = type_specific_priors.get("theta_mean", np.zeros(n_features))
            theta_cov = type_specific_priors.get(
                "theta_cov", np.eye(n_features) * 100.0
            )
            theta_mean = np.atleast_1d(np.asarray(theta_mean, dtype=float))
            theta_cov = np.atleast_2d(np.asarray(theta_cov, dtype=float))

        theta = stats.multivariate_normal.rvs(
            mean=theta_mean, cov=theta_cov, random_state=rng.np
        )
        theta = np.atleast_1d(theta)

        # Expected rewards per arm using this trial's features
        arm_means = current_features @ theta
        return {"arm_means": arm_means, "theta": theta}

    def expected_rewards_from_params(self, params: Dict[str, Any]) -> np.ndarray:
        return np.asarray(params["arm_means"], dtype=float)


# Hierarchical belief: hierarchical prior over types and params


@dataclasses.dataclass
class HierarchicalBelief:
    type_probs: np.ndarray
    type_specs: List[BanditTypeSpec]
    arm_mean_prior: NormalNormal
    type_specific_priors: Dict[str, Dict[str, Any]]
    type_specific_posteriors: Dict[str, Dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )

    def update(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        timesteps: np.ndarray,
    ) -> "HierarchicalBelief":
        if len(actions) == 0:
            return self

        n_types = len(self.type_specs)
        type_lls = np.zeros(n_types)

        param_posts = dict(self.type_specific_posteriors)
        for i, spec in enumerate(self.type_specs):
            # Merge prior with current posterior so specs see updated belief (e.g. Gaussian over eps)
            merged = {
                **self.type_specific_priors.get(spec.name, {}),
                **self.type_specific_posteriors.get(spec.name, {}),
            }
            ll, extra = spec.compute_log_marginal(
                actions,
                obs,
                timesteps,
                self.arm_mean_prior,
                merged,
            )
            type_lls[i] = ll

            # Store type-specific param posteriors
            if spec.name == "time_delayed" and "k_probs" in extra:
                param_posts["time_delayed"] = {
                    "k_values": extra["k_values"],
                    "k_probs": extra["k_probs"],
                }
            elif spec.name == "drifting" and "eps_mean" in extra:
                param_posts["drifting"] = {
                    "eps_mean": extra["eps_mean"],
                    "eps_stdev": extra["eps_stdev"],
                }
            elif spec.name == "patch_foraging" and "lambda_mean" in extra:
                param_posts["patch_foraging"] = {
                    "lambda_mean": extra["lambda_mean"],
                    "lambda_stdev": extra["lambda_stdev"],
                }
            elif spec.name == "two_context":
                tc_priors = self.type_specific_priors.get("two_context", {})
                param_posts["two_context"] = {
                    "mu_post_c0": extra.get("mu_post_c0"),
                    "sigma_sq_c0": extra.get("sigma_sq_c0"),
                    "mu_post_c1": extra.get("mu_post_c1"),
                    "sigma_sq_c1": extra.get("sigma_sq_c1"),
                }
                if "current_context" in tc_priors:
                    param_posts["two_context"]["next_context"] = int(
                        tc_priors["current_context"]
                    )
                if "context_history" in tc_priors:
                    param_posts["two_context"]["context_history"] = tc_priors[
                        "context_history"
                    ]
            elif spec.name == "three_context":
                tc3_priors = self.type_specific_priors.get("three_context", {})
                post = {}
                for c in range(3):
                    post[f"mu_post_c{c}"] = extra.get(f"mu_post_c{c}")
                    post[f"sigma_sq_c{c}"] = extra.get(f"sigma_sq_c{c}")
                if "current_context" in tc3_priors:
                    post["next_context"] = int(tc3_priors["current_context"])
                if "context_history" in tc3_priors:
                    post["context_history"] = tc3_priors["context_history"]
                param_posts["three_context"] = post
            elif spec.name == "bernoulli" and "alpha" in extra:
                param_posts["bernoulli"] = {
                    "alpha": extra["alpha"],
                    "beta": extra["beta"],
                }

            elif spec.name == "linear_contextual" and "theta_mean" in extra:
                lc_post = {
                    "theta_mean": extra["theta_mean"],
                    "theta_cov": extra["theta_cov"],
                }
                # Preserve trial_features and current_features references from priors
                lc_priors = self.type_specific_priors.get("linear_contextual", {})
                if "trial_features" in lc_priors:
                    lc_post["trial_features"] = lc_priors["trial_features"]
                if "current_features" in lc_priors:
                    lc_post["current_features"] = lc_priors["current_features"]
                param_posts["linear_contextual"] = lc_post

        # update the posterior probabilities for the types
        log_post = np.log(self.type_probs + 1e-300) + type_lls
        log_post = log_post - np.max(log_post)
        new_type_probs = np.exp(log_post) / np.exp(log_post).sum()

        # Arm mean posterior: conjugate update per arm
        n_arms = obs.shape[1]
        arm_prior_list = DistributionList(
            [
                NormalNormal(
                    hyper_loc=np.atleast_1d(self.arm_mean_prior.hyper_loc)[a],
                    hyper_stdev=np.atleast_1d(self.arm_mean_prior.hyper_stdev)[a],
                    stdev=np.atleast_1d(self.arm_mean_prior.stdev)[a],
                )
                for a in range(n_arms)
            ]
        )
        arm_posterior_list = arm_prior_list.update(obs)
        hyp_locs = np.array([arm_posterior_list[a].hyper_loc for a in range(n_arms)])
        hyp_stds = np.array([arm_posterior_list[a].hyper_stdev for a in range(n_arms)])
        arm_posterior = NormalNormal(
            hyper_loc=hyp_locs,
            hyper_stdev=hyp_stds,
            stdev=self.arm_mean_prior.stdev,
        )

        return HierarchicalBelief(
            type_probs=new_type_probs,
            type_specs=self.type_specs,
            arm_mean_prior=arm_posterior,
            type_specific_priors=self.type_specific_priors,
            type_specific_posteriors=param_posts,
        )

    def sample_type_and_params(
        self, rng=default_rng, n: int = 1
    ) -> List[Tuple[int, Dict]]:
        results = []
        for _ in range(n):
            t = rng.np.choice(len(self.type_probs), p=self.type_probs)
            spec = self.type_specs[t]
            params = spec.sample_params(
                self.arm_mean_prior,
                self.type_specific_priors.get(spec.name, {}),
                rng,
                self.type_specific_posteriors.get(spec.name),
            )
            results.append((int(t), params))
        return results

    @property
    def n_types(self) -> int:
        return len(self.type_specs)

    def apply_hint(self, hint_template: str) -> "HierarchicalBelief":
        from hint_functions import HINT_TEMPLATE_TO_META_FUNCTION, apply_meta_hint

        meta_fn = HINT_TEMPLATE_TO_META_FUNCTION.get(hint_template)
        if meta_fn is None:
            return self
        type_names = [s.name for s in self.type_specs]
        log_likelihoods = meta_fn(type_names)
        new_type_probs = apply_meta_hint(self.type_probs, log_likelihoods)
        return dataclasses.replace(self, type_probs=new_type_probs)


# Default type specs and priors for common use
def default_type_specs() -> List[BanditTypeSpec]:
    return [
        StationarySpec(),
        DriftingSpec(),
        TimeDelayedSpec(),
        PatchForagingSpec(),
        TwoContextSpec(),
        ThreeContextSpec(),
        BernoulliSpec(),
        LinearContextualSpec(),
    ]


def default_hierarchical_belief(
    n_arms: int = 5,
    arm_prior_loc=50.0,
    arm_prior_scale=10.0,
    arm_stdevs=None,
    drift_eps: float = 10.0,
    k_values: List[int] = [1, 2, 3, 4, 5],
    lam: float = 0.3,
    lam_stdev: float = 0.12,
    eps_stdev: float = 5.0,
    n_features: int = None,
    noise_std: float = None,
) -> HierarchicalBelief:
    specs = default_type_specs()
    type_probs = np.ones(len(specs)) / len(specs)
    if arm_stdevs is None:
        arm_stdevs = np.ones(n_arms) * 5.0

    arm_mean_prior = NormalNormal(
        hyper_loc=np.full(n_arms, arm_prior_loc),
        hyper_stdev=np.full(n_arms, arm_prior_scale),
        stdev=arm_stdevs,  # observation noise (must match bandit!)
    )
    type_specific_priors = {
        "drifting": {
            "eps_mean": drift_eps,
            "eps_stdev": eps_stdev,
        },
        "time_delayed": {
            "k_values": k_values,
            "k_probs": np.ones(len(k_values)) / len(k_values),  # uniform
        },
        "patch_foraging": {
            "lambda_mean": lam,
            "lambda_stdev": lam_stdev,
        },
    }
    # Linear contextual: set up theta prior based on n_features
    lc_prior = {}
    if n_features is not None:
        lc_prior["n_features"] = n_features
        lc_prior["theta_mean"] = np.zeros(n_features)
        lc_prior["theta_cov"] = np.eye(n_features) * 100.0
    if noise_std is not None:
        lc_prior["noise_std"] = float(noise_std)
    if lc_prior:
        type_specific_priors["linear_contextual"] = lc_prior

    return HierarchicalBelief(
        type_probs=type_probs,
        type_specs=specs,
        arm_mean_prior=arm_mean_prior,
        type_specific_priors=type_specific_priors,
    )
