import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, Optional
import functools
import dataclasses
from bandits import GaussianMAB
from distributions import DistributionList, default_rng, RandomNumberGenerator
from hierarchical import HierarchicalBelief


class BayesianDecisionMaker(ABC):
    @abstractmethod
    def update(
        self,
        obs: np.ndarray,
        actions: Optional[np.ndarray] = None,
        timesteps: Optional[np.ndarray] = None,
    ):
        pass

    @abstractmethod
    def sample_env(
        self,
        obs: np.ndarray = None,
        n=1,
        rng=default_rng,
        actions=None,
        timesteps=None,
        context=None,
    ):
        pass

    @abstractmethod
    def sample_action(
        self,
        obs: np.ndarray = None,
        n=1,
        rng=default_rng,
        actions=None,
        timesteps=None,
        context=None,
    ):
        pass

    def run_on(
        self, bandit: GaussianMAB, total_trials: int, agent_seed=None, env_seed=None
    ):
        agent_rng = RandomNumberGenerator(agent_seed)
        env_rng = RandomNumberGenerator(env_seed)
        obs = np.zeros((total_trials, bandit.n_arms))
        obs[:] = np.nan
        actions = np.zeros(total_trials, dtype=int)
        rewards = np.zeros((total_trials, bandit.n_arms)) + 0.0000001

        for t in range(total_trials):
            if t == 0:
                if hasattr(bandit, "reset"):
                    bandit.reset()
                if hasattr(bandit, "_context_history"):
                    bandit._context_history.clear()
                if hasattr(bandit, "_trial_features"):
                    bandit._trial_features.clear()
            if hasattr(bandit, "generate_context"):
                bandit.generate_context(t, env_rng)
            if hasattr(bandit, "generate_trial_features"):
                bandit.generate_trial_features(t, env_rng)
            context = (
                bandit._context_history.get(t)
                if hasattr(bandit, "_context_history")
                else None
            )
            try:
                a = self.sample_action(
                    obs=obs,
                    n=1,
                    rng=agent_rng,
                    actions=actions[:t] if t > 0 else np.array([], dtype=int),
                    timesteps=np.arange(t) if t > 0 else np.array([], dtype=int),
                    context=context,
                )[0]
            except ValueError:
                a = agent_rng.np.choice(bandit.n_arms)
            actions[t] = a
            if hasattr(bandit, "sample_rewards"):
                rewards[t] = bandit.sample_rewards(t=t, action=a, rng=env_rng)
                obs[t, :] = rewards[t, :]
            else:
                rewards[t] = bandit.reward_dists.sample(rng=env_rng)[0,]
                obs[t, a] = rewards[t, a]
        return actions, obs, rewards


@dataclasses.dataclass
class RandomAgent(BayesianDecisionMaker):
    action_list: tuple

    def update(self, obs: np.ndarray, actions=None, timesteps=None):
        raise NotImplementedError

    def sample_env(
        self,
        *,
        obs: np.ndarray = None,
        n=1,
        rng=default_rng,
        actions=None,
        timesteps=None,
    ):
        raise NotImplementedError

    def sample_action(
        self,
        *,
        obs: np.ndarray = None,
        n=1,
        rng=default_rng,
        actions=None,
        timesteps=None,
        context=None,
    ):
        return rng.np.choice(self.action_list, n)


class ThompsonSamplingAgent(BayesianDecisionMaker):
    def sample_action(
        self,
        *,
        obs: np.ndarray = None,
        n=1,
        rng=default_rng,
        actions=None,
        timesteps=None,
        context=None,
    ):
        env = self.sample_env(
            obs=obs, n=n, rng=rng, actions=actions, timesteps=timesteps, context=context
        )
        return np.argmax(env, axis=1)


@dataclasses.dataclass
class StandardAgent(ThompsonSamplingAgent):
    initial_beliefs: DistributionList
    name: str = "StandardAgent"

    @functools.cached_property
    def action_list(self):
        return tuple(range(len(self.initial_beliefs)))

    def update(self, obs: np.ndarray, actions=None, timesteps=None):
        return self.initial_beliefs.update(obs)

    def sample_env(
        self,
        *,
        obs: np.ndarray = None,
        n=1,
        rng=default_rng,
        actions=None,
        timesteps=None,
        context=None,
    ):
        arm_mean_beliefs = StandardAgent.update(self, obs).conjugate()
        arm_mean_sample = arm_mean_beliefs.sample(rng=rng, n=n)
        return arm_mean_sample


# HIERARCHICAL AGENT: infers problem type and params
@dataclasses.dataclass
class HierarchicalAgent(ThompsonSamplingAgent):
    hierarchical_belief: HierarchicalBelief
    name: str = "HierarchicalAgent"

    @functools.cached_property
    def action_list(self):
        return tuple(range(len(self.hierarchical_belief.arm_mean_prior.hyper_loc)))

    def update(self, obs: np.ndarray, actions=None, timesteps=None):
        if actions is None or timesteps is None or len(actions) == 0:
            return self.hierarchical_belief
        # Slice obs to observed rows only
        n_obs = len(actions)
        obs_slice = (
            obs[:n_obs, :]
            if obs is not None
            else np.full((n_obs, self.action_list[-1] + 1), np.nan)
        )
        return self.hierarchical_belief.update(
            actions=actions, obs=obs_slice, timesteps=timesteps
        )

    def sample_env(
        self,
        *,
        obs=None,
        n=1,
        rng=default_rng,
        actions=None,
        timesteps=None,
        context=None,
    ):
        belief = self.update(obs, actions, timesteps)
        samples = belief.sample_type_and_params(rng=rng, n=n)
        rewards = np.zeros((n, len(self.action_list)))
        for i, (t_idx, params) in enumerate(samples):
            spec = belief.type_specs[t_idx]
            rewards[i] = spec.expected_rewards_from_params(params)
        return rewards


# HIERARCHICAL + HINT: hierarchical belief over types/params, then hint-weighted sampling
@dataclasses.dataclass
class HierarchicalHintAgent(ThompsonSamplingAgent):
    hierarchical_belief: HierarchicalBelief
    hint: Callable[[np.ndarray], float]
    weighted_samples: int = 1000
    hint_strength: float = 5.0
    name: str = "HierarchicalHintAgent"

    @functools.cached_property
    def action_list(self):
        return tuple(range(len(self.hierarchical_belief.arm_mean_prior.hyper_loc)))

    def __post_init__(self):
        if self.name == "HierarchicalHintAgent":
            display = getattr(self.hint, "name", None) or getattr(
                self.hint, "__name__", "hint"
            )
            self.name = f"Hint: {display}"

    def update(self, obs: np.ndarray, actions=None, timesteps=None):
        if actions is None or timesteps is None or len(actions) == 0:
            return self.hierarchical_belief
        n_obs = len(actions)
        obs_slice = (
            obs[:n_obs, :]
            if obs is not None
            else np.full((n_obs, len(self.action_list)), np.nan)
        )
        return self.hierarchical_belief.update(
            actions=actions, obs=obs_slice, timesteps=timesteps
        )

    def sample_env(
        self,
        *,
        obs=None,
        n=1,
        rng=default_rng,
        actions=None,
        timesteps=None,
        context=None,
    ):
        belief = self.update(obs, actions, timesteps)
        samples = belief.sample_type_and_params(rng=rng, n=self.weighted_samples)
        t_ids = np.array([t_idx for t_idx, _ in samples])
        params_list = [params for _, params in samples]
        # (weighted_samples, n_arms) — each row is one sample of arm means
        arm_mean_samples = np.array(
            [
                belief.type_specs[t_idx].expected_rewards_from_params(params)
                for t_idx, params in samples
            ]
        )
        # hint expects (t_ids, rs, params, **kwargs) with rs (n_arms, n_samples)
        weights = np.asarray(
            self.hint(
                t_ids,
                arm_mean_samples.T,
                params_list,
                actions=actions,
                obs=obs,
                context=context,
            ),
            dtype=float,
        ) * float(self.hint_strength)
        weights = np.where(np.isfinite(weights), weights, -np.inf)
        wmax = np.max(weights)
        if not np.isfinite(wmax):
            probs = np.ones(self.weighted_samples) / self.weighted_samples
        else:
            weights = weights - wmax
            probs = np.exp(weights)
            probs = probs / (np.sum(probs) + 1e-300)
        sample_i = rng.np.choice(self.weighted_samples, n, p=probs, replace=True)
        return arm_mean_samples[sample_i]

    def action_probs(
        self, *, obs=None, rng=default_rng, actions=None, timesteps=None
    ) -> np.ndarray:
        belief = self.update(obs, actions, timesteps)
        samples = belief.sample_type_and_params(rng=rng, n=self.weighted_samples)

        # keep track of the type, arm means, and params for the hint function!!
        t_ids = np.array([t_idx for t_idx, _ in samples])
        params_list = [params for _, params in samples]
        arm_mean_samples = np.array(
            [
                belief.type_specs[t_idx].expected_rewards_from_params(params)
                for t_idx, params in samples
            ]
        )

        # everything that goes to the hint needs to come from the samples
        weights = np.asarray(
            self.hint(t_ids, arm_mean_samples.T, params_list, actions=actions, obs=obs),
            dtype=float,
        ) * float(self.hint_strength)

        weights = np.where(np.isfinite(weights), weights, -np.inf)
        weights = weights - np.max(weights)
        probs = np.exp(weights)
        probs = probs / (np.sum(probs) + 1e-300)
        # P(a) = sum(probs[i] for i where argmax(arm_mean_samples[i]) == a)
        n_arms = arm_mean_samples.shape[1]
        best_actions = np.argmax(arm_mean_samples, axis=1)
        action_probs = np.zeros(n_arms)
        for a in range(n_arms):
            action_probs[a] = np.sum(probs[best_actions == a])
        action_probs = action_probs / (np.sum(action_probs) + 1e-300)
        return action_probs


# STATIONARY BANDIT
@dataclasses.dataclass
class HintAgent(ThompsonSamplingAgent):
    initial_beliefs: DistributionList
    hint: Callable[[np.ndarray], float]
    weighted_samples: int = 1000
    name: str = None
    hint_strength: float = 5.0

    @functools.cached_property
    def action_list(self):
        return tuple(range(len(self.initial_beliefs)))

    def __post_init__(self):
        if self.hint_strength is None:
            self.hint_strength = 5.0
        if self.name is None:
            if hasattr(self.hint, "name"):
                self.name = f"Hint: {self.hint.name}"
            else:
                self.name = f"Hint: {self.hint.__name__}"

    def update(self, obs: np.ndarray, actions=None, timesteps=None):
        return self.initial_beliefs.update(obs)

    def sample_env(
        self,
        *,
        obs: np.ndarray = None,
        n=1,
        rng=default_rng,
        actions=None,
        timesteps=None,
        context=None,
    ):
        arm_mean_beliefs = StandardAgent.update(self, obs).conjugate()
        arm_mean_samples = arm_mean_beliefs.sample(rng=rng, n=self.weighted_samples).T
        if self.hint_strength is None:
            self.hint_strength = 5.0
        weights = np.asarray(self.hint(arm_mean_samples), dtype=float) * float(
            self.hint_strength
        )
        # numerically stable softmax -> sampling probabilities
        weights = weights - np.max(weights)
        probs = np.exp(weights)
        probs = probs / np.sum(probs)
        sample_i = rng.np.choice(self.weighted_samples, n, p=probs, replace=True)
        sample = arm_mean_samples[:, sample_i].T
        return sample
