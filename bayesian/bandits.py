from collections import deque
from typing import Optional
from utils import method_cache
import functools
import dataclasses
from distributions import Normal, default_rng, RandomNumberGenerator
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


@dataclasses.dataclass
class GaussianMAB:
    arm_means: tuple
    arm_stdevs: tuple

    @functools.cached_property
    def n_arms(self):
        assert len(self.arm_means) == len(self.arm_stdevs)
        return len(self.arm_means)

    @functools.cached_property
    def action_list(self):
        return tuple(range(self.n_arms))

    @functools.cached_property
    def reward_dists(self) -> Normal:
        return Normal(loc=self.arm_means, stdev=self.arm_stdevs)

    def reward(self, a, rng=default_rng):
        return self.reward_dists.sample(rng=rng, n=1)[0, a]

    @method_cache
    def expected_reward(self, a=None):
        if a is None:
            return [self.expected_reward(a) for a in self.action_list]
        return self.arm_means[a]

    def plot(self, xmin=-10, xmax=110, ax=None):
        if xmin is None or xmax is None:
            xmin = min(m - 2 * sd for m, sd in zip(self.arm_means, self.arm_stdevs))
            xmax = max(m + 2 * sd for m, sd in zip(self.arm_means, self.arm_stdevs))
        if ax is None:
            fig, ax = plt.subplots()
        x = np.linspace(xmin, xmax, 200)
        for mu, sd in zip(self.arm_means, self.arm_stdevs):
            y = norm.pdf(x=x, loc=mu, scale=sd)
            ax.plot(x, y)
        return ax

    def __str__(self) -> str:
        return ", ".join(
            [f"{m:.2f}±{sd:.2f}" for m, sd in zip(self.arm_means, self.arm_stdevs)]
        )


@dataclasses.dataclass
class DriftingMAB(GaussianMAB):
    drift_epsilon: float = 2
    _current_means: np.ndarray = dataclasses.field(init=False, repr=False, default=None)
    _last_t: int = dataclasses.field(init=False, repr=False, default=0)

    def __post_init__(self):
        self._current_means = np.asarray(self.arm_means, dtype=float).copy()
        self._last_t = 0

    @property
    def reward_dists(self) -> Normal:
        return Normal(loc=self._current_means, stdev=self.arm_stdevs)

    def expected_reward(self, a=None):
        if a is None:
            return [self.expected_reward(a) for a in self.action_list]
        return float(self._current_means[a])

    def sample_rewards(self, *, t: int, action: int, rng=default_rng) -> np.ndarray:
        if t < self._last_t:
            self._current_means = np.asarray(self.arm_means, dtype=float).copy()
            self._last_t = 0

        dt = t - self._last_t
        if dt > 0 and self.drift_epsilon and self.drift_epsilon != 0.0:
            self._current_means = self._current_means + rng.np.normal(
                loc=0.0,
                scale=float(self.drift_epsilon) * np.sqrt(float(dt)),
                size=self._current_means.shape,
            )
        if dt > 0:
            self._last_t = t

        rewards = np.full(self.n_arms, np.nan)
        rewards[action] = self.reward_dists.sample(rng=rng, n=1)[0, action]
        return rewards

    # just does the dynamic version of .reward_dists
    def reward(self, a, rng=default_rng, t: Optional[int] = None):
        if t is not None:
            return self.sample_rewards(t=t, action=a, rng=rng)[a]
        return self.reward_dists.sample(rng=rng, n=1)[0, a]

    def __str__(self) -> str:
        return ", ".join(
            [f"{m:.2f}±{sd:.2f}" for m, sd in zip(self._current_means, self.arm_stdevs)]
        )


@dataclasses.dataclass
class TimeDelayedMAB(GaussianMAB):
    delay_k: int = 3
    _queue: deque = dataclasses.field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._queue = deque(maxlen=self.delay_k)

    def sample_rewards(self, *, t: int, action: int, rng=default_rng) -> np.ndarray:
        rewards = np.zeros(self.n_arms)
        if len(self._queue) == self.delay_k:
            delayed_a, delayed_r = self._queue[0]
            rewards[delayed_a] = delayed_r

        immediate = self.reward_dists.sample(rng=rng, n=1)[0, action]
        self._queue.append((action, immediate))
        return rewards

    def reward(self, a, rng=default_rng, t: Optional[int] = None):
        if t is not None:
            return self.sample_rewards(t=t, action=a, rng=rng)[a]
        return self.reward_dists.sample(rng=rng, n=1)[0, a]

    def reset(self):
        self._queue.clear()


@dataclasses.dataclass
class PatchForagingMAB(GaussianMAB):
    change_rate: float = 0.2
    _current_rewards: np.ndarray = dataclasses.field(
        init=False, repr=False, default=None
    )
    _last_t: int = dataclasses.field(init=False, repr=False, default=0)

    def __post_init__(self):
        self._current_rewards = np.asarray(self.arm_means, dtype=float).copy()
        self._last_t = 0

    @property
    def reward_dists(self) -> Normal:
        return Normal(loc=self._current_rewards, stdev=self.arm_stdevs)

    def expected_reward(self, a=None):
        if a is None:
            return [self.expected_reward(a) for a in self.action_list]
        return float(self._current_rewards[a])

    def sample_rewards(self, *, t: int, action: int, rng=default_rng) -> np.ndarray:
        gt_means = np.asarray(self.arm_means, dtype=float)
        if t == 0 or t != self._last_t + 1:
            self._current_rewards = gt_means.copy()
            self._last_t = 0
        self._last_t = t

        rewards = np.full(self.n_arms, np.nan)
        rewards[action] = self.reward_dists.sample(rng=rng, n=1)[0, action]

        # Deplete chosen patch
        self._current_rewards[action] = np.maximum(
            0.0,
            self._current_rewards[action] * np.exp(-self.change_rate),
        )
        # Replenish others (capped at gt_means)
        other = np.ones(self.n_arms, dtype=bool)
        other[action] = False
        self._current_rewards[other] = np.minimum(
            gt_means[other],
            self._current_rewards[other] * np.exp(self.change_rate),
        )
        return rewards

    def reward(self, a, rng=default_rng, t: Optional[int] = None):
        if t is not None:
            return self.sample_rewards(t=t, action=a, rng=rng)[a]
        # One-off pull without timestep: sample then apply transition
        r = self.reward_dists.sample(rng=rng, n=1)[0, a]
        gt_means = np.asarray(self.arm_means, dtype=float)
        self._current_rewards[a] = np.maximum(
            0.0, self._current_rewards[a] * np.exp(-self.change_rate)
        )
        other = np.ones(self.n_arms, dtype=bool)
        other[a] = False
        self._current_rewards[other] = np.minimum(
            gt_means[other], self._current_rewards[other] * np.exp(self.change_rate)
        )
        return r

    def reset(self):
        self._current_rewards = np.asarray(self.arm_means, dtype=float).copy()
        self._last_t = 0

    def __str__(self) -> str:
        return ", ".join(
            [
                f"{m:.2f}±{sd:.2f}"
                for m, sd in zip(self._current_rewards, self.arm_stdevs)
            ]
        )


@dataclasses.dataclass
class TwoContextMAB(GaussianMAB):
    avg_context_duration: int = 4
    _rev_means: np.ndarray = dataclasses.field(init=False, repr=False, default=None)
    _context_history: dict = dataclasses.field(
        init=False, repr=False, default_factory=dict
    )
    _current_context: int = dataclasses.field(init=False, repr=False, default=0)

    @property
    def n_contexts(self) -> int:
        return 2

    def __post_init__(self):
        means = np.asarray(self.arm_means, dtype=float)
        ranks = np.argsort(np.argsort(means))
        rev_sorted_means = np.sort(means.copy())[::-1]
        self._rev_means = rev_sorted_means[ranks]

    def generate_context(self, t: int, rng=default_rng) -> int:
        if t in self._context_history:
            return self._context_history[t]
        if t == 0:
            self._current_context = 0
        else:
            if rng.np.random() < 1.0 / self.avg_context_duration:
                self._current_context = 1 - self._current_context
        self._context_history[t] = self._current_context
        return self._current_context

    def _means_for_context(self, context: int) -> np.ndarray:
        if context == 0:
            return np.asarray(self.arm_means, dtype=float)
        return self._rev_means.copy()

    def expected_reward(self, a=None, context: int = 0):
        means = self._means_for_context(context)
        if a is None:
            return [float(means[i]) for i in self.action_list]
        return float(means[a])

    def sample_rewards(
        self,
        *,
        t: int,
        action: int,
        rng=default_rng,
        context: int = None,
    ) -> np.ndarray:
        if context is None:
            context = self._context_history[t]
        means = self._means_for_context(context)
        rewards = np.full(self.n_arms, np.nan)
        mu = means[action]
        sd = np.asarray(self.arm_stdevs, dtype=float)[action]
        rewards[action] = rng.np.normal(loc=mu, scale=sd, size=1)[0]
        return rewards

    def reward(self, a, rng=default_rng, t: Optional[int] = None, context: int = None):
        if t is not None:
            return self.sample_rewards(t=t, action=a, rng=rng, context=context)[a]
        return self.sample_rewards(t=0, action=a, rng=rng, context=0)[a]

    def __str__(self) -> str:
        return (
            f"two_context(means={self.arm_means}, rev_means={self._rev_means.tolist()}, "
            f"stdevs={self.arm_stdevs})"
        )


@dataclasses.dataclass
class ThreeContextMAB(GaussianMAB):
    random_seed: int = 0
    avg_context_duration: int = 4
    _sorted_means: np.ndarray = dataclasses.field(init=False, repr=False, default=None)
    _rev_means: np.ndarray = dataclasses.field(init=False, repr=False, default=None)
    _rand_means: np.ndarray = dataclasses.field(init=False, repr=False, default=None)
    _context_history: dict = dataclasses.field(
        init=False, repr=False, default_factory=dict
    )
    _current_context: int = dataclasses.field(init=False, repr=False, default=0)

    @property
    def n_contexts(self) -> int:
        return 3

    def __post_init__(self):
        means = np.asarray(self.arm_means, dtype=float)
        self._sorted_means = np.sort(means)
        self._rev_means = np.sort(means)[::-1]
        perm_rng = np.random.RandomState(self.random_seed)
        self._rand_means = perm_rng.permutation(means.copy())

    def generate_context(self, t: int, rng=default_rng) -> int:
        if t in self._context_history:
            return self._context_history[t]
        if t == 0:
            self._current_context = 0
        else:
            if rng.np.random() < 1.0 / self.avg_context_duration:
                others = [c for c in range(3) if c != self._current_context]
                self._current_context = int(rng.np.choice(others))
        self._context_history[t] = self._current_context
        return self._current_context

    def _means_for_context(self, context: int) -> np.ndarray:
        if context == 0:
            return self._sorted_means.copy()
        elif context == 1:
            return self._rev_means.copy()
        else:
            return self._rand_means.copy()

    def expected_reward(self, a=None, context: int = 0):
        means = self._means_for_context(context)
        if a is None:
            return [float(means[i]) for i in self.action_list]
        return float(means[a])

    def sample_rewards(
        self,
        *,
        t: int,
        action: int,
        rng=default_rng,
        context: int = None,
    ) -> np.ndarray:
        if context is None:
            context = self._context_history[t]
        means = self._means_for_context(context)
        rewards = np.full(self.n_arms, np.nan)
        mu = means[action]
        sd = np.asarray(self.arm_stdevs, dtype=float)[action]
        rewards[action] = rng.np.normal(loc=mu, scale=sd, size=1)[0]
        return rewards

    def reward(self, a, rng=default_rng, t: Optional[int] = None, context: int = None):
        if t is not None:
            return self.sample_rewards(t=t, action=a, rng=rng, context=context)[a]
        return self.sample_rewards(t=0, action=a, rng=rng, context=0)[a]

    def __str__(self) -> str:
        return (
            f"three_context(sorted={self._sorted_means.tolist()}, "
            f"rev={self._rev_means.tolist()}, "
            f"rand={self._rand_means.tolist()}, stdevs={self.arm_stdevs})"
        )


@dataclasses.dataclass
class BernoulliMAB(GaussianMAB):
    @functools.cached_property
    def _probs(self) -> np.ndarray:
        return np.asarray(self.arm_means, dtype=float) / 100.0

    def expected_reward(self, a=None):
        if a is None:
            return [float(self.arm_means[i]) for i in self.action_list]
        return float(self.arm_means[a])

    def sample_rewards(self, *, t: int, action: int, rng=default_rng) -> np.ndarray:
        rewards = np.full(self.n_arms, np.nan)
        rewards[action] = float(rng.np.binomial(1, self._probs[action])) * 100
        return rewards

    def reward(self, a, rng=default_rng, t: Optional[int] = None):
        if t is not None:
            return self.sample_rewards(t=t, action=a, rng=rng)[a]
        return float(rng.np.binomial(1, self._probs[a])) * 100

    def __str__(self) -> str:
        probs = self._probs
        return ", ".join([f"p={p:.2f}" for p in probs])


@dataclasses.dataclass
class LinearContextualMAB(GaussianMAB):
    theta: np.ndarray = dataclasses.field(default=None, repr=False)
    n_features: int = 3
    noise_std: float = 0.1
    special_feat_idx: int = 2
    _trial_features: dict = dataclasses.field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self):
        if self.theta is not None:
            theta = np.atleast_1d(np.asarray(self.theta, dtype=float))
            assert len(theta) == self.n_features, (
                f"len(theta)={len(theta)} must match n_features={self.n_features}"
            )
            object.__setattr__(self, "special_feat_idx", int(np.argmax(theta)))
        n_arms = len(self.arm_means)
        object.__setattr__(
            self, "arm_stdevs", tuple(np.full(n_arms, self.noise_std).tolist())
        )

    def generate_trial_features(self, t: int, rng=default_rng) -> np.ndarray:
        if t in self._trial_features:
            return self._trial_features[t]
        n_arms = self.n_arms
        features = rng.np.integers(0, 2, size=(n_arms, self.n_features)).astype(float)
        one_idx = int(rng.np.integers(n_arms))
        features[:, self.special_feat_idx] = 0.0
        features[one_idx, self.special_feat_idx] = 1.0
        self._trial_features[t] = features
        return features

    def get_trial_features(self, t: int) -> np.ndarray:
        return self._trial_features[t]

    def sample_rewards(self, *, t: int, action: int, rng=default_rng) -> np.ndarray:
        features = self._trial_features[t]
        theta = np.atleast_1d(np.asarray(self.theta, dtype=float))
        mean = float(theta @ features[action])
        rewards = np.full(self.n_arms, np.nan)
        rewards[action] = rng.np.normal(loc=mean, scale=self.noise_std)
        return rewards

    def reward(self, a, rng=default_rng, t: Optional[int] = None):
        if t is not None:
            return self.sample_rewards(t=t, action=a, rng=rng)[a]
        raise ValueError(
            "LinearContextualMAB.reward() requires t to look up trial features."
        )

    def expected_reward_at_trial(self, t: int, a: int = None):
        features = self._trial_features[t]
        theta = np.atleast_1d(np.asarray(self.theta, dtype=float))
        means = features @ theta
        if a is None:
            return means.tolist()
        return float(means[a])

    def __str__(self) -> str:
        return f"linear_contextual(theta={self.theta}, n_features={self.n_features}, noise={self.noise_std})"
