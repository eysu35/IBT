from functools import cached_property
from abc import ABC, abstractmethod, abstractproperty
import random
from typing import Union, Tuple, List
import numpy as np
import scipy.stats as stats
import dataclasses


class RandomNumberGenerator(random.Random):
    def __init__(self, x=None):
        super().__init__(x)
        self.np = np.random.Generator(np.random.PCG64(self.randint(0, int(1e50))))

    def fork(self, n: int):
        return [RandomNumberGenerator(self.randint(0, int(1e100))) for _ in range(n)]

    def new_seed(self):
        return self.randint(0, int(1e100))


default_rng = RandomNumberGenerator(None)


class Distribution(ABC):
    @abstractmethod
    def sample(self, rng=default_rng, n: int = 1, name=None) -> np.ndarray:
        pass

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def prob(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractproperty
    def parameters(self) -> List[np.ndarray]:
        pass

    @cached_property
    def ndim(self) -> int:
        return max([np.ndim(p) for p in self.parameters])

    @cached_property
    def shape(self) -> Tuple[int]:
        if self.ndim == 0:
            return ()
        for p in self.parameters:
            if np.ndim(p) == 1:
                return np.shape(p)

    def __init__(self):
        for p in self.parameters:
            p = np.array(p)
            assert np.ndim(p) in (0, 1), f"Parameter {p} must be a scalar or 1D array"
        if self.ndim == 1:
            assert (
                len(
                    set([np.array(p).shape for p in self.parameters if np.ndim(p) == 1])
                )
                == 1
            ), "1D Parameters must have the same shape"


@dataclasses.dataclass
class DistributionList(Distribution):
    distributions: List[Distribution]

    @property
    def parameters(self):
        return [d.parameters for d in self.distributions]

    def sample(self, rng=default_rng, n=1, name=None):
        return np.vstack(
            [d.sample(rng=rng, n=n, name=name) for d in self.distributions]
        ).T

    def log_prob(self, x: np.ndarray):
        return np.sum([d.log_prob(x[..., i]) for i, d in enumerate(self.distributions)])

    def prob(self, x: np.ndarray):
        return np.exp(self.log_prob(x))

    def update(self, x: np.ndarray):
        if x is None:
            return self
        assert x.shape[-1] == len(self.distributions)
        return DistributionList(
            [d.update(x[..., i]) for i, d in enumerate(self.distributions)]
        )

    def conjugate(self):
        if all([isinstance(d, CompoundDistribution) for d in self.distributions]):
            return DistributionList([d.conjugate() for d in self.distributions])
        else:
            raise ValueError("Element distributions do not have conjugate priors")

    def __getitem__(self, i):
        return self.distributions[i]

    def __len__(self):
        return len(self.distributions)


class CompoundDistribution(Distribution):
    @abstractmethod
    def update(self, **kwargs) -> "CompoundDistribution":
        pass

    @abstractmethod
    def conjugate(self) -> Distribution:
        pass


@dataclasses.dataclass
class Normal(Distribution):
    loc: np.ndarray = 0
    stdev: np.ndarray = 1

    def __post_init__(self):
        super().__init__()

    @property
    def parameters(self):
        return (self.loc, self.stdev)

    def sample(self, rng: RandomNumberGenerator = default_rng, n: int = 1, name=None):
        size = (n,) + self.shape
        x = stats.norm.rvs(
            size=size, loc=self.loc, scale=self.stdev, random_state=rng.np
        )
        return x

    def log_prob(self, x):
        return stats.norm.logpdf(x, loc=self.loc, scale=self.stdev)

    def prob(self, x):
        return stats.norm.pdf(x, loc=self.loc, scale=self.stdev)


@dataclasses.dataclass
class CategoricalDistribution:
    probs: np.ndarray  # shape (n_outcomes,), must sum to 1

    def __post_init__(self):
        self.probs = np.asarray(self.probs, dtype=float)
        self.probs = self.probs / self.probs.sum()

    def sample(self, rng=default_rng, n: int = 1) -> np.ndarray:
        return rng.np.choice(len(self.probs), size=n, p=self.probs)

    def log_prob(self, i: int) -> float:
        return np.log(self.probs[int(i)] + 1e-300)


@dataclasses.dataclass
class NormalNormal(CompoundDistribution):
    hyper_loc: np.ndarray = 0
    hyper_stdev: np.ndarray = 1
    stdev: np.ndarray = 1

    def __post_init__(self):
        self.hyper_loc = np.array(self.hyper_loc)
        self.hyper_stdev = np.array(self.hyper_stdev)
        self.stdev = np.array(self.stdev)
        super().__init__()

    def sample(self, rng: RandomNumberGenerator = default_rng, n: int = 1, name=None):
        size = (n,) + self.shape
        loc = stats.norm.rvs(
            size=size, loc=self.hyper_loc, scale=self.hyper_stdev, random_state=rng.np
        )
        x = stats.norm.rvs(size=size, loc=loc, scale=self.stdev, random_state=rng.np)
        return x

    def log_prob(self, x):
        return stats.norm.logpdf(
            x, loc=self.hyper_loc, scale=(self.hyper_stdev**2 + self.stdev**2) ** 0.5
        )

    def prob(self, x):
        return np.exp(self.log_prob(x))

    def update(self, x):
        total_x = np.nansum(x, axis=0)
        nonnan_x = len(x) - np.isnan(x).sum(axis=0)
        hyper_var = self.hyper_stdev**2
        obs_var = self.stdev**2
        new_hyper_loc = (self.hyper_loc / hyper_var + total_x / obs_var) / (
            1 / hyper_var + nonnan_x / obs_var
        )
        new_hyper_stdev = (1 / (1 / hyper_var + nonnan_x / obs_var)) ** 0.5
        return NormalNormal(
            hyper_loc=new_hyper_loc,
            hyper_stdev=new_hyper_stdev,
            stdev=self.stdev.copy(),
        )

    def conjugate(self) -> "Normal":
        return Normal(loc=self.hyper_loc, stdev=self.hyper_stdev)

    @property
    def parameters(self):
        return (self.hyper_loc, self.hyper_stdev, self.stdev)
