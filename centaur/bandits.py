import numpy as np


# stationary multi-armed bandit
class stationary_MAB:
    def __init__(self):
        self.type = "stationary"
        self.narms = None
        self.means = None
        self.stds = None

    def set_narms(self, n):
        self.narms = n

    def set_means(self, arms):
        self.means = np.array(arms, dtype=float)

    def set_stds(self, stds):
        self.stds = np.array(stds, dtype=float)

    def get_reward(self, a):
        return np.random.normal(self.means[a], self.stds[a], 1)[0]


######## ALL REMAINING BANDITS INHERIT FROM STATIONARY MAB ########


# non-stationary multi-armed bandit (drifting)
class drifting_MAB(stationary_MAB):
    def __init__(self, drift_rate=2):
        super().__init__()
        self.type = "drifting"
        self.drift_rate = drift_rate

    def set_drift_rate(self, rate):
        self.drift_rate = rate

    def get_reward(self, a):
        r = np.random.normal(self.means[a], self.stds[a], 1)[0]
        self.means += np.random.normal(0, self.drift_rate, self.narms)
        return r


# 2-context multi-armed bandit
class two_context_MAB(stationary_MAB):
    def __init__(self):
        super().__init__()
        self.type = "two_context"
        self.rev_means = None

    def set_means(self, arms):
        self.means = np.array(arms, dtype=float)
        # precompute reverse means for efficiency
        ranks = np.argsort(np.argsort(self.means.copy()))
        rev_sorted_means = np.sort(self.means.copy())[::-1]
        self.rev_means = rev_sorted_means[ranks]

    def get_reward(self, a, c):
        if c == 0:
            return np.random.normal(self.means[a], self.stds[a], 1)[0]
        elif c == 1:
            return np.random.normal(self.rev_means[a], self.stds[a], 1)[0]


# 3-context multi-armed bandit
class three_context_MAB(stationary_MAB):
    def __init__(self):
        super().__init__()
        self.type = "three_context"
        self.s = None  # c0: sorted
        self.r = None  # c1: reversed
        self.rand = None  # c2: random

    def set_means(self, arms):
        self.means = np.array(arms, dtype=float)
        self.s = np.sort(self.means)
        self.r = np.sort(self.means)[::-1]
        self.rand = np.random.permutation(np.array(self.means, dtype=int))

    def get_reward(self, a, c):
        if c == 0:
            self.means = self.s
        elif c == 1:
            self.means = self.r
        else:
            self.means = self.rand
        return np.random.normal(self.means[a], self.stds[a], 1)[0]


# time-delayed multi-armed bandit
class time_delayed_MAB(stationary_MAB):
    def __init__(self, delay=3):
        super().__init__()
        self.type = "time_delayed"
        self.delay = delay
        self.rewards = []

    def get_reward(self, a):
        r = np.random.normal(self.means[a], self.stds[a], 1)[0]
        self.rewards.append(r)
        if len(self.rewards) < self.delay:
            return 0
        else:
            return self.rewards.pop(0)


# patch foraging multi-armed bandit
class patch_foraging_MAB(stationary_MAB):
    def __init__(self, n_arms=5, change_rate=0.2):
        super().__init__()
        self.type = "patch_foraging"
        self.narms = n_arms
        self.step = 0
        self.change_rate = change_rate

    def set_means(self, arms):
        self.gt_means = np.array(arms, dtype=float)
        self.means = np.array(arms, dtype=float)
        self.curr_rewards = np.copy(self.means)

    def set_change_rate(self, rate):
        self.change_rate = rate

    def get_reward(self, a):
        r = np.random.normal(self.means[a], self.stds[a], 1)[0]
        self.step += 1
        # replenish the rewards of the patches not chosen
        other_patches = np.ones(self.narms, dtype=bool)
        other_patches[a] = False  # exclude the chosen patch
        self.curr_rewards[other_patches] = np.minimum(
            self.gt_means[other_patches],
            self.curr_rewards[other_patches] * np.exp(self.change_rate),
        )
        # return depleted reward of chosen patch
        self.curr_rewards[a] = np.maximum(
            0, self.curr_rewards[a] * np.exp(-self.change_rate)
        )
        self.means = self.curr_rewards

        return r


# bernoulli multi-armed bandit
class bernoulli_MAB(stationary_MAB):
    def __init__(self):
        super().__init__()
        self.type = "bernoulli"

    def set_means(self, arms):
        self.means = np.array(arms, dtype=float) / 100.0

    def get_reward(self, a):
        return np.random.binomial(1, self.means[a]) * 100


class linear_contextual_MAB(stationary_MAB):
    def __init__(self, n_features=3):
        super().__init__()
        self.type = "linear_contextual"
        self.nfeats = n_features
        self.theta = np.zeros(n_features, dtype=float)

    def set_theta(self, theta):
        self.theta = theta
        assert len(self.theta) == self.nfeats, (
            "Dimension of theta must match number of features."
        )

    def get_reward(self, a):
        if self.theta is None:
            raise ValueError("Theta must be set before getting rewards.")
        # context is a vector of features
        return np.dot(self.theta, a) + np.random.normal(0, 0.1)


if __name__ == "__main__":
    # test out two context bandit
    bandit = two_context_MAB()
    bandit.set_narms(5)
    bandit.set_means([10, 50, 30, 40, 20])
    # rev should be [50, 10, 30, 20, 40]

    # rev should be [20 , 200,60 , 80, 0]
    bandit.set_stds([5, 5, 5, 5, 5])

    for _ in range(10):
        print(bandit.get_reward(0, 0))  # context 0, arm 0
        print(
            bandit.get_reward(0, 1)
        )  # context 1, arm 0 (should be highest reward arm)
        print(
            bandit.get_reward(4, 0)
        )  # context 0, arm 4 (should be highest reward arm)
        print(bandit.get_reward(4, 1))  # context 1, arm 4 (should be lowest reward arm)

        print("means", bandit.means.copy())
        print("rev means", bandit.rev_means.copy())
