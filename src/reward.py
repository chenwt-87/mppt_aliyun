import numpy as np

from src.common import History


class Reward:
    def __call__(self, history: History) -> float:
        raise NotImplementedError


class DiscreteRewardDeltaPower(Reward):
    """
    Returns a constant reward based on the value of the change of power

    if dp < neg_treshold:
        return a
    elif neg_treshold <= dp < pos_treshold:
        return b
    else:
        return c
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 1.0,
        neg_treshold: float = -0.1,
        pos_treshold: float = 0.1,
    ):
        self.a = abs(a)
        self.b = b
        self.c = c
        self.neg_treshold = neg_treshold
        self.pos_treshold = pos_treshold

    def __call__(self, history: History) -> float:
        dp = history.dp[-1]

        if dp < self.neg_treshold:
            return -self.a
        elif self.neg_treshold <= dp < self.pos_treshold:
            return self.b
        else:
            return self.c


class RewardDeltaPower:
    """
    Returns a reward based on the value of the change of power

    if dp < 0:
        return a * dp
    else:
        return b * dp
    """

    def __init__(self, a: float, b: float):
        self.a = abs(a)
        self.b = b

    def __call__(self, history: History) -> float:
        # return history.p[-1]
        dp = history.dp_act[-1]
        if dp < 0:
            return self.a * np.power(dp, 3)
        else:
            return self.b * np.power(dp, 3)


class RewardPower:
    """
    Returns a reward based on the produced power
    """

    def __init__(self, norm: bool = False):
        self.norm = norm

    def __call__(self, history: History) -> float:
        if self.norm:
            return history.p_norm[-1]
        return history.p[-1]


class RewardPowerDeltaPower:
    """
    Returns a reward based on the produced power
    """

    def __init__(self, norm: bool = False):
        self.norm = norm

    def __call__(self, history: History) -> float:
        if self.norm:
            if history.dp_norm[-1] < 0:
                return 0
            else:
                return history.p_norm[-1]
        return history.p[-1] + history.dp[-1] * 5


class RewardDeltaPowerVoltage:
    """
    Returns a reward based on the value of the change of power and change of voltage

    if dp < 0:
        return a * dp - c * abs(dv)
    else:
        return b * dp - c * abs(dv)
    """

    def __init__(self, a: float, b: float, c: float):
        self.a = abs(a)
        self.b = b
        self.c = abs(c)

    def __call__(self, history: History) -> float:
        dp = history.dp[-1]
        # print('dpppppp', dp)
        # dp = 0
        diff_v = history.dv_set2pv[-1]
        if dp < 0:
            return self.a * dp - self.c * abs(diff_v)
        else:
            return 300 - self.b * dp - 1000 * self.c * abs(diff_v) + 100 * history.p[-1]
        # return history.p[-1]


if __name__ == "__main__":
    history = History()
    reward_fn = RewardDeltaPowerVoltage(2, 0, 1)
    history.dp.append(-1.2)
    reward_fn(history)
    history.dp.append(0)
    reward_fn(history)
    history.dp.append(0.2)
    reward_fn(history)
