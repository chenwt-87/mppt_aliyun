import random

import gym
import pandas as pd
from typing import Optional, List
import numpy as np
import os
import matplotlib.pyplot as plt
from math import atan2

from src.pv_array import PVArray
from src.utils import read_his_data_csv
from src.common import StepResult, History
from src.logger import logger

G_MAX = 1200
T_MAX = 60


class PVEnvBase(gym.Env):
    "PV Environment abstract class for solving the MPPT by reinforcement learning"

    # metadata = {"render.modes": ["human"]}
    # spec = gym.envs.registration.EnvSpec("PVEnv-v0")

    def reset(self):
        raise NotImplementedError

    def step(self, action) -> np.ndarray:
        raise NotImplementedError

    def render(self, vars: List) -> None:
        raise NotImplementedError

    def _get_observation_space(self) -> gym.Space:
        raise NotImplementedError

    def _get_action_space(self) -> gym.Space:
        raise NotImplementedError

    def _get_delta_v(self, action: float) -> float:
        # print(action)
        raise NotImplementedError

    @classmethod
    def from_file(
            cls, pv_params_path: str, his_data_path: str, pvarray_ckp_path: str, **kwargs
    ):
        pvarray = PVArray.from_json(pv_params_path, ckp_path=pvarray_ckp_path, data_from_gateway_path=his_data_path)
        his_data = read_his_data_csv(his_data_path)
        his_data['power'] = his_data.apply(lambda x: x['voltage']*x['current']/1e6, axis=1)
        pv_environment = cls(pvarray, his_data, **kwargs)
        return pv_environment


class PVEnv(PVEnvBase):
    """
    PV Continuos Environment for solving the MPPT by reinforcement learning

    Parameters:
        - pvarray: the pvarray object
        - weather_df: a pandas dataframe object containing weather readings
        - states: list of states to return as observations
        - reward_fn: function that calculates the reward
        - seed: for reproducibility
    """

    def __init__(
            self,
            pvarray: PVArray,
            pv_history_data: pd.DataFrame,
            states: List[str],
            reward_fn: callable,
            seed: Optional[int] = None,
            v0: Optional[float] = None,
    ) -> None:

        self.pvarray = pvarray
        self.states = states
        self.pv_gateway_history = pv_history_data
        self.reward_fn = reward_fn
        self.v0 = v0
        if seed:
            np.random.seed(seed)

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    def reset(self) -> np.ndarray:
        self.history = History()
        self.step_counter = 0
        self.step_idx = 0
        self.done = False
        idx = random.randint(0, self.pv_gateway_history.shape[0])
        pv_v = self.pv_gateway_history.at[idx, 'voltage'] / 1000
        pv_i = self.pv_gateway_history.at[idx, 'current'] / 1000
        # 随机初始化一个电压
        # v = np.random.randint(2, self.pvarray.voc)
        v = 0.75 * self.pvarray.voc
        i = 0.8 * self.pvarray.isc
        #   self._store_step 中获取当前温度和光照， 并通过查历史数据 或者 matlab仿真，得到电流，功率，
        #   返回【v_norm,i_norm,dv】
        obs0 = self._store_step(v, i, pv_v, pv_i)
        print('obs0   reset', obs0)
        self.pvarray.READ_SENSOR_TIME -= 1
        return obs0

    #  experience.py 37 行， 调用该函数   new_obs, reward, done, _ = self.env.step(action)
    def step(self, action: float) -> StepResult:
        if self.done:
            raise ValueError("The episode is done")

        self.step_idx += 1
        print('self.step_counter', self.step_counter)
        pv_v = self.pv_gateway_history.at[self.step_counter, 'voltage'] / 1000
        pv_i = self.pv_gateway_history.at[self.step_counter, 'current'] / 1000
        # 依据 action 选择电压的增量，
        # delta_v = self.actions[action]
        # actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10]
        # for k in range(5):
        delta_v = self._get_delta_v(action)
        # clip 函数， 将v+delta_v 限制在 0 和 Voc 之间
        print('\n ======= self.v={},delta_v={}, action={}, pv_panel_v:{} pv_panel_i: {}'.format(
            self.v, delta_v, action, pv_v, pv_i))
        v = np.clip(self.v + delta_v,  0.5 * self.pvarray.voc, self.pvarray.voc)
        # 依据 v， 通过历史数据或者matlab仿真，得到 obs
        # self.history() 赋值
        obs = self._store_step(v, self.i, pv_v, pv_i)
        #  obs ['v_norm', 'i_norm', 'dv']
        # print('test_obs', obs)

        """
        Returns a reward based on the value of the change of power
        若 dp > 0 则 reward = 0.9*dp
        若 dp < 0 则 reward = 2*dp
        """
        reward = self.reward_fn(self.history)

        # if self.history.p[-1] < 0 or self.history.v[-1] < 1:
        #     self.done = True
        if self.step_counter >= len(self.pv_gateway_history) - 1:
            self.done = True

        return StepResult(
            obs,
            reward,
            self.done,
            {"step_idx": self.step_idx, "steps": self.step_counter},
        )

    def render(self, vars: List[str]) -> None:
        for var in vars:
            if var in ["dp", "dv"]:
                plt.hist(getattr(self.history, var), label=var)
            else:
                plt.plot(getattr(self.history, var), label=var)
            plt.legend()
            plt.show()

    def render_vs_true(self, po: bool = False) -> None:
        # p_real, v_real, _ = self.pvarray.get_true_mpp(self.history.g, self.history.t)
        if po:
            p_po, v_po = self.pv_gateway_history['power'], self.pv_gateway_history['voltage']/1e3
        # plt.plot(p_real, label="P Max")
        plt.plot(self.history.p, label="P RL")
        if po:
            plt.plot(p_po, label="P P&O")
        plt.legend()
        plt.show()
        # plt.plot(v_real, label="Vmpp")
        plt.plot(self.history.v, label="V RL")
        if po:
            plt.plot(v_po, label="V P&O")
        plt.legend()
        plt.show()

        # if po:
        #     logger.info(f"PO Efficiency={PVArray.mppt_eff(p_real, p_po)}")
        logger.info(f"RL P_  Efficiency={PVArray.mppt_eff(p_po, self.history.p)}")
        logger.info(f"RL V_ Efficiency={PVArray.mppt_mae(v_po, self.history.v)}")

    def _add_history(self, p, v, v_pv, i) -> None:
        self.history.p.append(p)
        self.history.v.append(v)
        self.history.v_pv.append(v_pv)
        self.history.i.append(i)
        # 无法获取气温和辐照，历史数据中不再存储之。
        # self.history.g.append(g)
        # self.history.t.append(t)
        self.history.p_norm.append(p / self.pvarray.pmax)
        self.history.v_norm.append(v / self.pvarray.voc)
        self.history.i_norm.append(i / self.pvarray.isc)
        # self.history.g_norm.append(g / G_MAX)
        # self.history.t_norm.append(t / T_MAX)

        if len(self.history.p) < 2:
            self.history.dp.append(0.0)
            self.history.dv.append(0.0)
            self.history.dv_set2pv.append(0.0)
            self.history.di.append(0.0)
            self.history.deg.append(0.0)
            self.history.dp_norm.append(0.0)
            self.history.dv_norm.append(0.0)
        else:
            self.history.dp.append(self.history.p[-1] - self.history.p[-2])
            self.history.dv.append(self.history.v[-1] - self.history.v[-2])
            # 计算设定电压值和组件电压值之间之差
            self.history.dv_set2pv.append(abs(self.history.v[-1] - self.history.v_pv[-1]))
            self.history.di.append(self.history.i[-1] - self.history.i[-2])
            self.history.dp_norm.append(
                self.history.p_norm[-1] - self.history.p_norm[-2]
            )
            self.history.dv_norm.append(
                self.history.v_norm[-1] - self.history.v_norm[-2]
            )
            self.history.deg.append(
                atan2(self.history.di[-1], self.history.dv[-1])
                + atan2(self.history.i[-1], self.history.v[-1])
            )

    def _get_delta_v(self, action: float) -> float:
        # print(action)
        if isinstance(action, list):
            action = action[0]
        return action

    def _get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=np.array([-np.inf] * len(self.states)),
            high=np.array([-np.inf] * len(self.states)),
            shape=(len(self.states),),
            dtype=np.float32,
        )

    def _get_action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-round(self.pvarray.voc * 0.1, 0),
            high=round(self.pvarray.voc * 0.1, 0),
            shape=(1,),
            dtype=np.float32,
        )

    def _store_step(self, v: float, i: float, p_v: float, p_i: float) -> np.ndarray:
        # g, t = 1000, 25
        # 从串口 或者 zigbee 获取数
        # v 为Actor输出的电压
        p, self.v, self.v_pv, = self.pvarray.simulate(v, i, p_v, p_i)
        # 组件 实际的电压v_pv
        self._add_history(p=p, v=self.v, v_pv=self.v_pv, i=i)

        # getattr(handler.request, 'GET') is the same as handler.request.GET
        # print('test  g,t,v',   np.array([getattr(self.history, state)[-1] for state in self.states]))
        return np.array([getattr(self.history, state)[-1] for state in self.states])


class PVEnvDiscrete(PVEnv):
    """
    PV Discrete Environment for solving the MPPT by reinforcement learning

    Parameters:
        - pvarray: the pvarray object
        - his_data_df: a pandas dataframe object from INC_MPPT, which is supposed to be the true MPP
        - states: list of states to return as observations
        - reward_fn: function that calculates the reward
        - seed: for reproducibility
    """

    def __init__(
            self,
            pvarray: PVArray,
            his_data_df: pd.DataFrame,
            states: List[str],
            reward_fn: callable,
            actions: List[float],
            seed: Optional[int] = None,
            v0: Optional[float] = None,
    ) -> None:
        self.actions = actions
        super().__init__(
            pvarray,
            his_data_df,
            states,
            reward_fn,
            seed,
            v0,
        )

    def _get_delta_v(self, action: int) -> float:
        # print(action)
        return self.actions[action]

    def _get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.actions))


if __name__ == "__main__":

    def reward_fn(history: History) -> float:
        dp = history.dp[-1]
        if dp < -0.1:
            return -1
        elif -0.1 <= dp < 0.1:
            return 0
        else:
            return 1


    env = PVEnvDiscrete.from_file(
        pv_params_path=os.path.join("../parameters", "01_pvarray.json"),
        his_data_path=os.path.join("../data", "data_for_train_A2C.csv"),
        pvarray_ckp_path=os.path.join("../data", "01_pvarray_iv.json"),
        states=["v", "p", "g", "t"],
        reward_fn=reward_fn,
        actions=[-0.1, 0, 0.1],
    )

    obs = env.reset()
    # while True:
    #     action = env.action_space.sample()
    #     new_obs, reward, done, info = env.step(action)

    #     if done:
    #         break

    # env.render()
