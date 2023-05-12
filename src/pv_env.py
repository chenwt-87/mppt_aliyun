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
from src.func import *
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
            cls, pv_params_path: str, his_data_path: str, pvarray_ckp_path: str, mode: str, **kwargs
    ):
        pvarray = PVArray.from_json(pv_params_path, ckp_path=pvarray_ckp_path, data_from_gateway_path=his_data_path)
        if mode == 'Train':
            flag = True
        elif mode == 'Test':
            flag = False
        his_data = read_his_data_csv(his_data_path, flag)
        his_data['power'] = his_data.apply(lambda x: x['voltage'] * x['current'] / 1e6, axis=1)

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

    def set_obs(self, idx) -> np.ndarray:
        # idx = 0
        num_pv_curve = self.pv_gateway_history.at[self.pv_gateway_history.index[idx], 'label']
        idx_max = self.pv_gateway_history[self.pv_gateway_history['label'] == num_pv_curve]['power'].idxmax()

        pv_v_curve_mpp = self.pv_gateway_history.at[idx_max, 'voltage'] / 1000
        pv_i_curve_mpp = self.pv_gateway_history.at[idx_max, 'current'] / 1000

        idx_min = self.pv_gateway_history[self.pv_gateway_history['label'] == num_pv_curve]['power'].idxmin()
        pv_v_curve_lpp = self.pv_gateway_history.at[idx_min, 'voltage'] / 1000
        pv_i_curve_lpp = self.pv_gateway_history.at[idx_min, 'current'] / 1000
        # num_pv_curve = self.pv_gateway_history.at[idx, 'label']

        # 随机初始化一个电压
        # v = np.random.randint(2, self.pvarray.voc)
        # v = 0.75 * self.pvarray.voc
        v = self.pv_gateway_history.at[self.pv_gateway_history.index[idx], 'voltage'] / 1000
        # v 已经曲线上， 获取i
        # i = 0.8 * self.pvarray.isc
        i = self.pv_gateway_history.at[self.pv_gateway_history.index[idx], 'current'] / 1000
        dv = pv_v_curve_mpp - v
        #   self._store_step 中获取当前温度和光照， 并通过查历史数据 或者 matlab仿真，得到电流，功率，
        #   返回【v_norm,i_norm,dv】
        # obs0 = np.array([v/self.pvarray.voc, i/self.pvarray.isc, dv/self.pvarray.voc])
        # obs0 = np.array([v / self.pvarray.voc, i / self.pvarray.isc])
        obs0 = np.array([v, i])
        # obs0 = [v, i]
        # env_train 和 env_test 初始化的时候，会生成两个obs0
        # print('obs   set', obs0)
        # self.pvarray.READ_SENSOR_TIME -= 1
        return obs0, num_pv_curve

    def reset(self) -> np.ndarray:
        self.history = History()
        self.history.p.append(0.0)
        self.history.v.append(0.0)
        self.history.v_pv.append(0.0)
        self.history.dp_act.append(0.0)
        self.history.p_mppt.append(0.0)
        self.history.i.append(0.0)
        # 无法获取气温和辐照，历史数据中不再存储之。
        # self.history.g.append(g)
        # self.history.t.append(t)
        self.history.p_norm.append(0.0)
        self.history.v_norm.append(0.0)
        self.history.i_norm.append(0.0)
        # self.history.g_norm.append(g / G_MAX)

        # self.history.dp.append(0.0)
        self.history.dv.append(0.0)
        self.history.dv_set2pv.append(0.0)
        self.history.di.append(0.0)
        self.history.deg.append(0.0)
        self.history.dp_norm.append(0.0)
        self.history.dv_norm.append(0.0)
        # counter_step 表示总的循环次数， step_idx表示历史数据的编号，会循环望夫
        # self.counter_step
        self.step_idx = 0
        self.done = False
        # 随机取了历史数据中一个点
        # idx = random.randint(0, self.pv_gateway_history.shape[0])
        idx = 0
        obs2, _ = self.set_obs(idx)
        return obs2

    #  experience.py 37 行， 调用该函数   new_obs, reward, done, _ = self.env.step(action)
    def step(self, action: float) -> StepResult:
        if self.done:
            raise ValueError("The episode is done")

        self.step_idx += 1
        print('self.counter_step', self.counter_step, 'self.step_idx', self.step_idx)
        tm_idx = self.pv_gateway_history.index[max(self.step_idx - 1, 0)]
        pv_v = self.pv_gateway_history.at[tm_idx, 'voltage'] / 1000
        pv_i = self.pv_gateway_history.at[tm_idx, 'current'] / 1000
        pv_curve_idx = self.pv_gateway_history.at[tm_idx, 'label']
        idx_max = self.pv_gateway_history[self.pv_gateway_history['label'] == pv_curve_idx]['power'].idxmax()

        pv_v_curve_now_mpp = self.pv_gateway_history.at[idx_max, 'voltage'] / 1000
        pv_i_curve_now_mpp = self.pv_gateway_history.at[idx_max, 'current'] / 1000

        idx_min = self.pv_gateway_history[self.pv_gateway_history['label'] == pv_curve_idx]['power'].idxmin()
        pv_v_curve_now_lpp = self.pv_gateway_history.at[idx_min, 'voltage'] / 1000
        pv_i_curve_now_lpp = self.pv_gateway_history.at[idx_min, 'current'] / 1000
        # print('当前曲线下，MPP idx={} i={}.v={}'.format(idx_max, pv_i_curve_now_mpp, pv_v_curve_now_mpp))
        # print('当前曲线下，LPP idx={} i={}.v={}'.format(idx_min, pv_i_curve_now_lpp, pv_v_curve_now_lpp))
        # print('当前曲线下，共采集{}个工作点'.format(
        #     self.pv_gateway_history[self.pv_gateway_history['label'] == pv_curve_idx].shape[0]))
        # 依据 action 选择电压的增量，
        # delta_v = self.actions[action]
        # actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10]
        delta_v = self._get_delta_v(action)
        # clip 函数， 将v+delta_v 限制在 0 和 Voc 之间
        # 22 为欠压门槛

        v = np.clip(pv_v + delta_v, 22, 56)

        i = inter1pd_iv_curve(1000 * v, self.pv_gateway_history[self.pv_gateway_history['label'] == pv_curve_idx])
        if i > self.pvarray.isc:
            print('拟合的电流：', i)
            i = 0
        print(
            '\n ======= ,原始数据-- {}, v--{}, i-{},delta_v={}, action={}, new_v = {}, 拟合 i ={} pv_mmp_v:{} pv_mmp_i: {}'.format(
                max(self.step_idx - 1, 0), pv_v, pv_i, delta_v, action, v, i, pv_v_curve_now_mpp, pv_i_curve_now_mpp))
        # 依据 v， 通过历史数据或者matlab仿真，得到 obs
        # self.history() 赋值
        obs = \
            self._store_step(v, i, pv_v_curve_now_mpp, pv_i_curve_now_mpp, pv_v, pv_i, self.step_idx - 1, True)
        self.pvarray.curve_num = self.pv_gateway_history.at[tm_idx, 'label']
        #  obs ['v_norm', 'i_norm', 'dv']
        """
        Returns a reward based on the value of the change of power
        若 dp > 0 则 reward = 0.9*dp
        若 dp < 0 则 reward = 2*dp
        """
        reward = self.reward_fn(self.history)
        if obs[1] == 0:
            print('--------')
        print('test_obs', obs, 'reward', reward)
        # if self.history.p[-1] < 0 or self.history.v[-1] < 1:
        #     self.done = True
        if self.step_idx >= len(self.pv_gateway_history) - 1:
            self.done = True

        return StepResult(
            obs,
            reward,
            self.done,
            {"step_idx": self.step_idx, "steps": self.counter_step, 'curve_idx': self.pvarray.curve_num},
        )

    def render(self, vars: List[str], source_tag) -> None:
        for var in vars:
            if var == 'v_pv':
                plt.figure(figsize=[8, 4])
                plt.plot(getattr(self.history, var), label=var)
                plt.savefig('img/{}--{}--line.jpg'.format(var, source_tag))
                plt.figure(figsize=[8, 4])
                plt.hist(getattr(self.history, var), label=var)
                plt.savefig('img/{}--{}--hist.jpg'.format(var, source_tag))
            elif var in ["dp", "dv", 'dp_act']:
                plt.figure(figsize=[8, 4])
                plt.hist(getattr(self.history, var), label=var)
                plt.legend()

                plt.savefig('img/{}--{}--hist.jpg'.format(var, source_tag))
            else:
                plt.figure(figsize=[8, 4])
                plt.plot(getattr(self.history, var), label=var)
                plt.legend()
                plt.savefig('img/{}--{}--line.jpg'.format(var, source_tag))

    # def get_ture_mpp_from_his(self, voltage, current):
    #     v, i =
    #     p = v*i
    #     return self.PVSimResult(p, v, v, i)
    def render_vs_true(self, po: bool = False, source_tag: str = 'train') -> None:
        # p_real, v_real, _ = self.pvarray.get_true_mpp(self.history.g, self.history.t)
        if po:
            p_po, v_po = [], []
            for idx in self.pv_gateway_history.index:
                tag = self.pv_gateway_history.at[idx, 'label']
                idx_mpp = self.pv_gateway_history[self.pv_gateway_history['label'] == tag]['power'].idxmax()
                p_po.append(self.pv_gateway_history.at[idx_mpp, 'power'])
                v_po.append(self.pv_gateway_history.at[idx_mpp, 'voltage'] / 1000)
        # plt.plot(p_real, label="P Max")
        plt.figure(figsize=[20, 6])
        plt.plot(self.history.p, 'o', markersize=1, label="P RL")
        plt.plot(np.array(self.history.p) - np.array(self.history.dp_act), 'o', markersize=1, label="P origin")
        if po:
            plt.plot(p_po, label="P mpp")
        plt.legend()
        plt.savefig('img/效果P--{}.jpg'.format(source_tag))
        # plt.plot(v_real, label="Vmpp")

        plt.figure(figsize=[10, 6])
        # plt.plot(self.history.v, self.history.p, 'o', markersize=1, label="P V RL ")
        plt.plot(self.history.v_pv, self.history.p, 'o', markersize=2, label="P V ----RL")
        plt.plot(self.history.v_pv, np.array(self.history.p) - np.array(self.history.dp_act), 'o', markersize=2,
                 label="P V origin")
        plt.grid()
        plt.xlim(0, 70)
        # plt.ylim(0, 0)
        plt.legend()
        plt.savefig('img/效果P-V--{}.jpg'.format(source_tag))

        plt.figure(figsize=[20, 6])
        plt.plot(self.history.v, 'o', markersize=1, label="V RL")
        plt.plot(self.history.v_pv, 'o', markersize=1, label="V origin")
        if po:
            plt.plot(v_po, label="V mpp")
        plt.legend()
        # plt.show()
        plt.savefig('img/效果V--{}.jpg'.format(source_tag))

        # if po:
        #     logger.info(f"PO Efficiency={PVArray.mppt_eff(p_real, p_po)}")
        logger.info(f"RL P_  Efficiency={PVArray.mppt_eff(p_po, self.history.p)}")
        logger.info(f"RL V_ MAE={PVArray.mppt_mae(v_po, self.history.v)}")
        logger.info(f"RL V_ MAPE={PVArray.mppt_mape(v_po, self.history.v)}")

    def _add_history(self, p, v, v_pv, i, dp_act, p_mppt) -> None:
        self.history.p.append(p)
        self.history.v.append(v)
        self.history.v_pv.append(v_pv)
        self.history.dp_act.append(dp_act)
        self.history.p_mppt.append(p_mppt)
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
            # self.history.dp.append(0.0)
            self.history.dv.append(0.0)
            self.history.dv_set2pv.append(0.0)
            self.history.di.append(0.0)
            self.history.deg.append(0.0)
            self.history.dp_norm.append(0.0)
            self.history.dv_norm.append(0.0)
        else:
            # self.history.dp.append(self.history.p[-1] - self.history.p[-2])
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
            low=np.array([-1] * len(self.states)),
            high=np.array([1] * len(self.states)),
            shape=(len(self.states),),
            dtype=np.float32,
        )

    def _get_action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-round(1, 0),
            high=round(1, 0),
            shape=(1,),
            dtype=np.float32,
        )

    def _store_step(self, v: float, i: float, p_v_max: float, p_i_max: float, p_v_org: float, p_i_org: float, idx,
                    set_flag) -> np.ndarray:
        # g, t = 1000, 25
        # 从串口 或者 zigbee 获取数
        # v 为Actor输出的电压
        p, _, self.v_pv, self.i = self.pvarray.simulate(v, i, p_v_max, p_i_max, p_v_org, p_i_org, idx, set_flag)
        # 组件 实际的电压v_pv
        dp_act = p - p_v_org * p_i_org
        p_mppt = p_i_max * p_v_max
        if set_flag:
            self._add_history(p=p, v=v, v_pv=p_v_org, i=i, dp_act=dp_act, p_mppt=p_mppt)

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
        self.counter_step = 0
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
