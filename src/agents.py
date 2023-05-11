from typing import Optional, List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import math
from torch.utils.tensorboard import SummaryWriter

from src.experience import ExperienceSorceDiscountedSteps
from src.policies import (
    BasePolicy,
    DiscreteCategoricalDistributionPolicy,
    DiscreteGreedyPolicy,
    GaussianPolicy,
)
from src.logger import logger


class AgentABC:
    "Abstract class of Agent"

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def learn(self) -> None:
        raise NotImplementedError

    def train_net(self) -> None:
        raise NotImplementedError

    def test(self, num_episodes: int):
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def plot_performance(self, metrics: List[str]) -> None:
        raise NotImplementedError

    def save(self, path: Optional[str]) -> None:
        raise NotImplementedError

    def load(self, path: Optional[str]) -> None:
        raise NotImplementedError

    def _value_state(self) -> torch.Tensor:
        raise NotImplementedError

    def _prepare_batch(self) -> Tuple[torch.Tensor]:
        raise NotImplementedError

    def _get_train_policy(self) -> BasePolicy:
        raise NotImplementedError

    def _get_test_policy(self) -> BasePolicy:
        raise NotImplementedError


class Agent(AgentABC):
    "Base class for an Agent"

    def __init__(
        self,
        env: gym.Env,
        test_env: gym.Env,
        net: nn.Module,
        device: torch.device,
        gamma: float,
        beta_entropy: float,
        lr: float,
        n_steps: int,
        batch_size: int,
        chk_path: str,
        optimizer: str = "adam",
    ):
        self.env = env
        self.test_env = test_env
        self.net = net
        self.device = device
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.n_steps = n_steps
        self.chk_path = chk_path
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        else:
            raise ValueError("Only `adam` is supported")
        self.policy = self._get_train_policy()
        self.test_policy = self._get_train_policy()
        self.exp_train_source = ExperienceSorceDiscountedSteps(
            env=env,
            policy=self.policy,
            gamma=gamma,
            n_steps=n_steps,
            steps=batch_size,
        )
        self.exp_test_source = ExperienceSorceDiscountedSteps(
            env=test_env,
            policy=self.test_policy,
            gamma=gamma,
            n_steps=n_steps,
            steps=batch_size,
        )

        self.reset()

        if chk_path:
            if os.path.exists(chk_path):
                self.load()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.policy(obs)

    def learn(
        self,
        steps: int,
        verbose_every: Optional[int] = 0,
        save_every: Optional[int] = 0,
    ):
        self.env.step_idx = 0
        self.env.counter_step = 0
        writer = SummaryWriter(comment='comment')
        for _ in tqdm(range(steps)):
            self.counter_step += 1
            batch = self._prepare_batch()
            # batch [obs,action, reward]
            # reward 为 target_values,
            # print('test_batch', batch)
            # 送16组数据进行训练 ，BATCH_SIZE=16
            # agent.py line 352
            self.train_net(*batch)

            if verbose_every:
                if self.counter_step % verbose_every == 0:
                    print(
                        "\n",
                        f"{ self.counter_step}: loss={self.total_loss:.6f}, ",
                        f"mean reward={self.mean_reward:.2f}, ",
                        f"steps/ep={self.steps_per_ep}, ",
                        f"episodes={self.counter_ep}",
                    )
                writer.add_scalar(tag="train_log/total_loss", scalar_value=self.total_loss,
                                  global_step=self.counter_step)
                writer.add_scalar(tag="train_log/mean_reward", scalar_value=self.mean_reward,
                                  global_step=self.counter_step)
                # writer.add_scalar(tag="train_log/hist_total_rew", scalar_value=self.hist_total_rew,
                #                   global_step=self.counter_step)
                writer.add_scalar(tag="train_log/entropy_loss", scalar_value=self.entropy_loss,
                                  global_step=self.counter_step)
                writer.add_scalar(tag="train_log/policy_loss", scalar_value=self.policy_loss,
                                  global_step=self.counter_step)
                writer.add_scalar(tag="train_log/value_loss", scalar_value=self.value_loss,
                                  global_step=self.counter_step)

                net_weight = self.net.net[0].weight
                writer.add_histogram("net_weight", net_weight, self.counter_step)
                net_bias = self.net.net[0].bias
                writer.add_histogram("net_bias", net_bias, self.counter_step)

                net_weight_grad = self.net.net[0].weight.grad
                writer.add_histogram("net_weight_grad", net_weight_grad, self.counter_step)
                net_bias_grad = self.net.net[0].bias.grad
                writer.add_histogram("net_bias_grad", net_bias_grad, self.counter_step)

                actor_weight = self.net.actor[0].weight
                writer.add_histogram("actor_weight", actor_weight, self.counter_step)
                actor_bias = self.net.actor[0].bias
                writer.add_histogram("actor_bias", actor_bias, self.counter_step)
                critic_weight = self.net.critic[0].weight
                writer.add_histogram("critic_weight", critic_weight, self.counter_step)
                critic_bias = self.net.critic[0].bias
                writer.add_histogram("critic_bias", critic_bias, self.counter_step)

                critic_weight_grad = self.net.critic[0].weight.grad
                writer.add_histogram("critic_weight_grad", critic_weight_grad, self.counter_step)
                actor_weight_grad = self.net.actor[0].weight.grad
                writer.add_histogram("actor_weight_grad", actor_weight_grad, self.counter_step)

                critic_bias_grad = self.net.critic[0].bias.grad
                writer.add_histogram("critic_bias_grad", critic_bias_grad, self.counter_step)
                actor_bias_grad = self.net.actor[0].bias.grad
                writer.add_histogram("actor_bias_grad", actor_bias_grad, self.counter_step)

            if save_every and self.chk_path:
                if self.counter_step % save_every == 0:
                    self.save()
        writer.close()
        if self.chk_path:
            self.save()

    def test(self, num_episodes: int = 1):
        episodes = self.exp_test_source.play_episodes(episodes=num_episodes)

        reward = 0
        for ep in episodes:
            for step in ep:
                reward += step.reward
        reward /= num_episodes

        return reward

    def reset(self) -> None:
        self.hist_total_rew = []
        self.hist_mean_rew = []
        self.hist_steps = []
        self.hist_total_loss = []
        self.hist_entropy_loss = []
        self.hist_value_loss = []
        self.hist_policy_loss = []
        self.counter_ep = 0
        self.counter_steps_per_ep = 0
        self.counter_step = 0
        self.ep_reward = 0
        self.total_loss = np.NaN
        self.policy_loss = np.NaN
        self.entropy_loss = np.NaN
        self.value_loss = np.NaN
        self.mean_reward = np.NaN
        self.steps_per_ep = np.NaN

    def plot_performance(
        self,
        metrics: List[str] = [
            "mean_rewards",
            "total_rewards",
            "total_loss",
            "policy_loss",
            "value_loss",
            "entropy_loss",
        ],
    ) -> None:
        dic = {
            "mean_rewards": self.hist_mean_rew,
            "total_rewards": self.hist_total_rew,
            "total_loss": self.hist_total_loss,
            "policy_loss": self.hist_policy_loss,
            "value_loss": self.hist_value_loss,
            "entropy_loss": self.hist_entropy_loss,
        }

        for metric in metrics:
            plt.plot(self.hist_steps, dic[metric], label=metric)
            # if "loss" in metric and not "entropy" in metric:
            #     plt.yscale("log")
            plt.legend()
            plt.show()

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.chk_path
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hist_total_rew": self.hist_total_rew,
                "hist_mean_rew": self.hist_mean_rew,
                "hist_steps": self.hist_steps,
                "hist_total_loss": self.hist_total_loss,
                "hist_entropy_loss": self.hist_entropy_loss,
                "hist_value_loss": self.hist_value_loss,
                "hist_policy_loss": self.hist_policy_loss,
                "counter_ep": self.counter_ep,
                "counter_step": self.counter_step,
                "total_loss": self.total_loss,
                "policy_loss": self.policy_loss,
                "entropy_loss": self.entropy_loss,
                "value_loss": self.value_loss,
                "mean_reward": self.mean_reward,
                "steps_per_ep": self.steps_per_ep,
            },
            path,
        )
        logger.info(f"Checkpoint saved to {path}")

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.chk_path
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.hist_total_rew = checkpoint["hist_total_rew"]
        self.hist_mean_rew = checkpoint["hist_mean_rew"]
        self.hist_steps = checkpoint["hist_steps"]
        self.hist_total_loss = checkpoint["hist_total_loss"]
        self.hist_entropy_loss = checkpoint["hist_entropy_loss"]
        self.hist_value_loss = checkpoint["hist_value_loss"]
        self.hist_policy_loss = checkpoint["hist_policy_loss"]
        self.counter_ep = checkpoint["counter_ep"]
        self.counter_step = checkpoint["counter_step"]
        self.counter_steps_per_ep = 0
        self.ep_reward = 0
        self.total_loss = checkpoint["total_loss"]
        self.policy_loss = checkpoint["policy_loss"]
        self.entropy_loss = checkpoint["entropy_loss"]
        self.value_loss = checkpoint["value_loss"]
        self.mean_reward = checkpoint["mean_reward"]
        self.steps_per_ep = checkpoint["steps_per_ep"]
        logger.info(f"Checkpoint loaded from {path}")

    def _prepare_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = []
        actions = []
        last_states = []
        values = []
        dones = []
        #   self.exp_train_source: 函数中有网络定义, 会进入到play_step()
        for exp in next(self.exp_train_source):
            # print('test_exp', exp)
            # reward 仅仅是动作价值，这里不考虑状态价值
            self.ep_reward += exp.reward
            self.counter_steps_per_ep += exp.steps

            states.append(exp.state)
            actions.append(exp.action)
            values.append(exp.discounted_reward)

            if exp.last_state is None:
                dones.append(True)
                last_states.append(exp.state)  # as a placeholder, we'll mask this val

                self.counter_ep += 1
                self.steps_per_ep = self.counter_steps_per_ep

                self.hist_total_rew.append(self.ep_reward)
                self.mean_reward = np.mean(self.hist_total_rew)
                self.hist_mean_rew.append(self.mean_reward)
                self.hist_steps.append(self.counter_step)
                self.hist_total_loss.append(self.total_loss)
                self.hist_entropy_loss.append(self.entropy_loss)
                self.hist_value_loss.append(self.value_loss)
                self.hist_policy_loss.append(self.policy_loss)

                self.ep_reward = 0
                self.counter_steps_per_ep = 0
            else:
                dones.append(False)
                last_states.append(exp.last_state)
        # states_t 经过  actions_t 后， 到达 last_states_t
        last_states_t = torch.tensor(last_states, dtype=torch.float32).to(self.device)
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.float).to(self.device)
        # critic 网络，# 返回critic 的值
        # values = self.net(states)[1].squeeze()
        # 表示状态价值函数 critic 即 v(s_{t},w) https://blog.csdn.net/weixin_41960890/article/details/121947760
        # values_last 即 \gamma*v(s_{t+1},w)
        values_last = self._value_state(last_states_t) * self.gamma ** self.n_steps
        values_last[dones] = 0  # the value of terminal states is zero

        # Normalize the rewards
        # values 是states_t 经过  actions_t 后 得到的奖励,即 r_t
        values_target_t = torch.tensor(values, dtype=torch.float32).to(self.device)
        std, mean = torch.std_mean(values_target_t)
        # values_target_t 归一化
        values_target_t -= mean
        values_target_t /= std + 1e-6

        values_target_t += values_last
        # values_target_t 表示TD target
        return states_t, actions_t, values_target_t


class DiscreteActorCritic(Agent):
    """
    Agent that has a network that predicts both the action probabilities and the value
    of the state. The value is used to calculate the Advantage (A) of and action given
    the state -> A(s,a) = Q(s,a) - V(s).
    """

    def __init__(
        self,
        env: gym.Env,
        test_env: gym.Env,
        net: nn.Module,
        device: torch.device,
        gamma: float,
        beta_entropy: float,
        lr: float,
        n_steps: int,
        batch_size: int,
        chk_path: str,
        optimizer: str = "adam",
        apply_softmax: bool = True,
    ):
        self.apply_softmax = apply_softmax
        super().__init__(
            env=env,
            test_env=test_env,
            net=net,
            device=device,
            gamma=gamma,
            beta_entropy=beta_entropy,
            lr=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            chk_path=chk_path,
            optimizer=optimizer,
        )

    def train_net(
        self, states: torch.Tensor, actions: torch.Tensor, values_target: torch.Tensor
    ) -> None:
        # 清空梯度
        self.optimizer.zero_grad()
        #  得到 actor， critic  = logits， values， 对一批数据进行训练
        logits, values = self.net(states)
        # 先看torch.squeeze() 这个函数主要对数据的维度进行压缩，
        # 去掉维数为1的的维度，比如是一行或者一列这种，
        # 一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。
        # squeeze(a)就是将a中所有为1的维度删掉。不为1的维度没有影响。a.squeeze(N) 就是去掉a中指定的维数为一的维度。
        # 还有一种形式就是b=torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。

        values = values.squeeze()
        # 定义损失函数
        # 即TD Error  values为 v(s_{t},w) values_target 为y_{t}
        loss_value = F.mse_loss(values_target, values)

        self.value_loss = loss_value.item()

        actions = torch.as_tensor(actions, dtype=torch.int64)
        # actor  log_softmax 转换
        log_prob_actions = F.log_softmax(logits, dim=1)
        # unsqueeze 就是在指定的位置插入一个维度，有两个参数
        # 以 actions.unsqueeze(1) 为索引，按axis = 1， 在log_prob_actions选取元素
        log_prob_chosen = log_prob_actions.gather(1, actions.unsqueeze(1)).squeeze()
        # detach()返回一个新的tensor，是从当前计算图中分离下来的，
        # 但是仍指向原变量的存放位置，其grad_fn=None且requires_grad=False，
        # 得到的这个tensor永远不需要计算其梯度，不具有梯度grad，即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。

        advantage = values_target - values.detach()
        loss_policy = (log_prob_chosen * advantage).mean()
        self.policy_loss = loss_policy.item()

        prob_actions = F.softmax(logits, dim=1)  # 对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1
        entropy_t = -(prob_actions * log_prob_actions).sum(dim=1).mean()
        loss_entropy = -self.beta_entropy * entropy_t
        self.entropy_loss = loss_entropy.item()

        loss_total = -loss_policy + loss_value + loss_entropy
        # 反向传播 求道
        loss_total.backward()
        #  求最小
        self.optimizer.step()
        self.total_loss = loss_total.item()

    def _value_state(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # 返回critic 的值
            values = self.net(states)[1].squeeze()
        return values

    def _get_train_policy(self) -> BasePolicy:
        return DiscreteCategoricalDistributionPolicy(
            net=self.net,
            device=self.device,
            apply_softmax=self.apply_softmax,
            net_index=0,
            add_batch_dim=True,
        )

    def _get_test_policy(self) -> BasePolicy:
        return DiscreteGreedyPolicy(
            net=self.net,
            device=self.device,
            net_index=0,
            add_batch_dim=True,
        )


class ContinuosActorCritic(Agent):
    """
    Agent that has a network that predicts both the action probabilities and the value
    of the state. The value is used to calculate the Advantage (A) of and action given
    the state -> A(s,a) = Q(s,a) - V(s).
    """

    def __init__(
        self,
        env: gym.Env,
        test_env: gym.Env,
        net: nn.Module,
        device: torch.device,
        gamma: float,
        beta_entropy: float,
        lr: float,
        n_steps: int,
        batch_size: int,
        chk_path: str,
        optimizer: str = "adam",
    ):
        super().__init__(
            env=env,
            test_env=test_env,
            net=net,
            device=device,
            gamma=gamma,
            beta_entropy=beta_entropy,
            lr=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            chk_path=chk_path,
            optimizer=optimizer,
        )

    def train_net(
        self, states: torch.Tensor, actions: torch.Tensor, values_target: torch.Tensor
    ) -> None:

        self.optimizer.zero_grad()
        mean, var, values = self.net(states)
        values = values.squeeze()

        loss_value = F.mse_loss(values, values_target)
        self.value_loss = loss_value.item()

        advantage = (values_target - values.detach()).unsqueeze(-1)
        log_prob = advantage * self._logprob(mean, var, actions)
        loss_policy = -log_prob.mean()
        self.policy_loss = loss_policy.item()

        entropy_t = -(torch.log(2 * math.pi * var) + 1) / 2
        loss_entropy = self.beta_entropy * entropy_t.mean()
        self.entropy_loss = loss_entropy.item()

        loss_total = loss_policy + loss_value + loss_entropy
        loss_total.backward()
        self.optimizer.step()
        self.total_loss = loss_total.item()

    def _value_state(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            values = self.net(states)[2].squeeze()
        return values

    def _get_train_policy(self) -> BasePolicy:
        return GaussianPolicy(
            net=self.net,
            env=self.env,
            device=self.device,
            add_batch_dim=True,
        )

    def _get_test_policy(self) -> BasePolicy:
        return GaussianPolicy(
            net=self.net,
            env=self.env,
            device=self.device,
            add_batch_dim=True,
            test=True,
        )

    def _logprob(self, mean, var, actions):
        p1 = -((mean - actions) ** 2) / (2 * var.clamp(min=1e-3))
        p2 = -torch.log(torch.sqrt(2 * math.pi * var))
        return p1 + p2
