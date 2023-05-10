import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteActorCriticNetwork(nn.Module):
    # 网络结构
    def __init__(self, input_size, n_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 1280),
            nn.ReLU(),
        )
        torch.nn.init.normal_(self.net[0].weight, mean=0, std=1)  #
        # self.actor = nn.Linear(128, n_actions)
        self.actor = nn.Sequential(
            nn.Linear(1280, n_actions),
            nn.Sigmoid(),
        )
        # torch.nn.init.normal_(self.actor[0].weight, mean=0, std=1)  #
        # self.critic = nn.Linear(128, 1)
        self.critic = nn.Sequential(
            nn.Linear(1280, 1),
            nn.Sigmoid(),
        )
        torch.nn.init.normal_(self.critic[0].weight, mean=0, std=1)  #

    # 由于每一个神经网络模块都继承于nn.Module，因此都会实现__call__与forward函数，
    # 所以forward函数中通过for循环依次调用添加到现有模块中的子模块，最后输出经过所有神经网络层的结果。

    def forward(self, x):
        x = self.net(x)
        # 13维 , 1维
        return self.actor(x), self.critic(x)


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size: int, n_actions: int, hidden_units: int = 128):
        super().__init__()

        self.base1 = nn.Sequential(nn.Linear(input_size, hidden_units), nn.ReLU())
        self.base2 = nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
        self.base3 = nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
        self.mean = nn.Sequential(nn.Linear(hidden_units, n_actions), nn.Tanh())
        # self.mean = nn.Sequential(nn.Linear(hidden_units, n_actions))
        self.var = nn.Sequential(nn.Linear(hidden_units, n_actions), nn.Softplus())
        self.value = nn.Sequential(nn.Linear(hidden_units, 1))

    def forward(self, x):
        x = self.base1(x)
        x = self.base2(x)
        x = self.base3(x)
        return self.mean(x), self.var(x), self.value(x)
