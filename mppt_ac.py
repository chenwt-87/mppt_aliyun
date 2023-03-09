import os

import torch

from src.agents import DiscreteActorCritic
from src.networks import DiscreteActorCriticNetwork
from src.pv_env import History, PVEnvDiscrete
from src.reward import RewardDeltaPower
from src.reward import RewardDeltaPowerVoltage
import time
# READ_SENSOR_TIME = 0

MODULE_NAME = "model_real_21.tar"
PV_PARAMS_PATH = os.path.join("parameters", "050_pvarray.json")
CHECKPOINT_PATH = os.path.join("models", MODULE_NAME)
PVARRAY_CKP_PATH = os.path.join("data", "051_pvarray_iv.json")
# 这个历史数据集里面，包含很多个辐照条件下的MPP，但是一个辐照下面的非MPP点太少，导致训练样本不够。
HiS_DATA_PATH = os.path.join("data", "data_for_train_A2C_with_label.csv")
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.002
GAMMA = 0.9
N_STEPS = 1
BATCH_SIZE = 16


if __name__ == "__main__":

    env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,  # 光伏组件参数
        HiS_DATA_PATH,   # 光伏组件历史数据
        pvarray_ckp_path=PVARRAY_CKP_PATH,  # 训练过程数据存储
        states=["v", "i", "dv_set2pv"],  # 训练输入，可以有多种组合
        reward_fn=RewardDeltaPowerVoltage(2, 0.9, 1),  # 奖励函数
        actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10],  # 策略函数
    )
    test_env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,
        HiS_DATA_PATH,
        pvarray_ckp_path=PVARRAY_CKP_PATH,
        states=["v", "i", "dv_set2pv"],
        reward_fn=RewardDeltaPowerVoltage(2, 0.9, 1),
        actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10],
    )
    device = torch.device("cpu")
    net = DiscreteActorCriticNetwork(
        input_size=env.observation_space.shape[0], n_actions=env.action_space.n
    ).to(device)
    agent = DiscreteActorCritic(
        env=env,
        test_env=test_env,
        net=net,
        device=device,
        gamma=GAMMA,
        beta_entropy=ENTROPY_BETA,
        lr=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        chk_path=CHECKPOINT_PATH,
    )

    # 训练模型
    # agent.learn(steps=50000, verbose_every=10, save_every=1000)

    agent.exp_train_source.play_episode()
    env.render_vs_true(po=True)
    env.render(["dv"])
    agent.plot_performance(["entropy_loss"])

