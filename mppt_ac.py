import os

import torch
from torch.utils.tensorboard import SummaryWriter

# 建立一个保存数据用的东西，save是输出的文件名

from src.agents import DiscreteActorCritic
from src.networks import DiscreteActorCriticNetwork
from src.pv_env import History, PVEnvDiscrete
from src.reward import RewardDeltaPower
from src.reward import RewardDeltaPowerVoltage
from src.logger import *
import numpy as np
import time

# READ_SENSOR_TIME = 0

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODULE_NAME = "model_real_6191.tar"
PV_PARAMS_PATH = os.path.join("parameters", "614_pvarray.json")
CHECKPOINT_PATH = os.path.join("models", MODULE_NAME)
PVARRAY_CKP_PATH = os.path.join("data", "051_pvarray_iv.json")
# 这个历史数据集里面，包含很多个辐照条件下的MPP，但是一个辐照下面的非MPP点太少，导致训练样本不够。
HiS_DATA_PATH_TRAIN = os.path.join("data", "600W_train_data_test.csv")
HiS_DATA_PATH_TEST = os.path.join("data", "600W_train_data_test.csv")
LEARNING_RATE = 0.01
ENTROPY_BETA = 0.02
GAMMA = 0.95
N_STEPS = 1
BATCH_SIZE = 238

my_actions = np.array([-25, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 25]) / 56

if __name__ == "__main__":
    env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,  # 光伏组件参数
        HiS_DATA_PATH_TRAIN,  # 光伏组件历史数据
        pvarray_ckp_path=PVARRAY_CKP_PATH,  # 训练过程数据存储
        mode='Train',
        states=["v_norm", "i_norm"],
        # states=["v", "i", 'v_pv'],# 训练输入，可以有多种组合
        # reward_fn=RewardDeltaPowerVoltage(2, 0.9, 1),  # 奖励函数
        reward_fn=RewardDeltaPower(4, 2),
        # actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10],  # 策略函数
        actions=my_actions,  # 策略函数
    )
    test_env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,
        HiS_DATA_PATH_TEST,
        pvarray_ckp_path=PVARRAY_CKP_PATH,
        mode='Test',
        states=["v_norm", "i_norm", 'v_pv'],
        # states=["v", "i", 'v_pv'],  # 训练输入，可以有多种组合
        reward_fn=RewardDeltaPowerVoltage(2, 0.9, 1),
        # reward_fn=RewardDeltaPower(4, 2),
        actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10],
    )
    device = torch.device("cpu")
    net = DiscreteActorCriticNetwork(
        input_size=env.observation_space.shape[0], n_actions=env.action_space.n
    ).to(device)
    dummy_input = torch.rand(16, 2)  # 假设输入13张1*28*28的图片
    graph = SummaryWriter()
    graph.add_graph(net, dummy_input)
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
    # env.pv_gateway_history.shape[0]
    set_log()
    agent.learn(steps=env.pv_gateway_history.shape[0], verbose_every=10, save_every=100)
    # agent.learn(steps=12000, verbose_every=100, save_every=100)

    # agent.exp_test_source.play_episode()
    # test_env.render_vs_true(po=True, source_tag='test')
    # test_env.render(["v_pv", 'dp_act'], 'test')

    agent.exp_train_source.play_episode()
    env.render_vs_true(po=True, source_tag='train')
    env.render(["dv"], 'train')
    env.render(['dp_act'], 'train')
