import gym
import stable_baselines3 as sb3

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from src.pv_env import PVEnv, PVEnvDiscrete
from src.reward import RewardDeltaPower
from src.func import *
from src.reward import RewardDeltaPowerVoltage
import os
import torch as th
from typing import Callable
import numpy as np
from tqdm import tqdm

WEATHER_TRAIN_PATH = os.path.join("data", "weather_sim.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real.csv")
PVARRAY_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
AGENT_CKP_PATH_1 = os.path.join("models", "01_mppt_a2c.tar")
AGENT_CKP_PATH_2 = os.path.join("models", "02_mppt_a2c.tar")
AGENT_CKP_PATH_3 = os.path.join("models", "sb3-onnx.tar")
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.001
GAMMA = 0.9
N_STEPS = 4
BATCH_SIZE = 16
PV_PARAMS_PATH = os.path.join("parameters", "614_pvarray.json")
HiS_DATA_PATH = os.path.join("data", "600W_train_data.csv")
HiS_DATA_PATH_test = os.path.join("data", "600W_train_data_test.csv")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


my_actions = np.array([-25, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 25]) / 56

# env = gym.make("MountainCarContinuous-v0")
if __name__ == '__main__':

    env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,  # 光伏组件参数
        HiS_DATA_PATH,  # 光伏组件历史数据
        pvarray_ckp_path=PVARRAY_CKP_PATH,  # 训练过程数据存储
        states=["v_norm", "i_norm"],
        # states=["v", "i"],
        mode='Train',
        # states=["v", "i", 'v_pv'],# 训练输入，可以有多种组合
        # reward_fn=RewardDeltaPowerVoltage(2, 0.1, 1),  # 奖励函数
        reward_fn=RewardDeltaPower(4, 2),
        # actions=np.array([-10, 0, 10])/56,  # 策略函数
        actions=my_actions,
    )

    env_test = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,  # 光伏组件参数
        HiS_DATA_PATH_test,  # 光伏组件历史数据
        pvarray_ckp_path=PVARRAY_CKP_PATH,  # 训练过程数据存储
        states=["v_norm", "i_norm"],
        mode='Test',
        # states=["v", "i", 'v_pv'],# 训练输入，可以有多种组合
        # reward_fn=RewardDeltaPowerVoltage(2, 0.1, 1),  # 奖励函数
        reward_fn=RewardDeltaPower(4, 2),
        actions=my_actions   # 策略函数
    )
    # check_env(env)
    # env_name = "CartPole-v1"
    # env = gym.make(env_name)  # 导入环境
    # "MlpPolicy"定义了DDPG的策略网络是一个MLP网络
    # agent = sb3.PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log_files/')
    net_arch = [128, dict(pi=[128, 15], vf=[128])]  # 定义一个新的神经网络架构
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=28, verbose=1)  # 设置奖励阈值为200，即reward达到200后停止训练
    save_path = os.path.join('models')
    eval_callback = EvalCallback(env,
                                 callback_on_new_best=stop_callback,  # 每次有新的最好的模型都会运行stop_callback
                                 eval_freq=100,  # 每10000次运行一次eval_callback
                                 best_model_save_path=save_path,  # 在eval_callback上运行最好的模型将会保存于此
                                 verbose=1)
    if 0:
        agent = sb3.A2C("MlpPolicy",
                        env,
                        verbose=1,
                        n_steps=2,
                        # learning_rate=0.0008,
                        learning_rate=linear_schedule(0.0008),
                        gamma=0.06,
                        # use_sde=True,
                        # batch_size=16,
                        # normalize_advantage=True,
                        create_eval_env=True,
                        tensorboard_log='./log_files/',
                        policy_kwargs={'net_arch': net_arch,
                                       'activation_fn': th.nn.ReLU,
                                       'ortho_init': True,
                                       # 'use_sde': True
                                       }
                        # policy_kwargs=a2c_param_dict

                        )
        agent.learn(total_timesteps=200000,
                    log_interval=50,
                    n_eval_episodes=40,
                    eval_env=env,
                    tb_log_name='sb3_mppt_a2c.log_files',
                    # reset_num_timesteps=True,
                    callback=eval_callback
                    )
        agent.save(AGENT_CKP_PATH_1)

        aget2 = sb3.A2C("MlpPolicy",
                        env,
                        verbose=1,
                        n_steps=2,
                        # learning_rate=0.0008,
                        learning_rate=linear_schedule(0.0004),
                        gamma=0.05,
                        # use_sde=True,
                        # batch_size=16,
                        # normalize_advantage=True,
                        create_eval_env=True,
                        tensorboard_log='./log_files/',
                        policy_kwargs={'net_arch': net_arch,
                                       'activation_fn': th.nn.SELU,
                                       'ortho_init': True,
                                       # 'use_sde': True
                                       }
                        # policy_kwargs=a2c_param_dict

                        )
        aget2.learn(total_timesteps=200000,
                    log_interval=50,
                    n_eval_episodes=40,
                    eval_env=env,
                    tb_log_name='sb3_mppt_a2c.log_files',
                    # reset_num_timesteps=True,
                    callback=eval_callback
                    )
        aget2.save(AGENT_CKP_PATH_3)
    elif 0:
        aget3 = sb3.A2C("MlpPolicy",
                        env,
                        verbose=1,
                        n_steps=2,
                        # learning_rate=0.008,
                        learning_rate=linear_schedule(0.008),
                        gamma=0.01,
                        # use_sde=True,
                        # batch_size=16,
                        # normalize_advantage=True,
                        create_eval_env=True,
                        tensorboard_log='./log_files/',
                        policy_kwargs={'net_arch': net_arch,
                                       'activation_fn': th.nn.Tanh,
                                       'ortho_init': True,
                                       # 'use_sde': True
                                       }
                        # policy_kwargs=a2c_param_dict

                        )
        aget3.learn(total_timesteps=150000,
                    log_interval=40,
                    n_eval_episodes=40,
                    eval_env=env,
                    tb_log_name='sb3_mppt_a2c.log_files',
                    # reset_num_timesteps=True,
                    callback=eval_callback
                    )
        aget3.save(AGENT_CKP_PATH_3)
    else:
        # agent = sb3.A2C.load('models/best_model.zip', env)
        agent = sb3.A2C.load(AGENT_CKP_PATH_3)
        obs = env.reset()
        for i in tqdm(range(env.pv_gateway_history.shape[0])):
            # for i in tqdm(range(1, 41)):
            obs, _ = env.set_obs(i)
            # print('obs', obs)
            action, _states = agent.predict(obs, deterministic=True)
            ll = env.pv_gateway_history.at[env.pv_gateway_history.index[max(i, 0)], 'label']
            idx_max = env.pv_gateway_history[env.pv_gateway_history['label'] == ll]['power'].idxmax()
            p_mpp = env.pv_gateway_history.at[idx_max, 'power']
            p = env.pv_gateway_history.at[env.pv_gateway_history.index[max(i, 0)], 'power']
            v = env.pv_gateway_history.at[env.pv_gateway_history.index[max(i, 0)], 'voltage']
            v_new = np.array(env.actions) * 1e3 * env.pvarray.voc + v
            df_tmp = env.pv_gateway_history[env.pv_gateway_history['label'] == ll]
            p_new = [inter1pd_iv_curve(x, df_tmp) * x / 1e3 for x in v_new]
            p_diff = abs(p_new - p_mpp)
            if i == 33:
                print(1)
            # action = np.argmin(p_diff)
            # for i in range(len(env.actions)):

            _, reward, done, info = env.step(action)
            # env.render()
            if done:
                break
        env.render_vs_true(po=True)
        print(np.sum(env.history.reward))
# env.close()


# env = PVEnvDiscrete.from_file(
#     PV_PARAMS_PATH,
#     WEATHER_TRAIN_PATH,
#     pvarray_ckp_path=PVARRAY_CKP_PATH,
#     states=["v_norm", "i_norm", "deg"],
#     reward_fn=RewardDeltaPower(1, 0.9),
#     actions=[-10, -1, -0.1, 0, 0.1, 1, 10],
# )
# agent = sb3.A2C("MlpPolicy", env, verbose=1)
# agent.learn(100000, log_interval=100)

# obs = env.reset()
# for i in range(1000):
#     action, _states = agent.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     # env.render()
#     if done:
#         break
# env.render_vs_true(po=True)
