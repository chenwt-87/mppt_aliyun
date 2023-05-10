import gym
import stable_baselines3 as sb3
from src.pv_env import PVEnv, PVEnvDiscrete
from src.reward import RewardDeltaPower
import os
from tqdm import tqdm


WEATHER_TRAIN_PATH = os.path.join("data", "weather_sim.csv")
WEATHER_TEST_PATH = os.path.join("data", "weather_real.csv")
PVARRAY_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
AGENT_CKP_PATH = os.path.join("models", "02_mppt_ppo.tar")
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.001
GAMMA = 0.9
N_STEPS = 4
BATCH_SIZE = 16
PV_PARAMS_PATH = os.path.join("parameters", "614_pvarray.json")
HiS_DATA_PATH = os.path.join("data", "600W_train_data.csv")

# env = gym.make("MountainCarContinuous-v0")
if __name__ == '__main__':

    env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,  # 光伏组件参数
        HiS_DATA_PATH,  # 光伏组件历史数据
        pvarray_ckp_path=PVARRAY_CKP_PATH,  # 训练过程数据存储
        states=["v_norm", "i_norm", 'v_pv'],
        # states=["v", "i", 'v_pv'],# 训练输入，可以有多种组合
        # reward_fn=RewardDeltaPowerVoltage(2, 0.9, 1),  # 奖励函数
        reward_fn=RewardDeltaPower(4, 2),
        actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10],  # 策略函数
    )

    # "MlpPolicy"定义了DDPG的策略网络是一个MLP网络
    # agent = sb3.PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/')
    agent = sb3.A2C("MlpPolicy", env, verbose=1)
    agent.learn(total_timesteps=1000, log_interval=1, tb_log_name='sb3_mppt_a2c.log', eval_env=env)
    agent.save(AGENT_CKP_PATH)
    obs = env.reset()
    for i in tqdm(range(env.pv_gateway_history.shape[0])):
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            break
    env.render_vs_true(po=True)
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
