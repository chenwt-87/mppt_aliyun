import os

import torch

from src.agents import DiscreteActorCritic
from src.networks import DiscreteActorCriticNetwork
from src.pv_env import History, PVEnvDiscrete
from src.reward import RewardDeltaPower
from src.reward import RewardDeltaPowerVoltage
# READ_SENSOR_TIME = 0

PV_PARAMS_PATH = os.path.join("parameters", "050_pvarray.json")
WEATHER_PATH = os.path.join("data", "weather_sim_online.csv")
CHECKPOINT_PATH = os.path.join("models", "model_real_063.tar")
PVARRAY_CKP_PATH = os.path.join("data", "050_pvarray_iv.json")
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.002
GAMMA = 0.9
N_STEPS = 1
BATCH_SIZE = 16


if __name__ == "__main__":

    env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,
        WEATHER_PATH,
        # states=["v_norm", "i_norm", "deg"],
        pvarray_ckp_path=PVARRAY_CKP_PATH,
        states=["v_norm", "i_norm", "dv_set2pv"],
        reward_fn=RewardDeltaPowerVoltage(2, 0.9, 1),
        actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10],
    )
    test_env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,
        WEATHER_PATH,
        # states=["v_norm", "i_norm", "deg"],
        pvarray_ckp_path=PVARRAY_CKP_PATH,
        states=["v_norm", "i_norm", "dv_set2pv"],
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

    # agent.train(steps=1_000_000, verbose_every=10_000, save_every=10_000)
    # agent.train_net(states=torch.tensor([-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10]),
    #                 actions=torch.tensor([-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10]),
    #                 values_target=torch.tensor([-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10]))
    # agent.learn(steps=50000, verbose_every=10, save_every=1000)

    agent.exp_train_source.play_episode()
    env.render_vs_true(po=True)
    env.render(["dv"])
    agent.plot_performance(["entropy_loss"])

