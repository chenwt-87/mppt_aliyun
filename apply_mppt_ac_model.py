from mppt_ac import *
VOC = 32.9
ISC = 8.21


def read_sensor():
    temp = 25.0
    irr = 200.0
    voltage = 30.0
    current = 4.34
    power = 26.0
    return temp, irr, voltage, current, power


def creat_agent(model_net, dev):
    env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,
        HiS_DATA_PATH,
        # states=["v_norm", "i_norm", "deg"],
        pvarray_ckp_path=PVARRAY_CKP_PATH,
        states=["v_norm", "i_norm", "dv"],
        reward_fn=RewardDeltaPower(2, 0.9),
        actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10],
    )

    test_env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,
        HiS_DATA_PATH,
        # states=["v_norm", "i_norm", "deg"],
        pvarray_ckp_path=PVARRAY_CKP_PATH,
        states=["v_norm", "i_norm", "dv"],
        reward_fn=RewardDeltaPower(2, 0.9),
        actions=[-10, -5, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 5, 10],
    )

    agent = DiscreteActorCritic(
        env=env,
        test_env=test_env,
        net=model_net,
        device=dev,
        gamma=GAMMA,
        beta_entropy=ENTROPY_BETA,
        lr=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        chk_path=CHECKPOINT_PATH,
    )

    return agent


def get_t_g_v_i():
    t, g, v, i, p = read_sensor()
    device = torch.device("cpu")
    model = DiscreteActorCriticNetwork(input_size=3, n_actions=13).to(device)
    CHECKPOINT_PATH = os.path.join("models", MODULE_NAME)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_agent = creat_agent(model, device)
    # obs = torch.tensor([v/VOC, i/ISC, 0])
    obs = torch.tensor([0.89238, 0.59427, 2])
    # 基于obs，计算当前状态，进入到网络结构
    action = model_agent.policy(obs)
    print(action)
    # 基于action，计算新的状态和 reward    调用  pv_env.py  line 97 的 step(函数)
    new_obs, reward, done, _ = model_agent.env.step(action)
    return new_obs, reward


if __name__ == '__main__':
    rl_obs, rl_reward = get_t_g_v_i()
    print(rl_obs)
    print(rl_reward)
