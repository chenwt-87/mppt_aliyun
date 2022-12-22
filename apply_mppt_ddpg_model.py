import torch

from mppt_ddpg import *

CHECKPOINT_PATH = os.path.join("models", "02_mppt_ac.tar")

if __name__ == '__main__':
    checkpoint = sb3.DDPG.load(CHECKPOINT_PATH)
    env = PVEnv.from_file(
        PV_PARAMS_PATH,
        WEATHER_TRAIN_PATH,
        pvarray_ckp_path=PVARRAY_CKP_PATH,
        states=["g", "t", "v_norm", "i_norm", "dv"],
        reward_fn=RewardDeltaPower(2, 0.9),
    )
    model = sb3.DDPG.load(CHECKPOINT_PATH)
    env.reset()
    obs = torch.tensor([1000, 20, 0.8, 0.2, 2])
    action, _states = model.predict(obs, deterministic=True)
    new_obs, reward, done, info = env.step(action)
    print(new_obs)
    print(info)
    print(reward)
