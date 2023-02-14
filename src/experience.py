import collections
import gym
from src.policies import BasePolicy

Experience = collections.namedtuple(
    "Experience", ["state", "action", "reward", "last_state"]
)
ExperienceDiscounted = collections.namedtuple(
    "Experience",
    ["state", "action", "reward", "last_state", "discounted_reward", "steps"],
)


class ExperienceSorce:
    def __init__(self, env: gym.Env, policy: BasePolicy):
        self.env = env
        self.policy = policy
        self.obs = self.env.reset()
        self.done = False
        self.render = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.play_step()

    def play_step(self):
        if self.done:
            self.obs = self.env.reset()
            self.done = False
        # obs =  ['v_norm', 'i_norm', 'dv']
        obs = self.obs

        # 基于obs，计算当前状态，进入到网络结构  转入policies.py line 46 行
        action = self.policy(obs)
        # 基于action 【int】，选择电压增量，计算新的状态和 reward , 调用  pv_env.py  line 97 的 step(函数)
        new_obs, reward, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
        if done:
            self.done = True
            return Experience(state=obs, action=action, reward=reward, last_state=None)
        self.obs = new_obs
        return Experience(state=obs, action=action, reward=reward, last_state=new_obs)

    def play_episode(self):
        ep_history = []
        self.obs = self.env.reset()
        iter_n = 0
        while True:
            iter_n += 1
            experience = self.play_step()
            ep_history.append(experience)

            if experience.last_state is None:
                # print('计算次数', iter_n)
                return ep_history

    def play_episodes(self, episodes):
        return [self.play_episode() for _ in range(episodes)]


class ExperienceSorceEpisodes(ExperienceSorce):
    def __init__(self, env: gym.Env, policy: BasePolicy, episodes: int):
        super().__init__(env, policy)

        self.max_episodes = episodes

    def __next__(self):
        return self.play_episodes(self.max_episodes)


class ExperienceSorceDiscounted(ExperienceSorce):
    def __init__(self, env: gym.Env, policy: BasePolicy, gamma: float, n_steps: int):
        super().__init__(env, policy)

        self.gamma = gamma
        self.max_steps = n_steps

    def __next__(self):
        return self.play_n_steps()

    def play_n_steps(self):
        history = []
        discounted_reward = 0.0
        reward = 0.0

        for step_idx in range(self.max_steps):
            # 【state， action， reward， last_state】

            self.env.step_counter += 1
            for act in range(6):
                exp = self.play_step()
                # print('依据网络计,', exp, step_idx)
                reward += exp.reward
                discounted_reward += exp.reward * self.gamma ** (step_idx)
                history.append(exp)

            if exp.last_state is None:
                break

        return ExperienceDiscounted(
            state=history[0].state,
            action=history[0].action,
            last_state=history[-1].last_state,
            reward=reward,
            discounted_reward=discounted_reward,
            steps=step_idx + 1,
        )

    def play_n_steps_pred(self):
        history = []
        discounted_reward = 0.0
        reward = 0.0

        for step_idx in range(self.max_steps):

            self.env.step_counter += 1
            exp = self.play_step()
            # print('依据网络计,', exp, step_idx)
            reward += exp.reward
            discounted_reward += exp.reward * self.gamma ** (step_idx)
            history.append(exp)

            if exp.last_state is None:
                break

        return ExperienceDiscounted(
            state=history[0].state,
            action=history[0].action,
            last_state=history[-1].last_state,
            reward=reward,
            discounted_reward=discounted_reward,
            steps=step_idx + 1,
        )

    def play_episode(self):
        ep_history = []
        self.obs = self.env.reset()
        iter_num = 0
        while True:
            iter_num += 1
            experience = self.play_n_steps_pred()
            ep_history.append(experience)
            print('计算次数', iter_num)
            if experience.last_state is None:
                return ep_history


class ExperienceSorceDiscountedSteps(ExperienceSorceDiscounted):
    def __init__(
        self,
        env: gym.Env,
        policy: BasePolicy,
        gamma: float,
        n_steps: int,  # 训练次数
        steps: int,  # batch
    ):
        super().__init__(env, policy, gamma, n_steps)

        self.steps = steps

    def __next__(self):
        # exp 【state,action,reward,last_state,discounted_reward,steps】
        return [self.play_n_steps() for _ in range(self.steps)]


class ExperienceSorceDiscountedEpisodes(ExperienceSorceDiscounted):
    def __init__(
        self,
        env: gym.Env,
        policy: BasePolicy,
        gamma: float,
        n_steps: int,
        episodes: int,
    ):
        super().__init__(env, policy, gamma, n_steps)

        self.max_episodes = episodes

    def __next__(self):
        return self.play_episodes(self.max_episodes)
