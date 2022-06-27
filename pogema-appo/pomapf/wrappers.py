import gym
import re
import time
from copy import deepcopy
import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box
from numpy import float32
from pogema import GridConfig

from pomapf.custom_maps import MAPS_REGISTRY


class StepDelayWrapper(gym.Wrapper):
    """
    FPS is rather slow at the end of training (when the number of active agent is low),
    thus we add some delay, which is based on the number of active agents
    """

    DELAY = 0.0002
    FROM_NUM_AGENTS = 4
    K = 1.15

    def step(self, action):
        observations, rewards, done, info = self.env.step(action)

        num_active_agents = sum([info[idx].get('is_active', False) for idx in range(len(info))])
        if num_active_agents >= self.FROM_NUM_AGENTS:
            time.sleep(self.DELAY * (num_active_agents ** self.K))

        return observations, rewards, done, info

    def _wait(self):
        self._go_wait = False
        time.sleep(self._wait_time)
        self._wait_time = 0.0


class RewardShaping(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._previous_xy = None

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        for agent_idx in range(self.env.get_num_agents()):
            reward = rewards[agent_idx]
            reward -= 0.0001
            if action[agent_idx] != 0:
                if tuple(self._previous_xy[agent_idx]) == tuple(observations[agent_idx]['xy']):
                    reward -= 0.0002
            rewards[agent_idx] = reward
            self._previous_xy[agent_idx] = observations[agent_idx]['xy']

        return observations, rewards, dones, infos

    def reset(self):
        observation = self.env.reset()
        self._previous_xy = [[0, 0] for _ in range(self.env.get_num_agents())]

        return observation


class MultiMapWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.env.config.seed)
        pattern = self.env.config.map_name

        if pattern:
            for map_name in MAPS_REGISTRY:
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.env.config)
                    cfg.map = MAPS_REGISTRY[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")

    def step(self, action):
        observations, rewards, done, info = self.env.step(action)
        cfg = self.env.unwrapped.config
        if cfg.map_name:
            for agent_idx in range(self.env.get_num_agents()):
                for key, value in list(info[agent_idx]['episode_extra_stats'].items()):
                    if key == 'Done':
                        continue
                    info[agent_idx]['episode_extra_stats'][f'{key}-{cfg.map_name.split("-")[0]}'] = value
        return observations, rewards, done, info

    def reset(self, **kwargs):
        if self._configs is not None and len(self._configs) >= 1:
            cfg = deepcopy(self._configs[self._rnd.integers(0, len(self._configs))])
            self.env.unwrapped.config = cfg

        return self.env.reset(**kwargs)


class MatrixObservationWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # full_size = self.config.obs_radius * 2 + 1
        full_size = self.env.observation_space['obstacles'].shape[0]
        self.observation_space = gym.spaces.Dict(
            obs=gym.spaces.Box(0.0, 1.0, shape=(3, full_size, full_size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

    @staticmethod
    def get_square_target(x, y, tx, ty, obs_radius):
        full_size = obs_radius * 2 + 1
        result = np.zeros((full_size, full_size))
        dx, dy = x - tx, y - ty

        dx = min(dx, obs_radius) if dx >= 0 else max(dx, -obs_radius)
        dy = min(dy, obs_radius) if dy >= 0 else max(dy, -obs_radius)
        result[obs_radius - dx, obs_radius - dy] = 1
        return result

    @staticmethod
    def to_matrix(observations):
        result = []
        obs_radius = observations[0]['obstacles'].shape[0] // 2
        for agent_idx, obs in enumerate(observations):
            result.append(
                {"obs": np.concatenate([obs['obstacles'][None], obs['agents'][None],
                                        MatrixObservationWrapper.get_square_target(*obs['xy'], *obs['target_xy'],
                                                                                   obs_radius)[None]]).astype(float32),
                 "xy": np.array(obs['xy'], dtype=float32),
                 "target_xy": np.array(obs['target_xy'], dtype=float32),
                 })
        return result

    def observation(self, observation):
        result = self.to_matrix(observation)
        return result


class MetricsWrapper(gym.Wrapper):
    def __init__(self, env, group_name='metrics'):
        super().__init__(env)
        self._ISR = None
        self._group_name = group_name
        self._ep_length = None
        self._steps = None

    def update_group_name(self, group_name):
        self._group_name = group_name

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        self._steps += 1
        for agent_idx in range(self.env.get_num_agents()):
            infos[agent_idx][self._group_name] = infos[agent_idx].get(self._group_name, {})

            if done[agent_idx]:
                infos[agent_idx][self._group_name].update(Done=True)
                if agent_idx not in self._ISR:
                    self._ISR[agent_idx] = float('TimeLimit.truncated' not in infos[agent_idx])
                if agent_idx not in self._ep_length:
                    self._ep_length[agent_idx] = self._steps
        if all(done):
            not_tl_truncated = all(['TimeLimit.truncated' not in info for info in infos])

            for agent_idx in range(self.env.get_num_agents()):
                infos[agent_idx][self._group_name].update(CSR=float(not_tl_truncated))
                infos[agent_idx][self._group_name].update(ISR=self._ISR[agent_idx])
                infos[agent_idx][self._group_name].update(ep_length=self._ep_length[agent_idx])
        return obs, reward, done, infos

    def reset(self, **kwargs):
        self._ISR = {}
        self._ep_length = {}
        self._steps = 0
        return self.env.reset(**kwargs)
