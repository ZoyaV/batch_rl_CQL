import gym
from gym.spaces import Box
from pogema import GridConfig
from pogema.animation import AnimationMonitor
from pogema.envs import Pogema
from pogema.integrations.sample_factory import IsMultiAgentWrapper, AutoResetWrapper
from pogema.wrappers.multi_time_limit import MultiTimeLimit

from pomapf.wrappers import RewardShaping, MultiMapWrapper, MetricsWrapper
from pomapf.fog_of_war import FogPOMAPF, FogAnimationMonitor


class FullStateNotAvailableError(Exception):
    pass


class POMAPF(Pogema):

    def __init__(self, grid_config: GridConfig):
        super().__init__(config=grid_config)
        full_size = self.config.obs_radius * 2 + 1
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            obstacles=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
            agents=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

    def _obs(self):

        results = []
        agents_xy_relative = self.grid.get_agents_xy_relative()
        targets_xy_relative = self.grid.get_targets_xy_relative()

        for agent_idx in range(self.config.num_agents):
            result = {}
            result['obstacles'] = self.grid.get_obstacles_for_agent(agent_idx)

            result['agents'] = self.grid.get_positions(agent_idx)
            result['xy'] = agents_xy_relative[agent_idx]
            result['target_xy'] = targets_xy_relative[agent_idx]
            results.append(result)
        return results

    def step(self, action: list):
        assert len(action) == self.config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.config.num_agents)]

        dones = []

        used_cells = {}

        agents_xy = self.grid.get_agents_xy()
        for agent_idx, (x, y) in enumerate(agents_xy):
            if self.active[agent_idx]:
                dx, dy = self.config.MOVES[action[agent_idx]]
                used_cells[x + dx, y + dy] = 'blocked' if (x + dx, y + dy) in used_cells else 'visited'
                used_cells[x, y] = 'blocked'
        for agent_idx in range(self.config.num_agents):
            if self.active[agent_idx]:
                x, y = agents_xy[agent_idx]
                dx, dy = self.config.MOVES[action[agent_idx]]
                if used_cells.get((x + dx, y + dy), None) != 'blocked':
                    self.grid.move(agent_idx, action[agent_idx])

            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            dones.append(on_goal)

        for agent_idx in range(self.config.num_agents):
            infos[agent_idx]['is_active'] = self.active[agent_idx]

            if self.grid.on_goal(agent_idx):
                self.grid.hide_agent(agent_idx)
                self.active[agent_idx] = False

        obs = self._obs()
        return obs, rewards, dones, infos

    def get_obstacles(self, ignore_borders=False):
        raise FullStateNotAvailableError

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return FullStateNotAvailableError

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return FullStateNotAvailableError

    def get_state(self, ignore_borders=False, as_dict=False):
        return FullStateNotAvailableError


class MAPF(POMAPF):
    def __init__(self, grid_config: GridConfig):
        super().__init__(grid_config=grid_config)

    def _obs(self):
        results = super()._obs()
        global_obstacles = self.grid.get_obstacles()
        global_agents_xy = self.grid.get_agents_xy()
        global_targets_xy = self.grid.get_targets_xy()

        for agent_idx in range(self.config.num_agents):
            result = results[agent_idx]
            result.update(global_obstacles=global_obstacles)
            result['global_xy'] = global_agents_xy[agent_idx]
            result['global_target_xy'] = global_targets_xy[agent_idx]

        return results


def make_pomapf(grid_config, with_animations=False, auto_reset=True, egocentric_idx=None, observation_type='POMAPF'):
    if observation_type == 'FOG_OF_WAR':
        env = FogPOMAPF(grid_config=grid_config)
    elif observation_type == 'POMAPF':
        env = POMAPF(grid_config=grid_config)
    else:
        env = MAPF(grid_config=grid_config)
    env = MultiTimeLimit(env, grid_config.max_episode_steps)
    if with_animations:
        if observation_type == 'FOG_OF_WAR':
            env = FogAnimationMonitor(env, egocentric_idx=None)
        else:
            env = AnimationMonitor(env, egocentric_idx=None)

    env = MetricsWrapper(env)
    env = RewardShaping(env)
    env.update_group_name(group_name='episode_extra_stats')
    env = IsMultiAgentWrapper(env)
    env = MultiMapWrapper(env)
    if auto_reset:
        env = AutoResetWrapper(env)

    return env
