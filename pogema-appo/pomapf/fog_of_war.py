import gym
import numpy as np
from pogema import GridConfig
from gym.spaces import Box
from pogema.animation import AnimationMonitor, AnimationSettings, GridHolder
import drawSvg
from copy import deepcopy

from pogema.envs import Pogema


def line_of_sight(i1, j1, i2, j2, obstacles, agents):
    delta_i = abs(i1 - i2)
    delta_j = abs(j1 - j2)
    step_i = 1 if i1 < i2 else -1
    step_j = 1 if j1 < j2 else -1
    error = 0
    i = i1
    j = j1
    if delta_i == 0:
        for j in range(j1, j2, step_j):
            if obstacles[i][j] or (i, j) in agents:
                return False
        return True
    elif delta_j == 0:
        for i in range(i1, i2, step_i):
            if obstacles[i][j] or (i, j) in agents:
                return False
        return True
    if delta_i > delta_j:
        for i in range(i1, i2, step_i):
            if obstacles[i][j] or (i, j) in agents:
                return False
            error += delta_j
            if (error << 1) > delta_i:
                if ((error << 1) - delta_j) < delta_i and (obstacles[i + step_i][j] or (i + step_i, j) in agents):
                    return False
                elif ((error << 1) - delta_j) > delta_i and (obstacles[i][j + step_j] or (i, j + step_j) in agents):
                    return False
                j += step_j
                error -= delta_i
    else:
        for j in range(j1, j2, step_j):
            if obstacles[i][j] or (i, j) in agents:
                return False
            error += delta_i
            if (error << 1) > delta_j:
                if ((error << 1) - delta_i) < delta_j and (obstacles[i][j + step_j] or (i, j + step_j) in agents):
                    return False
                elif ((error << 1) - delta_i) > delta_j and (obstacles[i + step_i][j] or (i + step_i, j) in agents):
                    return False
                i += step_i
                error -= delta_j
    return True


class FogAnimationMonitor(AnimationMonitor):
    def __init__(self, env, animation_settings=AnimationSettings(), egocentric_idx: int = None):
        super().__init__(env, animation_settings, egocentric_idx)

    def create_obstacles(self, grid_holder):
        gh = grid_holder
        cfg = self.cfg

        result = []
        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                if gh.obstacles[x][y] != self.grid_cfg.FREE:
                    obs_settings = {}
                    obs_settings.update(x=cfg.draw_start + i * cfg.scale_size - cfg.r,
                                        y=cfg.draw_start + j * cfg.scale_size - cfg.r,
                                        width=cfg.r * 2,
                                        height=cfg.r * 2,
                                        rx=cfg.rx,
                                        fill=self.cfg.obstacle_color)

                    if gh.egocentric_idx is not None and cfg.egocentric_shaded:
                        initial_positions = gh.agents_xy_history[0] if gh.agents_xy_history else gh.agents_xy
                        ego_x, ego_y = initial_positions[gh.egocentric_idx]
                        agents_pos = initial_positions.copy()
                        agents_pos.remove((ego_x, ego_y))
                        if not self.check_in_radius(x, y, ego_x, ego_y, self.grid_cfg.obs_radius) or not line_of_sight(ego_x, ego_y, x, y, gh.obstacles, agents_pos):
                            obs_settings.update(opacity=cfg.shaded_opacity)

                    result.append(drawSvg.Rectangle(**obs_settings))
        return result

    def animate_obstacles(self, obstacles, egocentric_idx, grid_holder):
        gh: GridHolder = grid_holder
        obstacle_idx = 0
        cfg = self.cfg

        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                if gh.obstacles[x][y] == self.grid_cfg.FREE:
                    continue
                opacity = []
                seen = set()
                for step_idx, agents_xy in enumerate(gh.agents_xy_history[:gh.episode_length]):
                    ego_x, ego_y = agents_xy[egocentric_idx]
                    agents_pos = agents_xy.copy()
                    agents_pos.remove((ego_x, ego_y))
                    if self.check_in_radius(x, y, ego_x, ego_y, self.grid_cfg.obs_radius) and line_of_sight(ego_x, ego_y, x, y, gh.obstacles, agents_pos):
                        seen.add((x, y))
                    if (x, y) in seen:
                        opacity.append(str(1.0))
                    else:
                        opacity.append(str(cfg.shaded_opacity))

                obstacle = obstacles[obstacle_idx]
                obstacle.appendAnim(self.compressed_anim('opacity', opacity, cfg.time_scale))

                obstacle_idx += 1

    def animate_agents(self, agents, egocentric_idx, grid_holder):
        gh: GridHolder = grid_holder
        cfg = self.cfg
        for agent_idx, agent in enumerate(agents):
            x_path = []
            y_path = []
            opacity = []
            for agents_xy in gh.agents_xy_history[:gh.episode_length]:
                x, y = agents_xy[agent_idx]
                x_path.append(str(cfg.draw_start + y * cfg.scale_size))
                y_path.append(str(-cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size))

                if egocentric_idx is not None:
                    ego_x, ego_y = agents_xy[egocentric_idx]
                    agents_pos = agents_xy.copy()
                    agents_pos.remove((ego_x, ego_y))
                    if self.check_in_radius(x, y, ego_x, ego_y, self.grid_cfg.obs_radius) and line_of_sight(ego_x, ego_y, x, y, gh.obstacles, agents_pos):
                        opacity.append('1.0')
                    else:
                        opacity.append(str(cfg.shaded_opacity))

            visibility = []
            for dones in gh.agents_done_history[:gh.episode_length]:
                visibility.append('hidden' if dones[agent_idx] else "visible")

            agent.appendAnim(self.compressed_anim('cy', y_path, cfg.time_scale))
            agent.appendAnim(self.compressed_anim('cx', x_path, cfg.time_scale))
            agent.appendAnim(self.compressed_anim('visibility', visibility, cfg.time_scale))
            if opacity:
                agent.appendAnim(self.compressed_anim('opacity', opacity, cfg.time_scale))


class FogPOMAPF(Pogema):
    def __init__(self, grid_config: GridConfig):
        super().__init__(config=grid_config)
        full_size = self.config.obs_radius * 2 + 1
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            obstacles=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
            agents=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
        )

    @staticmethod
    def add_fog(obs):
        results = deepcopy(obs)
        for k in range(len(obs)):
            c = len(obs[k]['obstacles']) // 2
            agents = np.transpose(np.nonzero(obs[k]['agents'])) - [c, c]
            other_agents = set()
            for n in range(len(agents)):
                other_agents.add((agents[n][0], agents[n][1]))
            for i in range(0, len(obs[k]['obstacles'])):
                for j in range(0, len(obs[k]['obstacles'])):
                    if not line_of_sight(c, c, i, j, obs[k]['obstacles'], other_agents):
                        results[k]['obstacles'][i][j] = 0
                        results[k]['agents'][i][j] = 0
        return results

    def _obs(self):
        results = []
        agents_xy_relative = self.grid.get_agents_xy_relative()
        targets_xy_relative = self.grid.get_targets_xy_relative()

        for agent_idx in range(self.config.num_agents):
            result = {'obstacles': self.grid.get_obstacles_for_agent(agent_idx),
                      'agents': self.grid.get_positions(agent_idx),
                      'xy': agents_xy_relative[agent_idx],
                      'target_xy': targets_xy_relative[agent_idx]}
            results.append(result)
        results = self.add_fog(results)
        return results
