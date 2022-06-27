try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra

from agents.replan import RePlanConfig
from agents.switching import SwitcherBaseConfig, SwitcherBase


class HSwitcherConfig(SwitcherBaseConfig, extra=Extra.forbid):
    name: Literal['HSwitcher'] = 'HSwitcher'
    planning: RePlanConfig = RePlanConfig(fix_loops=True, add_none_if_loop=False, no_path_random=True,
                                          use_best_move=True, fix_nones=True)

    num_agents_to_switch: int = 6


class HeuristicSwitcher(SwitcherBase):

    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        return [obs['agents'].sum().sum() > self.cfg.num_agents_to_switch for obs in observations]


def example():
    import gym
    # noinspection PyUnresolvedReferences
    import pomapf
    from pogema import GridConfig
    from agents.epom import EpomConfig
    gc = GridConfig(seed=42, num_agents=64, max_episode_steps=512, obs_radius=5, size=64, density=0.3,
                    map_name='mazes-s49_wc2_od50')

    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None)

    algo = HeuristicSwitcher(HSwitcherConfig(learning=EpomConfig(path_to_weights='../results/ma')))
    obs = env.reset()
    algo.after_reset()
    dones = [False, ...]

    while not all(dones):
        action = algo.act(obs)
        obs, _, dones, info = env.step(action)
        algo.after_step(dones)


if __name__ == '__main__':
    example()
