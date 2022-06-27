try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra

from agents.replan import RePlanConfig
from agents.switching import SwitcherBaseConfig, SwitcherBase


class ASwitcherConfig(SwitcherBaseConfig, extra=Extra.forbid):
    name: Literal['ASwitcher'] = 'ASwitcher'
    planning: RePlanConfig = RePlanConfig(fix_loops=True, add_none_if_loop=True, no_path_random=False,
                                          use_best_move=False, fix_nones=False)


class AssistantSwitcher(SwitcherBase):

    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        return [a is None for a in planning_actions]


def example():
    import gym
    # noinspection PyUnresolvedReferences
    import pomapf
    from pogema import GridConfig
    gc = GridConfig(seed=42, num_agents=64, max_episode_steps=512, obs_radius=5, size=64, density=0.3,
                    map_name='mazes-s49_wc2_od50')

    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None)

    from agents.epom import EpomConfig
    algo = AssistantSwitcher(ASwitcherConfig(learning=EpomConfig(path_to_weights='../results/ma')))
    obs = env.reset()
    algo.after_reset()
    dones = [False, ...]

    while not all(dones):
        action = algo.act(obs)
        obs, _, dones, info = env.step(action)
        algo.after_step(dones)


if __name__ == '__main__':
    example()
