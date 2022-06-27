try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra

from agents.utils_common import AlgoBase
from planning.prioritized import PrioritizedBase


class PrioritizedConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['Prioritized'] = 'Prioritized'
    num_process: int = 5
    device: str = 'cpu'


class Prioritized:
    def __init__(self, cfg: PrioritizedConfig):
        self.cfg = cfg
        self.agent = None
        self.env = None

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        return self.agent.act(observations, skip_agents)

    def after_step(self, dones):
        if all(dones):
            self.agent = None

    def after_reset(self, ):
        self.agent = PrioritizedBase()

    @staticmethod
    def get_additional_info():
        return {"rl_used": 0.0}


def example():
    import gym
    # noinspection PyUnresolvedReferences
    import pomapf
    from pogema import GridConfig
    gc = GridConfig(seed=42, num_agents=64, max_episode_steps=512, obs_radius=5, size=64, density=0.3,
                    map_name='mazes-s49_wc2_od50')

    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None, observation_type='MAPF')

    algo = Prioritized(PrioritizedConfig())
    obs = env.reset()
    algo.after_reset()
    dones = [False, ...]

    while not all(dones):
        action = algo.act(obs, None, dones)
        obs, _, dones, info = env.step(action)
        algo.after_step(dones)


if __name__ == '__main__':
    example()