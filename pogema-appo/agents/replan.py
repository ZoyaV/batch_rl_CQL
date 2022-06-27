try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra

from agents.utils_common import AlgoBase
from planning.replan_algo import RePlanBase, FixLoopsWrapper, NoPathSoRandomOrStayWrapper, FixNonesWrapper


class RePlanConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['RePlan'] = 'RePlan'
    num_process: int = 5
    fix_loops: bool = True
    no_path_random: bool = True
    fix_nones: bool = True
    add_none_if_loop: bool = False
    use_best_move: bool = True
    stay_if_loop_prob: float = 0.5
    max_planning_steps: int = 10000
    device: str = 'cpu'


class RePlan:
    def __init__(self, cfg: RePlanConfig):
        self.cfg = cfg
        self.agent = None
        self.fix_loops = cfg.fix_loops
        self.fix_nones = cfg.fix_nones
        self.stay_if_loop_prob = cfg.stay_if_loop_prob
        self.no_path_random = cfg.no_path_random
        self.use_best_move = cfg.use_best_move
        self.add_none_if_loop = cfg.add_none_if_loop

        self.env = None

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        return self.agent.act(observations, skip_agents)

    def after_step(self, dones):
        if all(dones):
            self.agent = None

    def after_reset(self, ):
        # self.env = env
        self.agent = RePlanBase(use_best_move=self.use_best_move, max_steps=self.cfg.max_planning_steps)

        if self.fix_loops:
            self.agent = FixLoopsWrapper(self.agent, stay_if_loop_prob=self.stay_if_loop_prob,
                                         add_none_if_loop=self.add_none_if_loop)
        if self.no_path_random:
            self.agent = NoPathSoRandomOrStayWrapper(self.agent)
        elif self.fix_nones:
            self.agent = FixNonesWrapper(self.agent)

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

    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None)

    algo = RePlan(RePlanConfig())
    obs = env.reset()
    algo.after_reset()
    dones = [False, ...]

    while not all(dones):
        action = algo.act(obs, None, dones)
        obs, _, dones, info = env.step(action)
        algo.after_step(dones)


if __name__ == '__main__':
    example()
