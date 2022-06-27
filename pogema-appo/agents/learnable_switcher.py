from copy import deepcopy

from pydantic import Extra

from agents.epom import EPOM, EpomConfig
from agents.replan import RePlan, RePlanConfig
from agents.switching import SwitcherBase, SwitcherBaseConfig
from policy_estimation.policy_estimator import PolicyEstimator
from pomapf.wrappers import MatrixObservationWrapper

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class LSwitcherConfig(SwitcherBaseConfig, extra=Extra.forbid):
    name: Literal['LSwitcher'] = 'LSwitcher'
    planning: RePlanConfig = RePlanConfig(fix_loops=True, add_none_if_loop=False, no_path_random=True,
                                          use_best_move=True, fix_nones=True)
    planning_path: str = "results/pe-replan"
    learning_path: str = "results/pe-epom"
    min_consequence_steps: int = 50


class LearnableSwitcher(SwitcherBase):
    def __init__(self, algo_cfg):
        super().__init__(algo_cfg)

        self.learning_estimator = PolicyEstimator()
        self.learning_estimator.load(algo_cfg.learning_path)

        self.planning_estimator = PolicyEstimator()
        self.planning_estimator.load(algo_cfg.planning_path)

        self._consequence_steps = None
        self._previous_mask = None

    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        num_agents = len(observations)
        if self._consequence_steps is None:
            self._consequence_steps = [0 for _ in range(num_agents)]
        learning_values = self.learning_estimator.predict(MatrixObservationWrapper.to_matrix(observations))
        planning_values = self.planning_estimator.predict(MatrixObservationWrapper.to_matrix(observations))

        mask = [learning_values[agent_idx] > planning_values[agent_idx] for agent_idx in range(num_agents)]
        if self._previous_mask is not None:
            for agent_idx in range(num_agents):
                if mask[agent_idx] != self._previous_mask[agent_idx]:
                    if self._consequence_steps[agent_idx] < self.cfg.min_consequence_steps:
                        mask[agent_idx] = self._previous_mask[agent_idx]
                    else:
                        self._consequence_steps[agent_idx] = 0
                self._consequence_steps[agent_idx] += 1
        self._previous_mask = mask

        return mask

    def after_reset(self):
        super().after_reset()
        self._consequence_steps = None
        self._previous_mask = None


def example():
    import gym
    # noinspection PyUnresolvedReferences
    import pomapf
    from pogema import GridConfig
    from agents.epom import EpomConfig
    gc = GridConfig(seed=42, num_agents=64, max_episode_steps=512, obs_radius=5, size=64, density=0.3,
                    map_name='mazes-s49_wc2_od50')

    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None)

    algo = LearnableSwitcher(
        LSwitcherConfig(learning=EpomConfig(path_to_weights='../results/ma'), planning_path='../results/pe-replan',
                        learning_path='../results/pe-epom'))
    obs = env.reset()
    algo.after_reset()
    dones = [False, ...]

    while not all(dones):
        action = algo.act(obs)
        obs, _, dones, info = env.step(action)
        algo.after_step(dones)


if __name__ == '__main__':
    example()
