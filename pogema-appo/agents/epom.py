import json
from copy import deepcopy
from os.path import join
from pathlib import Path
from typing import Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch

from pydantic import Extra

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict

from agents.utils_common import AlgoBase
from learning.epom_config import Environment
from pomapf.grid_memory import MultipleGridMemory
from pomapf.wrappers import MatrixObservationWrapper

from training_run import validate_config, register_custom_components


class EpomConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['EPOM'] = 'EPOM'
    path_to_weights: str = "results/ma"

    noise_tau: Optional[int] = None
    noise_variance: Optional[float] = None


class EPOM:
    def __init__(self, algo_cfg):
        self.algo_cfg: EpomConfig = algo_cfg

        path = algo_cfg.path_to_weights
        device = algo_cfg.device
        register_custom_components()

        self.path = path
        self.env = None
        config_path = join(path, 'cfg.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        exp, flat_config = validate_config(config['full_config'])
        algo_cfg = flat_config

        env = create_env(algo_cfg.env, cfg=algo_cfg, env_config={})
        actor_critic = create_actor_critic(algo_cfg, env.observation_space, env.action_space)
        env.close()

        if device == 'cpu' or not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        self.device = device

        # actor_critic.share_memory()
        actor_critic.model_to_device(device)
        policy_id = algo_cfg.policy_index
        checkpoints = join(path, f'checkpoint_p{policy_id}')
        checkpoints = LearnerWorker.get_checkpoints(checkpoints)
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        self.ppo = actor_critic
        self.device = device
        self.cfg = algo_cfg

        self.rnn_states = None
        self.mgm = MultipleGridMemory()
        self._step = 0
        self._hidden_noise = 0.0

    def after_reset(self):
        self.mgm.clear()
        self._step = 0

    def get_additional_info(self):
        result = {"rl_used": 1.0, "hidden_noise": self._hidden_noise}
        self._hidden_noise = 0.0
        return result

    def get_name(self):
        return Path(self.path).name

    def act(self, observations, rewards=None, dones=None, infos=None):
        observations = deepcopy(observations)
        if self.rnn_states is None or len(self.rnn_states) != len(observations):
            self.rnn_states = torch.zeros([len(observations), get_hidden_size(self.cfg)], dtype=torch.float32,
                                          device=self.device)
        env_cfg: Environment = Environment(**self.cfg.full_config['environment'])
        self.mgm.update(observations)
        gm_radius = env_cfg.grid_memory_obs_radius
        self.mgm.modify_observation(observations, obs_radius=gm_radius if gm_radius else env_cfg.grid_config.obs_radius)
        observations = MatrixObservationWrapper.to_matrix(observations)

        with torch.no_grad():
            variance = self.algo_cfg.noise_variance
            tau = self.algo_cfg.noise_tau
            if self.rnn_states is not None and variance and tau:
                avg_noise = 0.0
                for agent_idx in range(len(self.rnn_states)):
                    if self._step and (self._step + agent_idx) % tau == 0:
                        noise = (variance ** 0.5) * torch.randn(self.rnn_states[agent_idx].shape)
                        avg_noise += noise.abs().mean().numpy()
                        noise = noise.to(self.device)
                        self.rnn_states[agent_idx] += noise
                self._hidden_noise += avg_noise / len(self.rnn_states)

            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.device).float()
            policy_outputs = self.ppo(obs_torch, self.rnn_states, with_action_distribution=True)

            self.rnn_states = policy_outputs.rnn_states
            actions = policy_outputs.actions

        self._step += 1
        result = actions.cpu().numpy()
        return result

    def clear_hidden(self, agent_idx):
        if self.rnn_states is not None:
            self.rnn_states[agent_idx] = torch.zeros([get_hidden_size(self.cfg)], dtype=torch.float32,
                                                     device=self.device)

    def after_step(self, dones):
        for agent_idx, done_flag in enumerate(dones):
            if done_flag:
                self.clear_hidden(agent_idx)

        if all(dones):
            self.rnn_states = None
            self.mgm.clear()


def example():
    import gym
    # noinspection PyUnresolvedReferences
    import pomapf
    from pogema import GridConfig
    gc = GridConfig(seed=42, num_agents=64, max_episode_steps=512, obs_radius=5, size=64, density=0.3,
                    map_name='mazes-s49_wc2_od50')

    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None)

    algo = EPOM(EpomConfig(path_to_weights='../results/ma'))
    obs = env.reset()
    algo.after_reset()
    dones = [False, ...]

    while not all(dones):
        action = algo.act(obs)
        obs, _, dones, info = env.step(action)
        algo.after_step(dones)


if __name__ == '__main__':
    example()
