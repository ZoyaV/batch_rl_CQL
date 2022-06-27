from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import numpy as np
import uuid
from rlkit.core import logger
from rlkit.util.video import dump_video
import sys
sys.path.append("../pogema-appo/")
sys.path.append("./pogema-appo/")
sys.path.append("../pogema-appo")
sys.path.append("./pogema-appo")

import pomapf
import gym
from pogema import GridConfig

from pomapf.wrappers import MatrixObservationWrapper
filename = str(uuid.uuid4())

class ObsActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        print(len(self.config.MOVES))
        self.action_space = gym.spaces.Box(0.0, 1.0, shape = (1,))
        full_size = self.config.obs_radius * 2 + 1
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3*full_size*full_size,))

    def reset(self, **kwargs):
        obs = MatrixObservationWrapper.to_matrix(super().reset())
        obs = obs[0]['obs']
        obs = np.asarray(obs)
        return obs.reshape( -1,)

    def step(self, action):
        right_action = 0
        if action<=0.26:
            right_action = 1
        elif action >0.26 and action<=0.53:
            right_action = 2
        elif action > 0.53 and action<=0.76:
            right_action = 3
        else:
            right_action = 4
        #if action>1:
        #    raise Exception("Life is good")
       # action = int(action +0.5)
        observations, reward, done, info = super().step([right_action])
       # info['']
        obs = MatrixObservationWrapper.to_matrix(observations)
        obs = obs[0]['obs']
        obs = np.asarray(obs)
        return obs.reshape(-1,), reward[0], done[0], info[0]


def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    gc = GridConfig(seed=None, num_agents=1, max_episode_steps=64, obs_radius=5, size=32, density=0.3)
    env = ObsActionWrapper(
        gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None,
                 observation_type='MAPF'))
  #  expl_env = eval_env
   # dump_video(env,policy, "movie.mp4", rollout)
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
        )
       # print(path)
      #  break
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
