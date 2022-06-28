import sys

print(sys.path)
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import argparse, os
import numpy as np
import pickle
import h5py
import gym
import sys
sys.path.append("../pogema-appo/")
sys.path.append("./pogema-appo/")
sys.path.append("../pogema-appo")
sys.path.append("./pogema-appo")

import pomapf
from pogema import GridConfig

from pomapf.wrappers import MatrixObservationWrapper

import torch


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
        obs = MatrixObservationWrapper.to_matrix(observations)
        obs = obs[0]['obs']
        obs = np.asarray(obs)
        return obs.reshape(-1,), reward[0], done[0], info[0]


def load_buffer(dataset, replay_buffer):
  #  b = dataset['observations'].shape[0]//6
    replay_buffer._observations = dataset['observations'][:,:]
    print("obs shape", dataset['observations'].shape)
    print("memory use", replay_buffer._observations.nbytes)
    replay_buffer._next_obs = dataset['next_observations'][:,:]
    replay_buffer._actions = dataset['actions'][:]
    print(type(replay_buffer._next_obs))
    print("act shape", dataset['actions'].shape)
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards'][:]), 1)
    print(" reward shape", dataset['rewards'].shape)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals'][:]), 1)
    print("done shape", dataset['terminals'].shape)
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum().shape)
    replay_buffer._top = replay_buffer._size


def experiment(variant):
    gc = GridConfig(seed=None, num_agents=1, max_episode_steps=64, obs_radius=5, size=128, density=0.3)
    eval_env = ObsActionWrapper(
        gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None,
                 observation_type='MAPF'))
    expl_env = eval_env

    obs = eval_env.reset()
    print(obs)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    print(obs_dim, action_dim)

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
  #  import pickle
   # with open('examples/data.pickle', 'rb') as f:
     #  data = pickle.load(f)
    data = h5py.File('examples/data_v20.hdf5', 'r')
    load_buffer(data, replay_buffer)

    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    print("____________________")
    print(torch.cuda.is_available())
    print()
    print("____________________")
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,
        env_name='Hopper-v2',
        sparse_reward=False,
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=40000,
            num_qs=2,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=1.0,

            # lagrange
            with_lagrange=True,  # Defaults to true
            lagrange_thresh=10.0,

            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='hopper-medium-v0')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--max_q_backup", type=str,
                        default="False")  # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--deterministic_backup", type=str,
                        default="True")  # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--policy_eval_start", default=40000,
                        type=int)  # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument('--min_q_weight', default=1.0,
                        type=float)  # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--policy_lr', default=1e-4, type=float)  # Policy learning rate
    parser.add_argument('--min_q_version', default=3, type=int)  # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho))
    parser.add_argument('--lagrange_thresh', default=5.0,
                        type=float)  # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    if args.lagrange_thresh < 0.0:
        variant['trainer_kwargs']['with_lagrange'] = False

    variant['buffer_filename'] = None

    variant['load_buffer'] = True
    variant['env_name'] = args.env
    variant['seed'] = args.seed

    rnd = np.random.randint(0, 1000000)
    setup_logger(os.path.join('CQL_offline_pogema_runs', str(rnd)), variant=variant,
                 base_log_dir='./random_expert_pogema_runs')
    ptu.set_gpu_mode(True)
    experiment(variant)
