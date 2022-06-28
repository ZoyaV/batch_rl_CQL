from agents.prioritzied import Prioritized, PrioritizedConfig
from agents.replan import RePlanConfig, RePlan
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema import GridConfig
import h5py
from pomapf.wrappers import MatrixObservationWrapper
import numpy as np
import datetime
from multiprocessing import Pool

def drop_global_information(observations):
    for agent_idx, obs in enumerate(observations):
        del obs['global_target_xy']
        del obs['global_xy']
        del obs['global_obstacles']
    return observations



def example(ind = 0, saver = 'h5py'):
    gc = GridConfig(seed=None, num_agents=1, max_episode_steps=64, obs_radius=5, size=16, density=0.3)
    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None,
                   observation_type='MAPF')
    algo = Prioritized(PrioritizedConfig())
    start_point = 0
    agent_expirience = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': []}
    with h5py.File('../examples/data_v2%d.hdf5'%ind, 'w') as f:
        observations_ = f.create_dataset('observations', (100,363), maxshape=(500000000,363))
        next_observations_ = f.create_dataset('next_observations', (100,363), maxshape=(500000000,363))
        actions_ = f.create_dataset('actions', (100,1), maxshape=(500000000,1))
        rewards_ = f.create_dataset('rewards', (100,), maxshape=(500000000,))
        terminals_ = f.create_dataset('terminals', (100,), maxshape=(50000000,))
        data_count = 500000
        log_interval = 1000
        for k in range(data_count):
            if k%log_interval == 0:
                print(f"{k}/{data_count}")
            observations = env.reset()
            algo.after_reset()
            dones = [False, ...]
            obs = MatrixObservationWrapper.to_matrix(observations)
            obs = obs[0]['obs']
            obs = np.asarray(obs)
            obs = obs.reshape(-1, )
            while not all(dones):
                agent_expirience['observations'].append(obs)
                action = algo.act(observations, None, dones)
                observations, rewards, dones, info = env.step(action)
                obs = MatrixObservationWrapper.to_matrix(observations)
                obs = obs[0]['obs']
                obs = np.asarray(obs)
                obs = obs.reshape(-1, )
                agent_expirience['next_observations'].append(obs)
                agent_expirience['actions'].append([(action[0])/(4)])
                agent_expirience['rewards'].append(rewards[0])
                agent_expirience['terminals'].append(dones[0])
                algo.after_step(dones)

            add_data = len(agent_expirience['rewards'])
          #  print()
            observations_.resize((start_point+add_data,363))
            next_observations_.resize((start_point+add_data,363))
            actions_.resize((start_point+add_data,1))
            rewards_.resize((start_point+add_data,))
            terminals_.resize((start_point+add_data,))

            observations_[start_point:start_point+add_data] = agent_expirience['observations']
            next_observations_[start_point:start_point+add_data] = agent_expirience['next_observations']
            actions_[start_point:start_point+add_data] = agent_expirience['actions']
            rewards_[start_point:start_point+add_data] = agent_expirience['rewards']
            terminals_[start_point:start_point+add_data] = agent_expirience['terminals']

            agent_expirience = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [],
                                'terminals': []}
            start_point = add_data+start_point



    if saver == "pickle":
        import pickle
        for key in agent_expirience:
            agent_expirience[key] = np.asarray(agent_expirience[key])
        agent_expirience['actions'] = agent_expirience['actions'].reshape(-1, 1)
        with open('../examples/data.pickle', 'wb') as f:
            pickle.dump(agent_expirience, f)
    else:
        pass

    # with h5py.File(f'../examples/data{ind}.hdf5', 'w') as f:
    #     f.create_dataset('next_observations', data=agent_expirience['next_observations'])
    #     f.create_dataset('actions', data= agent_expirience['actions'])
    #     f.create_dataset('rewards', data=agent_expirience['rewards'])
    #     f.create_dataset('terminals', data=agent_expirience['terminals'])
    #     f.create_dataset('observations', data=agent_expirience['observations'])


if __name__ == '__main__':
    # p = Pool(2)
     a = datetime.datetime.now()
  #   with p:
     example()
     b = datetime.datetime.now()
   #  print(b-a)