from agents.prioritzied import Prioritized, PrioritizedConfig
from agents.replan import RePlanConfig, RePlan
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema import GridConfig
import h5py
from pomapf.wrappers import MatrixObservationWrapper
import numpy as np

def drop_global_information(observations):
    for agent_idx, obs in enumerate(observations):
        del obs['global_target_xy']
        del obs['global_xy']
        del obs['global_obstacles']
    return observations


def example():
    num_agents = 2
    gc = GridConfig(seed=None, num_agents=num_agents, max_episode_steps=64, obs_radius=5, size=32, density=0.3)

    # turn off with_animation to speedup generation speed
    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None,
                   observation_type='MAPF')

    agent_expirience = {'observations': [[]]*num_agents, 'next_observations': [[]]*num_agents,
                        'actions': [[]]*num_agents, 'rewards': [[]]*num_agents,
                        'terminals': [[]]*num_agents}

    # fully centralized planning approach
    algo = Prioritized(PrioritizedConfig())
    # partially observable decentralized planning approach
    # algo = RePlan(RePlanConfig())

    for _ in range(100000):
        observations = env.reset()
        algo.after_reset()
        dones = [False, ...]

        obs = MatrixObservationWrapper.to_matrix(observations)

      #  print(obs.shape)
      #  break

       # print(agent_expirience['observations'])
        while not all(dones):
            for i in range(num_agents):
               # print(obs[0]['obs'])
                obs_ = obs[i]['obs']
                #  print(obs.shape)
                obs_ = np.asarray(obs_)
                obs_ = obs_.reshape(-1, )
                agent_expirience['observations'][i].append(obs_)

            action = algo.act(observations, None, dones)
           # print(action)
            observations, rewards, dones, info = env.step(action)
           # obs = drop_global_information(observations.copy())
            obs = MatrixObservationWrapper.to_matrix(observations)
            for i in range(num_agents):
               # obs = MatrixObservationWrapper.to_matrix(observations)
                obs_ = obs[i]['obs']
                obs_ = np.asarray(obs_)
                obs_ = obs_.reshape(-1, )
             #   print(action[i])
                agent_expirience['next_observations'][i].append(obs_)
                act = 0 if action[i] is None else action[i]
                agent_expirience['actions'][i].append((act)/(4))
                agent_expirience['rewards'][i].append(rewards[i])
                agent_expirience['terminals'][i].append(dones[i])

            algo.after_step(dones)

  #  print(agent_expirience['actions'])
  #  print(agent_expirience['terminals'][-1])
    data = {'observations': [] , 'next_observations': [] ,
                        'actions': [] , 'rewards': [] ,
                        'terminals': [] }
    for i in range(num_agents):
        data['observations'] += agent_expirience['observations'][i]
        data['next_observations'] += agent_expirience['next_observations'][i]
        data['actions'] += agent_expirience['actions'][i]
        data['rewards'] += agent_expirience['rewards'][i]
        data['terminals'] += agent_expirience['terminals'][i]

    import pickle


    for key in agent_expirience:
        data[key] = np.asarray(data[key])
  #  agent_expirience['next_observations'] = agent_expirience['next_observations'].reshape(-1,1)
   # agent_expirience['observations'] = agent_expirience['observations'].reshape(-1, 1)
    data['actions'] = data['actions'].reshape(-1, 1)
    with open('../examples/data.pickle', 'wb') as f:
        pickle.dump(data, f)

    with h5py.File('../examples/data.hdf5', 'w') as f:
        f.create_dataset('next_observations', data=data['next_observations'])
        f.create_dataset('actions', data= data['actions'])
        f.create_dataset('rewards', data=data['rewards'])
        f.create_dataset('terminals', data=data['terminals'])
        f.create_dataset('observations', data=data['observations'])


if __name__ == '__main__':
    example()
