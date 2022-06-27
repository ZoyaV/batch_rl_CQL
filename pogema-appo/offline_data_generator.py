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

agent_expirience = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': []}
def example():

    gc = GridConfig(seed=None, num_agents=1, max_episode_steps=64, obs_radius=5, size=16, density=0.3)

    # turn off with_animation to speedup generation speed
    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None,
                   observation_type='MAPF')

    # fully centralized planning approach
    algo = Prioritized(PrioritizedConfig())
    # partially observable decentralized planning approach
    # algo = RePlan(RePlanConfig())

    for _ in range(100000):
        observations = env.reset()
        algo.after_reset()
        dones = [False, ...]

        obs = MatrixObservationWrapper.to_matrix(observations)
        obs = obs[0]['obs']
      #  print(obs.shape)
        obs = np.asarray(obs)
        obs = obs.reshape(-1, )
      #  print(obs.shape)
      #  break
        while not all(dones):
            agent_expirience['observations'].append(obs)

            action = algo.act(observations, None, dones)
            observations, rewards, dones, info = env.step(action)
           # obs = drop_global_information(observations.copy())
            obs = MatrixObservationWrapper.to_matrix(observations)
            obs = obs[0]['obs']
            obs = np.asarray(obs)
            obs = obs.reshape(-1, )

            agent_expirience['next_observations'].append(obs)
            agent_expirience['actions'].append((action[0])/(4))
            agent_expirience['rewards'].append(rewards[0])
            agent_expirience['terminals'].append(dones[0])

            algo.after_step(dones)

    print(agent_expirience['actions'])
    print(agent_expirience['terminals'][-1])
    import pickle


    for key in agent_expirience:
        agent_expirience[key] = np.asarray(agent_expirience[key])
  #  agent_expirience['next_observations'] = agent_expirience['next_observations'].reshape(-1,1)
   # agent_expirience['observations'] = agent_expirience['observations'].reshape(-1, 1)
    agent_expirience['actions'] = agent_expirience['actions'].reshape(-1, 1)
    with open('../examples/data.pickle', 'wb') as f:
        pickle.dump(agent_expirience, f)

    with h5py.File('../examples/data.hdf5', 'w') as f:
        f.create_dataset('next_observations', data=agent_expirience['next_observations'])
        f.create_dataset('actions', data= agent_expirience['actions'])
        f.create_dataset('rewards', data=agent_expirience['rewards'])
        f.create_dataset('terminals', data=agent_expirience['terminals'])
        f.create_dataset('observations', data=agent_expirience['observations'])


if __name__ == '__main__':
    example()