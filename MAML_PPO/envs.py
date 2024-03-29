import os

import gym
import gym_hc
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import datetime

#import pybullet_envs
# import roboschool

'''
try:
    import pybullet_envs
    import roboschool
except ImportError:
    pass
'''

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)
        # env = gym.wrappers.Monitor(env, log_dir, video_callable=lambda episode_id: episode_id%10==0)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if is_atari:
            env = wrap_deepmind(env)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)
        directory = './videos/'+'hc-v3'+ str(datetime.datetime.now())
        # env = gym.wrappers.Monitor(env.env,directory, video_callable=lambda episode_id: episode_id%100==0)
        return env
    return _thunk


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)
