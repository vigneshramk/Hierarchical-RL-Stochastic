import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy, BPW_MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot

from agent import VecEnvAgent

args = get_args()

#TODO: 

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
	assert args.algo in ['a2c', 'ppo'], \
		'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)
	
args.log_dir = args.log_dir + args.env_name + '_' + args.algo
try:
	os.makedirs(args.log_dir)
except OSError:
	files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
	for f in files:
		os.remove(f)


def main():
	print("#######")
	print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
	print("#######")

	os.environ['OMP_NUM_THREADS'] = '1'

	if args.vis:
		from visdom import Visdom
		viz = Visdom(port=args.port)
		win = None

	envs = [make_env(args.env_name, args.seed, i, args.log_dir)
				for i in range(args.num_processes)]

	if args.num_processes > 1:
		envs = SubprocVecEnv(envs)
	else:
		envs = DummyVecEnv(envs)

	if len(envs.observation_space.shape) == 1:
		envs = VecNormalize(envs)

	agent = VecEnvAgent(envs, args)	
	agent.train_maml(num_updates)
	
if __name__ == "__main__":
	main()
