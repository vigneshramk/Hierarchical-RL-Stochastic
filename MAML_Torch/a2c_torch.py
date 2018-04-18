#!/usr/bin/env python
import numpy as npy
import gym
import sys
from copy import copy, deepcopy
import argparse
import matplotlib.pyplot as plt
import random
import numpy as np
from math import fmod
import time
import os
import datetime
import csv
from gym import wrappers
import torch
import torch.nn as nn
from A2C import A2C
from common.utils import agg_double_list

import gym_cp

# Selecting the gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class CartPoleNet(nn.Module):
	def __init__(self): #, state_size=4, action_size=2, hidden_size=24):
		super(CartPoleNet, self).__init__()
		hidden_size = 24
		state_size = 4
		action_size = 2
		
		self.branch1 = nn.Sequential(
				nn.Linear(state_size, hidden_size),
				nn.ReLU(),

				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),

				nn.Linear(hidden_size, 1)
		)

		self.branch2 = nn.Sequential(
				nn.Linear(state_size, hidden_size),
				nn.ReLU(),

				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),

				nn.Linear(hidden_size, action_size)
		)

	def forward(self, x):
		x1 = self.branch1(x)
		x2 = self.branch2(x)

		x2_mean = x2.mean()
		x2 = x2 - x2_mean

		x = x1 + x2

		return x


class MountainCarNet(nn.Module):
	def __init__(self):  # , state_size=4, action_size=2, hidden_size=24):
		super(CartPoleNet, self).__init__()
		hidden_size = 32
		state_size = 2
		action_size = 3

		self.branch1 = nn.Sequential(
			nn.Linear(state_size, hidden_size),
			nn.ReLU(),

			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),

			nn.Linear(hidden_size, 1)
		)

		self.branch2 = nn.Sequential(
			nn.Linear(state_size, hidden_size),
			nn.ReLU(),

			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),

			nn.Linear(hidden_size, action_size)
		)

	def forward(self, x):
		x1 = self.branch1(x)
		x2 = self.branch2(x)

		x2_mean = x2.mean()
		x2 = x2 - x2_mean

		x = x1 + x2

		return x


class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, environment_name):
		# Define your network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		self.environment_name = environment_name

		# Model for Cartpole-v0
		if(environment_name == 'CartPole-v0'):
			self.learning_rate = 0.001
			self.momentum = 0.9

			self.model = CartPoleNet()
			self.model.cuda()

		# Model for MountainCar-v0
		elif(environment_name == 'MountainCar-v0'):
			self.learning_rate = 0.001
			self.momentum = 0.9

			self.model = MountainCarNet()
			self.model.cuda()

		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), \
										 lr=self.learning_rate, \
										 momentum=self.momentum)

	def save_model_weights(self, count):
		# Helper function to save your model / weights.
		state = {'count': count, \
				 'state_dict': self.model.state_dict(),
				 'optimizer': self.optimizer.state_dict()}

		if not os.path.exists(os.path.join(self.environment_name, 'DuQN_weights')):
			os.makedirs(os.path.join(self.environment_name, 'DuQN_weights'))
		time_now = self.environment_name + str(count) + '.pth.tar'
		time_now.replace(" ","")

		file_name = os.path.join(self.environment_name, 'DuQN_weights', time_now)
		torch.save(state, file_name)

		time_now = self.environment_name + str(datetime.datetime.now()) + '.pth.tar'
		time_now.replace(" ","")
		file_name = os.path.join(self.environment_name, 'DuQN_weights', time_now)
		torch.save(state, file_name)
		
	def load_model(self, model_file):
		# Helper function to load an existing model.
		pass

	def load_model_weights(self, weight_file):
		# Helper funciton to load model weights.
		print("=> loading checkpoint '{}'".format(weight_file))
		checkpoint = torch.load(weight_file)
		self.model.load_state_dict(checkpoint['state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}'".format(weight_file))


class a2c_Agent():

	# In this class, we will implement functions to do the following.
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
	#       (a) Epsilon Greedy Policy.
	#       (b) Greedy Policy.
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.


		random_seed = 2017

		self.env = gym.make('cp-v0')
		self.env.seed(random_seed)

		self.env_eval = gym.make('cp-v0')
		self.env_eval.seed(random_seed)

		self.num_episodes = 2000

		self.state_dim = self.env.observation_space.shape[0]
		if len(self.env.action_space.shape) > 1:
			self.action_dim = self.env.action_space.shape[0]
		else:
			self.action_dim = self.env.action_space.n

		self.max_episodes = 5000
		self.episodes_before_train = 0
		self.eval_episodes = 10
		self.eval_iterval = 100

		# roll out n steps
		self.roll_out_n_steps = 10
		# only remember the latest ROLL_OUT_N_STEPS
		self.memory_capacity = self.roll_out_n_steps
		# only use the latest ROLL_OUT_N_STEPS for training A2C
		self.batch_size = self.roll_out_n_steps


		if(environment_name == 'CartPole-v0'):
			self.reward_discounted_gamma = 0.99  # 0.95
		elif(environment_name == 'MountainCar-v0'):
			self.reward_discounted_gamma = 1

		self.entropy_reg = 0.00
		self.done_penalty = -10.

		self.critic_loss = "mse"
		self.max_grad_norm = None

		self.epsilon_start = 0.99
		self.epsilon_end = 0.05
		self.epsilon_decay = 500

		self.a2c = A2C(env=self.env, memory_capacity=self.memory_capacity,
					   state_dim=self.state_dim, action_dim=self.action_dim,
					   batch_size=self.batch_size, entropy_reg=self.entropy_reg,
					   done_penalty=self.done_penalty, roll_out_n_steps=self.roll_out_n_steps,
					   reward_gamma=self.reward_discounted_gamma,
					   epsilon_start=self.epsilon_start, epsilon_end=self.epsilon_end,
					   epsilon_decay=self.epsilon_decay, max_grad_norm=self.max_grad_norm,
					   episodes_before_train=self.episodes_before_train,
					   critic_loss=self.critic_loss)

		# Video saving
		time_now = str(datetime.datetime.now())
		time_now.replace(" ","")

		str_c = './tmp/'+ environment_name + '/DuQN_-experiment/' + time_now
		self.env = wrappers.Monitor(self.env, str_c,force=True, video_callable=lambda episode_id: episode_id%30==0)

		self.environment_name = environment_name
		self.input_shape = []
		self.obs_list = []
		self.iteration_count_list = []
		self.mean_reward_list = []

	def train(self,render=1):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.

		# If you are using a replay memory, you should interact with environment here, and store these
		# transitions to memory, while also updating your model.

		# Variables init

		# Burn in memory

		episodes = []
		eval_rewards = []

		while self.a2c.n_episodes < self.max_episodes:
			self.a2c.interact()
			if self.a2c.n_episodes >= self.episodes_before_train:
				self.a2c.train()
			if self.a2c.episode_done and ((self.a2c.n_episodes+1) % self.eval_iterval == 0):
				rewards, _ = self.a2c.evaluation(self.env_eval, self.eval_episodes)
				rewards_mu, rewards_std = agg_double_list(rewards)
				print("Episode %d, Average Reward %.2f" %
					  (self.a2c.n_episodes+1, rewards_mu))
				episodes.append(self.a2c.n_episodes+1)
				eval_rewards.append(rewards_mu)

				# Save the weights
				print("=> Saving weights after {} episodes".format(
					self.a2c.n_episodes+1))
				self.a2c.save_weights(self.environment_name, self.a2c.n_episodes+1)

		episodes = np.array(episodes)
		eval_rewards = np.array(eval_rewards)

		# Save the plot
		base_path = os.path.join(self.environment_name, 'a2c_plot_eval')
		if not os.path.exists(base_path):
			os.makedirs(base_path)
		file_name = os.path.join(base_path, 'Average_reward.png')

		plt.figure()
		plt.plot(episodes, eval_rewards)
		plt.title("%s"%self.environment_name)
		plt.xlabel("Episode")
		plt.ylabel("Average Reward")
		plt.legend(["A2C"])
		plt.savefig(file_name)
		
	def train_maml(self, render=1):
		
		task_list = []
		num_tasks = len(task_list)
		sample_size = 10
		
		for i in range(10000):
			sample_indexes = np.random.randint(num_tasks, size=sample_size)

			for sample_index in sample_indexes:
				task = task_list[sample_index]
				# make the env
				env = gym.make(task)
				env.seed(self.random_seed)

				# delete the env
		
	def test_final(self, actor_weight_file, critic_weight_file):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		
		episodes = []
		eval_rewards = []
	
		self.env_fin = gym.make('cp-v0')
		num_episodes = 50
		self.a2c.load_weights(actor_weight_file, critic_weight_file)
	

		base = np.array([1,0.5,2])
		G = np.array([1])*9.8
		MC = base*0.5
		MP = base*0.1
		L = base*0.5
		F = base*10

		fl = open('Experiments.csv','w')
		fl.write('List of parameters: Gravity, Mass of Cart, Mass of Pole, Length, Force Magnitude\n')
		fl.write('Output Reward: Mean, Standard Deviation\n')
	
		for g in G:
			for mc in MC:
				for mp in MP:
					for l in L:
						for f in F:
							self.env_fin.env.my_init(G=g,MC=mc,MP=mp,L=l,F=f)
							for i in range(num_episodes):
								self.a2c.interact()
								rewards, _ = self.a2c.evaluation(self.env_fin, 1)
								rewards_mu, rewards_std = agg_double_list(rewards)
								#print("Episode %d, Average Reward %.2f" %
								#      (self.a2c.n_episodes+1, rewards_mu))
								episodes.append(i+1)
								eval_rewards.append(rewards_mu)
							print(g,mc,mp,l,f)
							rm = float("{0:.2f}".format(np.mean(eval_rewards)))
							rs = float("{0:.2f}".format(np.std(eval_rewards)))
							str_cp = str(mc) + '& ' + str(mp) + '& ' + str(l) + '& ' + str(f) + '& '
							str_cp = str_cp + str(rm) + ' &' + str(rs) + '\n'
							fl.write(str_cp)
							print("Rewards: Mean: %d, Std: %d" %(np.mean(eval_rewards),np.std(eval_rewards)))

		episodes = np.array(episodes)
		eval_rewards = np.array(eval_rewards)
	
		# Save the plot
		base_path = os.path.join(self.environment_name, 'a2c_plot_test')
		if not os.path.exists(base_path):
			os.makedirs(base_path)
		file_name = os.path.join(base_path, 'Average_reward.png')

		plt.figure()
		plt.plot(episodes, eval_rewards)
		plt.title("%s" % self.environment_name)
		plt.xlabel("Episode")
		plt.ylabel("Average Reward")
		plt.legend(["A2C"])
		plt.savefig(file_name)



def a2c_main(args):
	environment_name = args.env
  
	# You want to create an instance of the DuQN_Agent class here, and then train / test it.
	agenta2c = a2c_Agent(environment_name)

	if args.train == 1:
			agenta2c.train(args.render)
			sys.exit(0)

	else:
		if args.actor_model_file and args.critic_model_file is not None:
			agenta2c.test_final(args.actor_model_file, args.critic_model_file)

