#!/usr/bin/env python
import numpy as npy
import gym
import gym_cp
import gym_bipedal
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

from gym import wrappers
import torch
import torch.nn as nn
from A2C import A2C
from common.utils import agg_double_list

# Selecting the gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

		self.random_seed = 2018

		self.env = gym.make(environment_name)
		self.env.seed(self.random_seed)

		self.env_eval = gym.make(environment_name)
		self.env_eval.seed(self.random_seed)

		self.state_dim = self.env.observation_space.shape[0]
		if len(self.env.action_space.shape) > 1:
			self.action_dim = self.env.action_space.shape[0]
		else:
			self.action_dim = self.env.action_space.n

		self.max_episodes = 1
		self.episodes_before_train = 0
		self.eval_episodes = 0
		self.eval_iterval = 0

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
		elif(environment_name == 'cp-v0'):
			self.reward_discounted_gamma = 0.99  # 0.95
		elif(environment_name == 'Bipedal-v0'):
			self.reward_discounted_gamma = 0.99  # 0.95

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

		# str_c = './tmp/'+ environment_name + '/DuQN_-experiment/' + time_now
		# self.env = wrappers.Monitor(self.env, str_c,force=True, video_callable=lambda episode_id: episode_id%30==0)

		self.environment_name = environment_name
		self.save_weight_interval = 200

		# Cartpole parameters
		if(environment_name == 'cp-v0'):
			self.G = 9.8
			self.MC = 1
			self.MP = 0.1
			self.L = 0.5
			self.F = 10
		elif(environment_name == 'Bipedal-v0'):
			self.F = 2.5 

		self.range = [0.5, 2]
		
	def train_maml(self, render=1):
		
		sample_size = 10
		theta_list = []
		K = 1
		num_iterations = 50000
		task_list = []

		plt.figure()

		for i in range(num_iterations*sample_size):
			if(self.environment_name == 'cp-v0'):
				task = {'G': np.random.uniform(self.range[0]*self.G, self.range[1]*self.G, 1)[0],
						'MC': np.random.uniform(self.range[0]*self.MC, self.range[1]*self.MC, 1)[0],
						'MP': np.random.uniform(self.range[0]*self.MP, self.range[1]*self.MP, 1)[0],
						'L': np.random.uniform(self.range[0]*self.L, self.range[1]*self.L, 1)[0],
						'F': np.random.uniform(self.range[0]*self.F, self.range[1]*self.F, 1)[0]}
			elif(self.environment_name == 'Bipedal-v0'):
				task = {'F': np.random.uniform(self.range[0]*self.F, self.range[1]*self.F, 1)[0]}

			task_list.append(task)

		num_tasks = len(task_list)

		# Outer loop
		for i in range(num_iterations):
			sample_indexes = np.random.randint(0, num_tasks, size=sample_size)

			# Get the theta
			if i == 0:
				theta_actor_critic = self.a2c.get_weights()

			# Inner loop
			# First gradient
			for j, sample_index in enumerate(sample_indexes):
				task = task_list[sample_index]
				# Set the configuration
				if(self.environment_name == 'cp-v0'):
					self.env.env.my_init(task['G'],
										 task['MC'],
										 task['MP'],
										 task['L'],
										 task['F'])
				elif(self.environment_name == 'Bipedal-v0'):
					self.env.env.my_init(task['F'])

				# Set the model weights to theta before training
				self.a2c.set_weights(theta_actor_critic)

				# Train the a2c network for this task for K episodes
				while self.a2c.n_episodes < K:
					self.a2c.interact()
					self.a2c.train()

				if i == 0:
					theta_list.append(self.a2c.get_weights())
				else:
					theta_list[j] = self.a2c.get_weights()

			# Second gradiet
			for j, sample_index in enumerate(sample_indexes):
				task = task_list[sample_index]
				# Set the configuration
				if(self.environment_name == 'cp-v0'):
					self.env.env.my_init(task['G'],
										 task['MC'],
										 task['MP'],
										 task['L'],
										 task['F'])
				elif(self.environment_name == 'Bipedal-v0'):
					self.env.env.my_init(task['F'])

				# Set the model weights to theta before training
				self.a2c.set_weights(theta_list[j])

				# Get the network loss for this task for 1 episode
				# TODO: There should be no while loop
				# while self.a2c.n_episodes < 1:
				self.a2c.interact()
				combined_loss = self.a2c.get_loss()

				# Set the model weights to theta
				self.a2c.set_weights(theta_actor_critic)

				# Update theta
				self.a2c.update_net(combined_loss)
				theta_actor_critic = self.a2c.get_weights()

			# Evaluate the network
			self.a2c.interact()
			rewards, _ = self.a2c.evaluation(self.env_eval, 1)
			rewards_mu, rewards_std = agg_double_list(rewards)
			print("Episode %d, Average Reward %.2f" %
				  (i+1, rewards_mu))

			# Plot iteration vs reward
			plt.scatter(i, rewards_mu)
			#plt.pause(0.0001)

			# Save the weights
			if i%self.save_weight_interval == 0 and i != 0:
				self.a2c.save_weights(self.environment_name, i)
			base_path = os.path.join(self.environment_name, 'a2c_plot_train')
			if not os.path.exists(base_path):
					os.makedirs(base_path)
			file_name = os.path.join(base_path, 'Average_reward_train.png')
			plt.title("%s" % self.environment_name)
			plt.xlabel("Episode")
			plt.ylabel("Average Reward")
			plt.legend(["A2C"])
			plt.savefig(file_name)



	def test_final(self, actor_weight_file, critic_weight_file):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		
		# Load the weights
		self.a2c.load_weights(actor_weight_file, critic_weight_file)

		# Setting the env configuration
		if(environment_name == 'cp-v0'):
			self.env.env.my_init(self.G * 10.5,
								 self.MC * 0.9,
								 self.MP * 2.5,
								 self.L * 1.5,
								 self.F * 2.5)
		elif(environment_name == 'Bipedal-v0'):
			self.env.env.my_init(self.F * 2.5)

		# mini train
		num_minitrain_episodes = 10
		while self.a2c.n_episodes < 10:
			self.a2c.interact()
			self.a2c.train()

		episodes = []
		eval_rewards = []
		num_episodes = 40 + num_minitrain_episodes

		while self.a2c.n_episodes < num_episodes:
			self.a2c.interact()
			rewards, _ = self.a2c.evaluation(self.env_eval, 1)
			rewards_mu, rewards_std = agg_double_list(rewards)
			print("Episode %d, Average Reward %.2f" %
				  (self.a2c.n_episodes+1, rewards_mu))
			episodes.append(self.a2c.n_episodes+1)
			eval_rewards.append(rewards_mu)

		episodes = np.array(episodes)
		eval_rewards = np.array(eval_rewards)

		# Print mean and std.dev
		mean_reward = np.mean(eval_rewards)
		stddev_reward = np.std(eval_rewards)
		print("Mean Reward:{}\n Std. dev:{}".format(mean_reward, stddev_reward))

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
		plt.show()
		plt.savefig(file_name)

def maml_main(args):
	environment_name = args.env
  
	# You want to create an instance of the DuQN_Agent class here, and then train / test it.
	agenta2c = a2c_Agent(environment_name)

	if args.train == 1:
			agenta2c.train_maml(args.render)
			sys.exit(0)

	else:
		if args.actor_model_file and args.critic_model_file is not None:
			agenta2c.test_final(args.actor_model_file, args.critic_model_file)

