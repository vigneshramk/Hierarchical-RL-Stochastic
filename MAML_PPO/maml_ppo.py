#!/usr/bin/env python
import numpy as npy
import gym
import glob
# import gym_cp
# import gym_bipedal
import sys
from copy import copy, deepcopy
import argparse
# import matplotlib.pyplot as plt
import random
import numpy as np
from math import fmod
import time
import os
import datetime

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

# Selecting the gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class ppo_agent():

	# In this class, we will implement functions to do the following.
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
	#       (a) Epsilon Greedy Policy.
	#       (b) Greedy Policy.
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self):

		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.

		args.log_dir = args.log_dir + args.env_name + '_' + args.algo
		try:
			os.makedirs(args.log_dir)
		except OSError:
			files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
			for f in files:
				os.remove(f)

		envs = [make_env(args.env_name, args.seed, i, args.log_dir)
				for i in range(args.num_processes)]

		if args.num_processes > 1:
			envs = SubprocVecEnv(envs)
		else:
			envs = DummyVecEnv(envs)

		if len(envs.observation_space.shape) == 1:
			envs = VecNormalize(envs)


		self.environment_name = args.env_name
		self.agent = VecEnvAgent(envs, args)	
		

	def train_maml(self, render=1):
		
		sample_size = 10
		theta_list = []
		K = 1
		num_iterations = 50000
		task_list = []

		# fig2 = plt.figure()
		# ax2 = fig2.gca()
		# ax2.set_title('Test Reward Plot')

		path_name = './fig_maml_a2c_test'
		if not os.path.exists(path_name):
			os.makedirs(path_name)

		self.range = [1,1]

		# Cartpole parameters
		if(self.environment_name == 'CartPole-v0'):
			self.G = 9.8
			self.MC = 1
			self.MP = 0.1
			self.L = 0.5
			self.F = 10


		# TODO:
		for i in range(num_iterations*sample_size):
			if(self.environment_name == 'CartPole-v0'):
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
				theta = self.agent.get_weights()

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
				self.agent.set_weights(theta)

				# Train the a2c network for this task for K episodes
				# while self.a2c.n_episodes < K:
				# 	self.a2c.interact()
				
				dist_entropy, value_loss, action_loss = self.agent.run()

				if i == 0:
					theta_list.append(self.agent.get_weights())
				else:
					theta_list[j] = self.agent.get_weights()

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
				self.agent.set_weights(theta_list[j])

				# Get the network loss for this task for 1 episode
				# TODO: There should be no while loop
				# while self.a2c.n_episodes < 1:
				dist_entropy, value_loss, action_loss = self.agent.run()

				# Set the model weights to theta
				self.agent.set_weights(theta)

				# Update theta
				# Change the update network function
				# theta['state_dict'] = self.agent.update_net(theta['state_dict'],dist_entropy,value_loss,action_loss)

				self.agent.update_net(theta['state_dict'],dist_entropy,value_loss,action_loss)
	
			# # Evaluate the network

			if i%100 == 0:
				self.agent.evaluate(i,dist_entropy,value_loss,action_loss)

			# rewards, _ = self.a2c.evaluation(self.env_eval, 1)
			# rewards_mu, rewards_std = agg_double_list(rewards)
			

			# # Plot iteration vs reward
			# plt.scatter(i, rewards_mu)
			# #plt.pause(0.0001)

			# # Save the weights
			# if i%self.save_weight_interval == 0 and i != 0:
			# 	self.a2c.save_weights(self.environment_name, i)
			# base_path = os.path.join(self.environment_name, 'a2c_plot_train')
			# if not os.path.exists(base_path):
			# 		os.makedirs(base_path)
			# file_name = os.path.join(base_path, 'Average_reward_train.png')
			# plt.title("%s" % self.environment_name)
			# plt.xlabel("Episode")
			# plt.ylabel("Average Reward")
			# plt.legend(["A2C"])
			# plt.savefig(file_name)


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

def main(args):
  
	# You want to create an instance of the DuQN_Agent class here, and then train / test it.
	agenta2c = ppo_agent()

	
	agenta2c.train_maml(render=0)

	# else:
	# 	if args.actor_model_file and args.critic_model_file is not None:
	# 		agenta2c.test_final(args.actor_model_file, args.critic_model_file)


if __name__ == '__main__':
	main(sys.argv)

