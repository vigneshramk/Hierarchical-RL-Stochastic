import copy
import glob
import os
import time
import random
import gym
import gym_hc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot
from algo import update,meta_update
from torch.distributions import Categorical
from adam_new import Adam_Custom

from copy import deepcopy

class VecEnvAgent(object):
	def __init__(self, envs, args):
		self.envs = envs
		self.args = args

		obs_shape = self.envs.observation_space.shape
		self.obs_shape = (obs_shape[0] * self.args.num_stack, *obs_shape[1:])
		
		self.actor_critic = self.select_network()
		self.optimizer = self.select_optimizer()    
		if self.args.cuda:  self.actor_critic.cuda()

		self.action_shape = 1 if self.envs.action_space.__class__.__name__ == "Discrete" \
							else self.envs.action_space.shape[0]        
		
		self.current_obs = torch.zeros(self.args.num_processes, *self.obs_shape)
		obs = self.envs.reset()
		self.update_current_obs(obs)
		
		self.rollouts = RolloutStorage(self.args.num_steps, self.args.num_processes, 
			self.obs_shape, self.envs.action_space, self.actor_critic.state_size)
		self.rollouts.observations[0].copy_(self.current_obs)

		# These variables are used to compute average rewards for all processes.
		self.episode_rewards = torch.zeros([self.args.num_processes, 1])
		self.final_rewards = torch.zeros([self.args.num_processes, 1])

		if self.args.cuda:
			self.current_obs = self.current_obs.cuda()
			self.rollouts.cuda()

		if self.args.vis:
			from visdom import Visdom
			self.viz = Visdom(port=args.port)
			self.win = None 




	def select_network(self):
		if len(self.envs.observation_space.shape) == 3:
			actor_critic = CNNPolicy(self.obs_shape[0], self.envs.action_space, 
				self.args.recurrent_policy)
		else:
			assert not self.args.recurrent_policy, \
				"Recurrent policy is not implemented for the MLP controller"
			actor_critic = MLPPolicy(self.obs_shape[0], self.envs.action_space)
			#actor_critic = BPW_MLPPolicy(obs_shape[0], self.envs.action_space)     
		return actor_critic


	def select_optimizer(self):
		if self.args.algo == 'a2c' and not self.args.use_adam:
			optimizer = optim.RMSprop(self.actor_critic.parameters(), self.args.lr, 
				eps=self.args.eps, alpha=self.args.alpha)
		elif self.args.algo == 'ppo' or self.args.algo == 'a2c':
			optimizer = optim.Adam(self.actor_critic.parameters(), self.args.lr, 
				 eps=self.args.eps)
			self.meta_optimizer = Adam_Custom(self.actor_critic.parameters(), lr=self.args.lr,eps=self.args.eps)
		elif self.args.algo == 'acktr':
			optimizer = KFACOptimizer(self.actor_critic)    
		else:
			raise TypeError("Optimizer should be any one from {a2c, ppo, acktr}")   
		return optimizer    


	def update_current_obs(self, obs):
		shape_dim0 = self.envs.observation_space.shape[0]
		obs = torch.from_numpy(obs).float()
		if self.args.num_stack > 1:
			self.current_obs[:, :-shape_dim0] = self.current_obs[:, shape_dim0:]
		self.current_obs[:, -shape_dim0:] = obs


	def run(self):
		for step in range(self.args.num_steps):
			value, action, action_log_prob, states = self.actor_critic.act(
				Variable(self.rollouts.observations[step], volatile=True),
				Variable(self.rollouts.states[step], volatile=True),
				Variable(self.rollouts.masks[step], volatile=True)
				)
			cpu_actions = action.data.squeeze(1).cpu().numpy()
			#print (cpu_actions)
			#input()

			# Obser reward and next obs
			obs, reward, done, info = self.envs.step(cpu_actions)
			reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
			self.episode_rewards += reward

			# If done then clean the history of observations.
			masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
			self.final_rewards *= masks
			self.final_rewards += (1 - masks) * self.episode_rewards
			self.episode_rewards *= masks

			if self.args.cuda: masks = masks.cuda()

			if self.current_obs.dim() == 4:
				self.current_obs *= masks.unsqueeze(2).unsqueeze(2)
			else:
				self.current_obs *= masks

			self.update_current_obs(obs)
			self.rollouts.insert(step, self.current_obs, states.data, action.data, 
				action_log_prob.data, value.data, reward, masks)
	
		next_value = self.actor_critic(
						Variable(self.rollouts.observations[-1], volatile=True),
						Variable(self.rollouts.states[-1], volatile=True),
						Variable(self.rollouts.masks[-1], volatile=True)
						)[0].data

		self.rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma, self.args.tau)
		dist_entropy, value_loss, action_loss = update(self)
		self.rollouts.after_update()
		
		return dist_entropy, value_loss, action_loss

	def meta_run(self,theta_loss,theta_grad):
		for step in range(self.args.num_steps):
			value, action, action_log_prob, states = self.actor_critic.act(
				Variable(self.rollouts.observations[step], volatile=True),
				Variable(self.rollouts.states[step], volatile=True),
				Variable(self.rollouts.masks[step], volatile=True)
				)
			cpu_actions = action.data.squeeze(1).cpu().numpy()
			#print (cpu_actions)
			#input()

			# Obser reward and next obs
			obs, reward, done, info = self.envs.step(cpu_actions)
			reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
			self.episode_rewards += reward

			# If done then clean the history of observations.
			masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
			self.final_rewards *= masks
			self.final_rewards += (1 - masks) * self.episode_rewards
			self.episode_rewards *= masks

			if self.args.cuda: masks = masks.cuda()

			if self.current_obs.dim() == 4:
				self.current_obs *= masks.unsqueeze(2).unsqueeze(2)
			else:
				self.current_obs *= masks

			self.update_current_obs(obs)
			self.rollouts.insert(step, self.current_obs, states.data, action.data, 
				action_log_prob.data, value.data, reward, masks)
	
		next_value = self.actor_critic(
						Variable(self.rollouts.observations[-1], volatile=True),
						Variable(self.rollouts.states[-1], volatile=True),
						Variable(self.rollouts.masks[-1], volatile=True)
						)[0].data

		self.rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma, self.args.tau)
		dist_entropy, value_loss, action_loss = meta_update(self,theta_loss,theta_grad)
		self.rollouts.after_update()
		
		return dist_entropy, value_loss, action_loss

	# def update_net(self,dist_entropy,value_loss,action_loss):

	# 	# self.optimizer.zero_grad()
	# 	# (value_loss + action_loss - dist_entropy * 0.01).backward()
	# 	# nn.utils.clip_grad_norm(self.actor_critic.parameters(), 0.2)
	# 	# self.optimizer.step()

	# 	update_network(self,dist_entropy,value_loss,action_loss)


	def evaluate(self,j,dist_entropy,value_loss,action_loss,model_file=None):
		end = time.time()
		total_num_steps = (j + 1) * self.args.num_processes * self.args.num_steps
		print("Updates {}, num timesteps {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
			format(j, total_num_steps,
				   self.final_rewards.mean(),
				   self.final_rewards.median(),
				   self.final_rewards.min(),
				   self.final_rewards.max(), dist_entropy.data[0],
				   value_loss.data[0], action_loss.data[0]))

		try:
			# Sometimes monitor doesn't properly flush the outputs
			self.win = visdom_plot(self.viz, self.win, self.args.log_dir, 
				self.args.env_name, self.args.algo)
		except IOError:
			pass
		

	def train(self, num_updates):
		start = time.time()
		for j in range(num_updates):
			dist_entropy, value_loss, action_loss = self.run()

			if j % self.args.save_interval == 0 and self.args.save_dir != "":
				save_path = os.path.join(self.args.save_dir, self.args.algo)
				try:
					os.makedirs(save_path)
				except OSError:
					pass

				# A really ugly way to save a model to CPU
				save_model = self.actor_critic
				if self.args.cuda:
					save_model = copy.deepcopy(self.actor_critic).cpu()

				save_model = [save_model,
								hasattr(self.envs, 'ob_rms') and self.envs.ob_rms or None]

				torch.save(save_model, os.path.join(save_path, self.args.env_name + ".pt"))

			if j % self.args.log_interval == 0:
				end = time.time()
				total_num_steps = (j + 1) * self.args.num_processes * self.args.num_steps
				print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
					format(j, total_num_steps,
						   int(total_num_steps / (end - start)),
						   self.final_rewards.mean(),
						   self.final_rewards.median(),
						   self.final_rewards.min(),
						   self.final_rewards.max(), dist_entropy.data[0],
						   value_loss.data[0], action_loss.data[0]))
			if self.args.vis and j % self.args.vis_interval == 0:
				try:
					# Sometimes monitor doesn't properly flush the outputs
					self.win = visdom_plot(self.viz, self.win, self.args.log_dir, 
						self.args.env_name, self.args.algo)
				except IOError:
					pass


	def train_maml(self, num_updates):
		start = time.time()
		theta_list = []

		num_tasks = 1000
		sample_size = 10
		 
		# episode_id: episode_id%10==0)

		# env = gym.wrappers.Monitor(self.envs, self.args.save_dir, video_callable=lambda episode_id: episode_id%10==0)

		# Create the variations needed
		task_list = []
		for i in range(num_tasks):
			friction = np.random.randint(low=1, high=10, size=3).astype('float32')/10.
			friction_1 = np.random.uniform(low=0.1, high=0.8, size=3).astype('float32')
			task = {'default/geom': ['', 'friction', '{0:.1f} {1:.1f} {2:.1f}'.format(
				friction[0],
				friction[1],
				friction[2])],
				'worldbody/body/body/geom': [[['name', 'fthigh'], ['type', 'capsule']], 
											 'friction',
											 '{0:.1f} {1:.1f} {2:.1f}'.format(
											  friction_1[0],
											  friction_1[1],
											  friction_1[2])]
			}
			# task2 = {'option': ['gravity', '{0:.2f} {1:.2f} {2:.2f}'.format(0,0,gravity_z)]}
			task_list.append(task)


		for j in range(num_updates):

			sample_indexes = np.random.randint(0, num_tasks, size=sample_size)
			# Get the theta
			if j == 0:
				theta = self.get_weights()

			# Inner loop
			# First gradient
			for i, sample_index in enumerate(sample_indexes):

				# Get the task
				task = task_list[sample_index]
				env = self.envs.venv.envs[0]

				# env = gym.wrappers.Monitor(env.env, './videos2/', video_callable=lambda episode_id: episode_id%10==0)

				_tag_names = []
				_tag_identifiers = []
				_attributes = []
				_values = []

				for k in task.keys():
					v = task[k]
					_tag_names.append(k)
					_tag_identifiers.append(v[0])
					_attributes.append(v[1])
					_values.append(v[2])

				env.env.env.my_init(_tag_names, \
									_tag_identifiers,
                                    _attributes, \
                                    _values,
									None)

				# Set the model weights to theta before training
				self.set_weights(theta)

				dist_entropy, value_loss, action_loss = self.run()

				if j == 0:
					theta_list.append(self.get_weights())
				else:
					print(i)
					theta_list[i] = self.get_weights()

			# Second gradiet
			theta_copy = deepcopy(theta)
			for k1, sample_index in enumerate(sample_indexes):

				# Get the task
				task = task_list[sample_index]
				env = self.envs.venv.envs[0]

				_tag_names = []
				_tag_identifiers = []
				_attributes = []
				_values = []

				for k in task.keys():
					v = task[k]
					_tag_names.append(k)
					_tag_identifiers.append(v[0])
					_attributes.append(v[1])
					_values.append(v[2])

				env.env.env.my_init(_tag_names, \
									_tag_identifiers,
                                    _attributes, \
                                    _values,
									None)


				# Get the network loss for this task for 1 episode
				# TODO: There should be no while loop
				# while self.a2c.n_episodes < 1:
				dist_entropy, value_loss, action_loss = self.meta_run(theta_list[k1],theta_copy)

				theta = self.get_weights()

				# Set the model weights to theta
				# self.set_weights(theta)

				# Update theta
				# Change the update network function
				# theta['state_dict'] = self.agent.update_net(theta['state_dict'],dist_entropy,value_loss,action_loss)

			# env = gym.wrappers.Monitor(env, './videos/', video_callable=lambda episode_id: episode_id%10==0,force=True)	


			if j % self.args.save_interval == 0 and self.args.save_dir != "":
				save_path = os.path.join(self.args.save_dir, self.args.algo)
				try:
					os.makedirs(save_path)
				except OSError:
					pass

				model_state = {'num_updates': j,
						    'state_dict': self.actor_critic.state_dict(),
						    'optimizer': self.meta_optimizer.state_dict()
							}
				model_state = [model_state,hasattr(self.envs, 'ob_rms') and self.envs.ob_rms or None]

				torch.save(model_state, os.path.join(save_path, self.args.env_name + 'update_'+ str(j) +".pt"))

				# # A really ugly way to save a model to CPU
				# save_model = self.actor_critic
				# if self.args.cuda:
				# 	save_model = copy.deepcopy(self.actor_critic).cpu()

				# save_model = [save_model,
				# 				hasattr(self.envs, 'ob_rms') and self.envs.ob_rms or None]

				# torch.save(save_model, os.path.join(save_path, self.args.env_name + ".pt"))

			if j % self.args.log_interval == 0:
				end = time.time()
				total_num_steps = (j + 1) * self.args.num_processes * self.args.num_steps
				print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
					format(j, total_num_steps,
						   int(total_num_steps / (end - start)),
						   self.final_rewards.mean(),
						   self.final_rewards.median(),
						   self.final_rewards.min(),
						   self.final_rewards.max(), dist_entropy.data[0],
						   value_loss.data[0], action_loss.data[0]))
			if self.args.vis and j % self.args.vis_interval == 0:
				try:
					# Sometimes monitor doesn't properly flush the outputs
					self.win = visdom_plot(self.viz, self.win, self.args.log_dir, 
						self.args.env_name, self.args.algo)
				except IOError:
					pass

	def get_weights(self):
		# state_dicts = {'id': id,
		# 			   'state_dict': self.actor_critic.state_dict(),
		# 			   }

		return self.actor_critic.state_dict()

	def set_weights(self, state_dicts):
		
		checkpoint = state_dicts

		self.actor_critic.load_state_dict(checkpoint)
		# self.optimizer.load_state_dict(checkpoint['optimizer'])
