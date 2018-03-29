
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np
import os

from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var


class A2C(Agent):
    """
    An agent learned with Advantage Actor-Critic
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        super(A2C, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        self.roll_out_n_steps = roll_out_n_steps

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim,
                                    self.critic_hidden_size, 1)
        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)
        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()

    # agent interact with the environment to collect experience
    def interact(self):
        super(A2C, self)._take_n_steps()

    def get_loss(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(
            batch.states, self.use_cuda).view(-1, self.state_dim)
        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)
        actions_var = to_tensor_var(
            one_hot_actions, self.use_cuda).view(-1, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # Get the actor network loss
        self.actor_optimizer.zero_grad()
        action_log_probs = self.actor(states_var)
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        action_log_probs = th.sum(action_log_probs * actions_var, 1)
        values = self.critic(states_var, actions_var)
        advantages = rewards_var - values.detach()
        pg_loss = -th.mean(action_log_probs * advantages)
        actor_loss = pg_loss - entropy_loss * self.entropy_reg

        # Get the critic network loss
        self.critic_optimizer.zero_grad()
        target_values = rewards_var
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)

        combined_loss = {'actor_loss': actor_loss,
                         'critic_loss': critic_loss}

        return combined_loss

    def update_net(self, combined_loss):
        actor_loss = combined_loss['actor_loss']
        critic_loss = combined_loss['critic_loss']

        # Update the actor network
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(
                self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update the critic network
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(
                self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()


    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)
        actions_var = to_tensor_var(one_hot_actions, self.use_cuda).view(-1, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # update actor network
        self.actor_optimizer.zero_grad()
        action_log_probs = self.actor(states_var)
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        action_log_probs = th.sum(action_log_probs * actions_var, 1)
        values = self.critic(states_var, actions_var)
        advantages = rewards_var - values.detach()
        pg_loss = -th.mean(action_log_probs * advantages)
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = rewards_var
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

    # predict softmax action based on state
    def _softmax_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        softmax_action_var = th.exp(self.actor(state_var))
        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu().numpy()[0]
        else:
            softmax_action = softmax_action_var.data.numpy()[0]
        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(softmax_action)
        return action

    # choose an action based on state for execution
    def action(self, state):
        softmax_action = self._softmax_action(state)
        action = np.argmax(softmax_action)
        return action

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)

        value_var = self.critic(state_var, action_var)
        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]
        return value

    def get_weights(self):
        state_actor = {'id': id,
                       'state_dict': self.actor.state_dict(),
                       'optimizer': self.actor_optimizer.state_dict()}

        state_critic = {'id': id,
                        'state_dict': self.critic.state_dict(),
                        'optimizer': self.critic_optimizer.state_dict()}

        state_dicts = {'state_actor': state_actor,
                       'state_critic': state_critic}
        return state_dicts

    def set_weights(self, state_dicts):
        actor_checkpoint = state_dicts['state_actor']
        critic_checkpoint = state_dicts['state_critic']

        self.actor.load_state_dict(actor_checkpoint['state_dict'])
        self.critic.load_state_dict(critic_checkpoint['state_dict'])

        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer'])

    def save_weights(self, env_name, id):
        state_actor = {'id': id,
                       'state_dict': self.actor.state_dict(),
                       'optimizer': self.actor_optimizer.state_dict()}

        state_critic = {'id': id,
                        'state_dict': self.critic.state_dict(),
                        'optimizer': self.critic_optimizer.state_dict()}

        base_path = os.path.join(env_name, 'a2c_weights')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        file_name_actor = os.path.join(base_path, 'actor_' + str(id) + '.pth.tar')
        file_name_critic = os.path.join(
            base_path, 'critic_' + str(id) + '.pth.tar')

        th.save(state_actor, file_name_actor)
        th.save(state_critic, file_name_critic)

    def load_weights(self, actor_weight_file, critic_weight_file):
        print("=> loading checkpoint '{}, {}'".format(
            actor_weight_file, critic_weight_file))
        actor_checkpoint = th.load(actor_weight_file)
        critic_checkpoint = th.load(critic_weight_file)

        self.actor.load_state_dict(actor_checkpoint['state_dict'])
        self.critic.load_state_dict(critic_checkpoint['state_dict'])

        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer'])

        print("=> loaded checkpoint '{}, {}'".format(
            actor_weight_file, critic_weight_file))

