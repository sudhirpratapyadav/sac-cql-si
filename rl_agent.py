import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import gtimer as gt
import time
import random
import os
import sys
import pathlib
import copy
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation
from collections import OrderedDict
import pprint

import util
from robo_sim import RoboverseSim

from rl_fn_approximators import PolicyNetwork, QNetwork

import gtimer as gt

class RlAgent:

	def __init__(self, param_config, logger, device):

		self.logger = logger
		self.device = device
		self.logger.log("Device: ", self.device)

		# Dictionary of different timings
		self.param_config = param_config
		self.algo_config = param_config['algorithm_kwargs']
		self.train_cql = False

		self.seed = 10
		self.torch_deterministic = True
		self.agent_name = f"{self.seed}__{int(time.time())}"

		self.state_space = self.algo_config['state_space_dim']
		self.action_space = self.algo_config['action_space_dim']

		self.gamma = 0.99

		# Algorithm specific arguments
		self.target_network_frequency = 1
		self.tau = 0.005  # target smoothing coefficient (default: 0.005)
		self.alpha = 0.2  # Entropy regularization coefficient
		self.use_automatic_entropy_tuning = self.algo_config['automatic_entropy_tuning']  # automatic tuning of the entropy coefficient

		# CQL parameters
		self.cql_alpha = self.algo_config['cql_alpha']
		self.cql_temp= self.algo_config['cql_temp']
		self.num_random = 1
		self.min_q_version = 3

		# Additional hyper parameters for tweaks
		# Separating the learning rate of the policy and value commonly seen: (Original implementation, Denis Yarats)
		self.policy_lr = 1e-4  # the learning rate of the policy network optimizer
		self.alpha_lr = 1e-4  # the learning rate of alpha
		self.q_lr = 3e-4  # the learning rate of the Q network network optimizer
		self.policy_frequency = 1  # delays the update of the actor, as per the TD3 paper.')

		self.flag_collect_stats = False

		# random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		torch.backends.cudnn.deterministic = self.torch_deterministic

		self.num_tasks = len(self.param_config['tasks'])

		augmentation_transform = util.RandomCrop(size = self.param_config['policy_params']['input_width'], padding = 4, device=self.device)
		self.param_config['policy_params']['augmentation_transform'] = augmentation_transform
		self.param_config['q_params']['augmentation_transform'] = augmentation_transform

		# print(self.param_config['policy_params'])

		# print(self.param_config['policy_params']['output_heads_num'])

		self.policy = PolicyNetwork(**self.param_config['policy_params']).to(self.device)

		self.q1 = QNetwork(**self.param_config['q_params']).to(self.device)
		self.q2 = QNetwork(**self.param_config['q_params']).to(self.device)
		self.q1_target = QNetwork(**self.param_config['q_params']).to(self.device)
		self.q2_target = QNetwork(**self.param_config['q_params']).to(self.device)

		self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.q_lr)
		self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.q_lr)
		self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

		self.loss_fn = nn.MSELoss()

		# Automatic entropy tuning
		if self.use_automatic_entropy_tuning:
			self.target_entropy = - torch.prod(torch.Tensor((self.action_space,)).to(self.device)).item()
			## self.target_entropy = -8
			self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha = self.log_alpha.exp().item()
			self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
		else:
			self.alpha = 0.2

		self.task_id = 0

		self.regularize_critic = self.algo_config['regularize_critic']

		if self.regularize_critic:

			self.theta_shape = dict(
							policy=self.policy.get_theta(task_id=self.task_id, copy=True).shape,
							q1=self.q1.get_theta(task_id=self.task_id, copy=True).shape,
							q2=self.q2.get_theta(task_id=self.task_id, copy=True).shape,
							)

			self.importance_vars = dict(
				prev_theta 	=dict(
								policy=torch.zeros(self.theta_shape['policy']).to(self.device),
								q1=torch.zeros(self.theta_shape['q1']).to(self.device),
								q2=torch.zeros(self.theta_shape['q2']).to(self.device),
								),
				omega_total =dict(
								policy=torch.zeros(self.theta_shape['policy']).to(self.device),
								q1=torch.zeros(self.theta_shape['q1']).to(self.device),
								q2=torch.zeros(self.theta_shape['q2']).to(self.device),
								),
				omega 		=dict(
								policy=torch.zeros(self.theta_shape['policy']).to(self.device),
								q1=torch.zeros(self.theta_shape['q1']).to(self.device),
								q2=torch.zeros(self.theta_shape['q2']).to(self.device),
								),
				delta_theta =dict(
								policy=torch.zeros(self.theta_shape['policy']).to(self.device),
								q1=torch.zeros(self.theta_shape['q1']).to(self.device),
								q2=torch.zeros(self.theta_shape['q2']).to(self.device),
								),
				)
		else:

			self.theta_shape = dict(
							policy=self.policy.get_theta(task_id=self.task_id, copy=True).shape
							)

			self.importance_vars = dict(
				prev_theta 	=dict(
								policy=torch.zeros(self.theta_shape['policy']).to(self.device),
								),
				omega_total =dict(
								policy=torch.zeros(self.theta_shape['policy']).to(self.device),
								),
				omega 		=dict(
								policy=torch.zeros(self.theta_shape['policy']).to(self.device),
								),
				delta_theta =dict(
								policy=torch.zeros(self.theta_shape['policy']).to(self.device),
								),
				)

		self.importance_params = self.algo_config['importance_params']	

	@property
	def networks(self):
		base_list = [
			self.policy,
			self.q1,
			self.q2,
			self.q1_target,
			self.q2_target,
		]
		return base_list

	def _get_tensor_values(self, obs, actions, network=None):
		num_repeat = int (actions.shape[0] / obs.shape[0])
		obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
		preds = network(obs_temp, actions)
		preds = preds.view(obs.shape[0], num_repeat, 1)
		return preds

	def _get_policy_actions(self, obs, num_actions, network=None):

		# From [N, Channel, Width, Height] to [N*num_actions, Channel, Width, Height]
		# From [N, 6912] to [N*num_actions, 6912]
		obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
		new_obs_actions, new_obs_log_pi, _, _ = network.get_action(obs_temp)
		return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)

	def _compute_policy_loss(self, states, actions, rewards, next_states, dones):
		a_pi, log_pi, policy_mean, policy_std = self.policy.get_action(states)

		if self.use_automatic_entropy_tuning:
			alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
			# al = (0.1--->0.9)*(log_pi-8) 
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
			self.alpha = self.log_alpha.exp().item() 

		q1_pi = self.q1(states, a_pi) #dim - [N, 1]
		q2_pi = self.q2(states, a_pi)   
		min_q_pi = torch.min(q1_pi, q2_pi)

		policy_loss = (self.alpha * log_pi - min_q_pi).mean()

		if not self.train_cql:
			# print('behaviour clonning')
			policy_log_prob = self.policy.log_prob(states, actions)
			policy_loss = (self.alpha * log_pi - policy_log_prob).mean()

		with torch.no_grad():
			if self.flag_collect_stats:

				policy_loss_tmp = (self.alpha *log_pi - min_q_pi).mean().detach()

				self.logger.stats_dict.update(util.create_stats_ordered_dict(
					'q1_pi',
					util.to_numpy(q1_pi),
				))

				self.logger.stats_dict.update(util.create_stats_ordered_dict(
					'q2_pi',
					util.to_numpy(q1_pi),
				))

				self.logger.stats_dict.update(util.create_stats_ordered_dict(
					'min_q_pi',
					util.to_numpy(min_q_pi),
				))

				self.logger.stats_dict.update(util.create_stats_ordered_dict(
					'Policy mu',
					util.to_numpy(policy_mean),
				))
				self.logger.stats_dict.update(util.create_stats_ordered_dict(
					'Policy std',
					util.to_numpy(policy_std),
				))

				self.logger.stats_dict.update(util.create_stats_ordered_dict(
					'Log Pis',
					util.to_numpy(log_pi),
				))

				self.logger.stats_dict['Policy Loss'] = util.to_numpy(policy_loss)
				
				self.logger.stats_dict['Policy Loss (cql)'] = np.mean(util.to_numpy(policy_loss_tmp)) ## alpha=1 set i.e. temparature is 0

				if self.use_automatic_entropy_tuning:
					self.logger.stats_dict['Alpha'] = self.alpha
					self.logger.stats_dict['Alpha Loss'] = alpha_loss.item()
					self.logger.stats_dict['log alpha'] = self.log_alpha.item()

		return policy_loss

	def _compute_q_loss(self, states, actions, rewards, next_states, dones):
		
		# Normal Q-Loss
		with torch.no_grad():
			next_state_actions, next_state_log_pis, _, _ = self.policy.get_action(next_states)
			min_q_target = torch.min(
				self.q1_target(next_states, next_state_actions),
				self.q2_target(next_states, next_state_actions)
				)

			# min_q_target = min_q_target- self.alpha * next_state_log_pis
			q_target = rewards + (1.0 - dones) * self.gamma * (min_q_target.view(-1))  # view(-1) --> [N, 1] to [N]
			

		q1_pred = self.q1(states, actions).view(-1)  # view(-1) --> [N, 1] to [N]
		q2_pred = self.q2(states, actions).view(-1)  # view(-1) --> [N, 1] to [N]

		q1_loss = self.loss_fn(q1_pred, q_target)  
		q2_loss = self.loss_fn(q2_pred, q_target)

		# Here q_target is without_gradient i.e. detached from computation/optimisatio graph, thus acting as ground-truth
		# In other words weights of Q1 will be adjust such that, q1_pred matches q_target


		# Computing CQL Loss
		# [N*r_n, 8]
		random_actions_tensor = torch.FloatTensor(q1_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).to(self.device)
		
		# actions: [N*r_n, 8]   log_pis: [N, n_r, 1]
		curr_state_actions_tensor, curr_state_log_pis_tensor = self._get_policy_actions(states, num_actions=self.num_random, network=self.policy)
		next_state_actions_tensor, next_state_log_pis_tensor = self._get_policy_actions(next_states, num_actions=self.num_random, network=self.policy)

		# [N, n_r, 1] Q value
		q1_rand = self._get_tensor_values(states, random_actions_tensor, network=self.q1)
		q1_curr_actions = self._get_tensor_values(states, curr_state_actions_tensor, network=self.q1)
		q1_next_actions = self._get_tensor_values(states, next_state_actions_tensor, network=self.q1)
		q2_rand = self._get_tensor_values(states, random_actions_tensor, network=self.q2)
		q2_curr_actions = self._get_tensor_values(states, curr_state_actions_tensor, network=self.q2)
		q2_next_actions = self._get_tensor_values(states, next_state_actions_tensor, network=self.q2)


		if self.min_q_version == 3:
			# importance sampled version
			random_density = np.log(0.5 ** curr_state_actions_tensor.shape[-1])
			cat_q1 = torch.cat(
				[q1_rand - random_density, q1_next_actions - next_state_log_pis_tensor.detach(), q1_curr_actions - curr_state_log_pis_tensor.detach()], 1
			)
			cat_q2 = torch.cat(
				[q2_rand - random_density, q2_next_actions - next_state_log_pis_tensor.detach(), q2_curr_actions - curr_state_log_pis_tensor.detach()], 1
			)
		else:
			cat_q1 = torch.cat(
				[q1_rand, q1_pred.unsqueeze(1), q1_next_actions,
				 q1_curr_actions], 1
			)
			cat_q2 = torch.cat(
				[q2_rand, q2_pred.unsqueeze(1), q2_next_actions,
				 q2_curr_actions], 1
			)

		std_q1 = torch.std(cat_q1, dim=1) # Standard Deviation of Q1(s, a_r), Q1(s, a_s), Q1(s, a_ns)
		std_q2 = torch.std(cat_q2, dim=1)

		# cat_Q[N, 3*n_r, 1] concatenated
		q1_cql_loss = torch.logsumexp(cat_q1 / self.cql_temp, dim=1,).mean()*self.cql_temp - q1_pred.mean()
		q2_cql_loss = torch.logsumexp(cat_q2 / self.cql_temp, dim=1,).mean()*self.cql_temp - q2_pred.mean()

		# Adding CQL loss to normal Q-loss
		q1_loss_total = q1_loss + self.cql_alpha*q1_cql_loss
		q2_loss_total = q2_loss + self.cql_alpha*q2_cql_loss

		q_loss = (q1_loss_total + q2_loss_total).detach() / 2

		with torch.no_grad():
			if self.flag_collect_stats:

				self.logger.stats_dict['Q Loss'] = np.mean(util.to_numpy(q_loss))

				self.logger.stats_dict['QF1 Loss'] = np.mean(util.to_numpy(q1_loss))
				self.logger.stats_dict['Q1-CQL Loss'] = np.mean(util.to_numpy(q1_cql_loss))
				self.logger.stats_dict['QF1 Loss Total'] = np.mean(util.to_numpy(q1_loss_total))

				self.logger.stats_dict['QF2 Loss'] = np.mean(util.to_numpy(q2_loss))
				self.logger.stats_dict['Q2-CQL Loss'] = np.mean(util.to_numpy(q2_cql_loss))
				self.logger.stats_dict['QF2 Loss Total'] = np.mean(util.to_numpy(q2_loss_total))

				self.logger.stats_dict['Std QF1 values'] = np.mean(util.to_numpy(std_q1))
				self.logger.stats_dict['Std QF2 values'] = np.mean(util.to_numpy(std_q2))

				self.logger.stats_dict.update(util.create_stats_ordered_dict('QF1 in-distribution values', util.to_numpy(q1_curr_actions)))
				self.logger.stats_dict.update(util.create_stats_ordered_dict('QF2 in-distribution values', util.to_numpy(q2_curr_actions)))

				self.logger.stats_dict.update(util.create_stats_ordered_dict('QF1 random values', util.to_numpy(q1_rand)))
				self.logger.stats_dict.update(util.create_stats_ordered_dict('QF2 random values', util.to_numpy(q2_rand)))

				self.logger.stats_dict.update(util.create_stats_ordered_dict('QF1 next values', util.to_numpy(q1_next_actions)))
				self.logger.stats_dict.update(util.create_stats_ordered_dict('QF2 next values', util.to_numpy(q2_next_actions)))

				self.logger.stats_dict.update(util.create_stats_ordered_dict('actions', util.to_numpy(actions)))
				self.logger.stats_dict.update(util.create_stats_ordered_dict('rewards', util.to_numpy(rewards)))

				self.logger.stats_dict.update(util.create_stats_ordered_dict('Q1 pred', util.to_numpy(q1_pred)))
				self.logger.stats_dict.update(util.create_stats_ordered_dict('Q2 pred', util.to_numpy(q2_pred)))

				self.logger.stats_dict.update(util.create_stats_ordered_dict('Q target', util.to_numpy(q_target)))

				self.logger.stats_dict.update(util.create_stats_ordered_dict('min Q target', util.to_numpy(min_q_target)))

		return q1_loss_total, q2_loss_total

	def _update_target_q_network(self):
		for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def _get_model_theta(self, copy=False):

		if self.regularize_critic:
			return dict(
				policy = self.policy.get_theta(task_id=self.task_id, copy=copy).to(self.device),
				q1 = self.q1.get_theta(task_id=self.task_id, copy=copy).to(self.device),
				q2 = self.q2.get_theta(task_id=self.task_id, copy=copy).to(self.device),
				)
		else:
			return dict(
				policy = self.policy.get_theta(task_id=self.task_id, copy=copy).to(self.device),
				)

	def _get_model_diff_theta(self, theta):
		if self.regularize_critic:
			return dict(
				policy = self.policy.get_theta(task_id=self.task_id, copy=True).to(self.device)-theta['policy'],
				q1 = self.q1.get_theta(task_id=self.task_id, copy=True).to(self.device)-theta['q1'],
				q2 = self.q2.get_theta(task_id=self.task_id, copy=True).to(self.device)-theta['q2'],
				)
		else:
			return dict(
				policy = self.policy.get_theta(task_id=self.task_id, copy=True).to(self.device)-theta['policy'],
				)

	def _get_model_theta_grads(self):
		if self.regularize_critic:
			return dict(
				policy = self.policy.get_theta_grads(task_id=self.task_id).to(self.device),
				q1 = self.q1.get_theta_grads(task_id=self.task_id).to(self.device),
				q2 = self.q2.get_theta_grads(task_id=self.task_id).to(self.device),
				)
		else:
			return dict(
				policy = self.policy.get_theta_grads(task_id=self.task_id).to(self.device),
				)

	def _quad_loss(self, omega, curr_theta, prev_theta):
		return torch.sum(omega*torch.square(curr_theta-prev_theta))

	def _compute_omega_task(self, grads2, curr_theta, prev_theta, epsilon=1e-3):
		return grads2/(torch.square(curr_theta-prev_theta) + epsilon)

	def _update_omega(self, step_prev_theta):
		step_delta_theta = self._get_model_diff_theta(step_prev_theta)
		step_grad_theta = self._get_model_theta_grads()


		#omega (sum of products of gradient and delta_theta (of one step))
		if self.regularize_critic:
			self.importance_vars['omega'] = dict(
												policy = self.importance_vars['omega']['policy'] - step_grad_theta['policy']*step_delta_theta['policy'],
												q1 = self.importance_vars['omega']['q1'] - step_grad_theta['q1']*step_delta_theta['q1'],
												q2 = self.importance_vars['omega']['q2'] - step_grad_theta['q2']*step_delta_theta['q2'],
											)
		else:
			self.importance_vars['omega'] = dict(
												policy = self.importance_vars['omega']['policy'] - step_grad_theta['policy']*step_delta_theta['policy'],
											)

		# print('\n----inside update omega----')
		# print('step_prev_theta')
		# pprint.pprint(step_prev_theta)
		# print('step_delta_theta')
		# pprint.pprint(step_delta_theta)
		# print('step_grad_theta')
		# pprint.pprint(step_grad_theta)
		# print('importance_vars')
		# pprint.pprint(self.importance_vars)
		# print('\n')
		# self.importance_vars['omega']['policy'] = importance_vars['omega']['policy'] - step_grad_theta['policy']*step_delta_theta['policy']
		# self.importance_vars['omega']['q1'] = importance_vars['omega']['q1'] - step_grad_theta['q1']*step_delta_theta['q1']
		# self.importance_vars['omega']['q2'] = importance_vars['omega']['q2'] - step_grad_theta['q2']*step_delta_theta['q2']

	def update_importance_vars(self):

		# print('\n----Before Update after TASK------')
		# pprint.pprint(self.importance_vars)

		self.importance_vars['delta_theta'] =  self._get_model_diff_theta(self.importance_vars['prev_theta'])

		if self.regularize_critic:
			omega_task = dict(
								policy=self._compute_omega_task(
									self.importance_vars['omega']['policy'],
									self.importance_vars['delta_theta']['policy'],
									self.importance_params['epsilon']
									),
								q1=self._compute_omega_task(
									self.importance_vars['omega']['q1'],
									self.importance_vars['delta_theta']['q1'],
									self.importance_params['epsilon']
									),
								q2=self._compute_omega_task(
									self.importance_vars['omega']['q2'],
									self.importance_vars['delta_theta']['q2'],
									self.importance_params['epsilon']
									)
								)

			self.importance_vars['omega_total'] = dict(
													policy=F.relu(self.importance_vars['omega_total']['policy'] + omega_task['policy']),
													q1=F.relu(self.importance_vars['omega_total']['q1'] + omega_task['q1']),
													q2=F.relu(self.importance_vars['omega_total']['q2'] + omega_task['q2']),
													)

			self.importance_vars['omega'] = dict(
												policy=torch.zeros(self.theta_shape['policy']).to(self.device),
												q1=torch.zeros(self.theta_shape['q1']).to(self.device),
												q2=torch.zeros(self.theta_shape['q2']).to(self.device),
												)
		else:
			omega_task = dict(
								policy=self._compute_omega_task(
									self.importance_vars['omega']['policy'],
									self.importance_vars['delta_theta']['policy'],
									self.importance_params['epsilon']
									),
								)

			self.importance_vars['omega_total'] = dict(
													policy=F.relu(self.importance_vars['omega_total']['policy'] + omega_task['policy']),
													)

			self.importance_vars['omega'] = dict(
												policy=torch.zeros(self.theta_shape['policy']).to(self.device),
												)


		self.importance_vars['prev_theta'] = self._get_model_theta(copy=True)

		# print('----After Update after TASK------')
		# pprint.pprint(self.importance_vars)
		# print('\n\n')

	def set_task_id(self, task_id):
		self.task_id = task_id
		for net in self.networks:
			net.set_output_head_idx(task_id)

	def set_new_task(self, task_id):

		# Resetting Optimizers
		self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.q_lr)
		self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.q_lr)
		self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

		# Resetting alpha (entropy control variable)
		if self.use_automatic_entropy_tuning:
			self.target_entropy = - torch.prod(torch.Tensor((self.action_space,)).to(self.device)).item()
			self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha = self.log_alpha.exp().item()
			self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
		else:
			self.alpha = 0.2

		self.task_id = task_id
		for net in self.networks:
			net.set_output_head_idx(task_id)

		# Prev step theta
		self.step_prev_theta = self._get_model_theta(copy=True)

	def train(self, batch):
		states = batch['obs']
		actions = batch['actions']
		rewards = batch['rewards']
		next_states = batch['next_obs']
		dones = batch['terminals']

		# gt.stamp('train_load', unique=False)

		# print(f"inside rl_agent train fn")

		## Setting all Neural Networks in training Mode
		for net in self.networks:
			net.train(True)

		# gt.stamp('train_on', unique=False)

		policy_loss = self._compute_policy_loss(states, actions, rewards, next_states, dones)
		# gt.stamp('train_p_loss', unique=False)
		q1_loss, q2_loss = self._compute_q_loss(states, actions, rewards, next_states, dones)
		# gt.stamp('train_q_loss', unique=False)
		
		# print(f"self.task_id {self.task_id}")
		if self.regularize_critic:
			surrogate_loss = dict(
								policy=0.0,
								q1=0.0,
								q2=0.0,
								)
		else:
			surrogate_loss = dict(
								policy=0.0,
								)
		if (self.task_id>0):

			curr_theta = self._get_model_theta()

			surrogate_loss['policy'] =  self._quad_loss(self.importance_vars['omega_total']['policy'], self. importance_vars['prev_theta']['policy'], curr_theta['policy'])
			if self.regularize_critic:
				surrogate_loss['q1'] =  self._quad_loss(self.importance_vars['omega_total']['q1'], self. importance_vars['prev_theta']['q1'], curr_theta['q1'])
				surrogate_loss['q2'] =  self._quad_loss(self.importance_vars['omega_total']['q2'], self. importance_vars['prev_theta']['q2'], curr_theta['q2'])

			policy_loss = policy_loss + self.importance_params['c']*surrogate_loss['policy']
			if self.regularize_critic:
				q1_loss = q1_loss + self.importance_params['c']*surrogate_loss['q1']
				q2_loss = q2_loss + self.importance_params['c']*surrogate_loss['q2']

			if self.flag_collect_stats:
				self.logger.stats_dict['s_loss_policy'] = np.mean(util.to_numpy(surrogate_loss['policy']))
				if self.regularize_critic:
					self.logger.stats_dict['s_loss_q1'] = np.mean(util.to_numpy(surrogate_loss['q1']))
					self.logger.stats_dict['s_loss_q2'] = np.mean(util.to_numpy(surrogate_loss['q2']))
		else:
			if self.flag_collect_stats:
				self.logger.stats_dict['s_loss_policy'] = 0.0
				if self.regularize_critic:
					self.logger.stats_dict['s_loss_q1'] = 0.0
					self.logger.stats_dict['s_loss_q2'] = 0.0

		# gt.stamp('train_si_loss', unique=False)
		
		


		self.q1_optimizer.zero_grad()
		q1_loss.backward(retain_graph=True)
		# gt.stamp('train_q1_back', unique=False)

		self.q2_optimizer.zero_grad()
		q2_loss.backward(retain_graph=True)
		# gt.stamp('train_q2_back', unique=False)
		
		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		# gt.stamp('train_p_back', unique=False)

		self.q1_optimizer.step()
		# gt.stamp('train_q1_step', unique=False)
		self.q2_optimizer.step()
		# gt.stamp('train_q2_step', unique=False)
		self.policy_optimizer.step()
		# gt.stamp('train_p_step', unique=False)

		self._update_target_q_network() #soft copy from q to q_target
		# gt.stamp('train_qt_update', unique=False)

		## Setting all Neural Networks in eval Mode
		for net in self.networks:
			net.train(False)
		# gt.stamp('train_off', unique=False)

		## update omega
		self._update_omega(self.step_prev_theta)

		self.step_prev_theta = self._get_model_theta(copy=True)
		# gt.stamp('update_omega', unique=False)
				

	## This function takes image_input (from simulator) and returns the action
	def get_action(self, obs):
		obs = torch.from_numpy(np.expand_dims(np.float64(obs), axis=0)).type('torch.FloatTensor').to(self.device)
		action_tensor, _, _, _ = self.policy.get_action(obs, deterministic=True)
		action = action_tensor.detach().cpu().numpy()[0].astype(np.float64)
		
		return action