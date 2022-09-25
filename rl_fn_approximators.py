from torch import nn as nn
import torch
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F

import util

class PolicyNetwork(nn.Module):
	def __init__(
			self,
			input_width,
			input_height,
			input_channels,
			output_size,

			kernel_sizes_lst,
			n_channels_lst,
			strides_lst,
			paddings_lst,
			conv_normalization_type='none',
			fc_normalization_type='none',
			pool_type='none',
			pool_sizes=None,
			pool_strides=None,
			pool_paddings=None,
			fc_hidden_sizes=None,
			init_w=1e-4,
			hidden_init_fn=nn.init.xavier_uniform_,
			hidden_activation_fn=nn.ReLU(),
			output_activation_fn=nn.Identity(),

			added_fc_input_size=0,
			output_heads_num=1,
			augmentation_transform=None,

			log_std_max=2,
			log_std_min=-5,
			mean_max=9.0,
			mean_min=-9.0,
			):

		super().__init__()

		self.output_heads_num = output_heads_num
		self.output_head_idx = 0

		self.log_std_max = log_std_max
		self.log_std_min = log_std_min
		self.mean_max = mean_max
		self.mean_min = mean_min

		self.conv1 = nn.Conv2d(3, 16, (3, 3), padding='same')
		self.conv2 = nn.Conv2d(16, 16, (3, 3), padding='same')
		self.conv3 = nn.Conv2d(16, 16, (3, 3), padding='same')

		self.fc1 = nn.Linear(16 * 12 * 12, 1024)  # 16*12*12 from previous and 8 for action
		self.fc2 = nn.Linear(1024, 512)
		self.fc3 = nn.Linear(512, 256)

		self.fc4 = nn.Linear(256, 256)
		self.fc5 = nn.Linear(256, 256)
		self.fc6 = nn.Linear(256, 256)

		self.mean_output_heads = nn.ModuleList()
		self.logstd_output_heads = nn.ModuleList()

		for i in range(self.output_heads_num):
			mean_layer = nn.Linear(256, output_size)
			logstd_layer = nn.Linear(256, output_size)
			self.mean_output_heads.append(mean_layer)
			self.logstd_output_heads.append(logstd_layer)

		## weights and bias initialisation
		torch.nn.init.xavier_uniform_(self.conv1.weight)
		torch.nn.init.xavier_uniform_(self.conv2.weight)
		torch.nn.init.xavier_uniform_(self.conv2.weight)
		self.conv1.bias.data.zero_()
		self.conv2.bias.data.zero_()
		self.conv3.bias.data.zero_()

		init_w = 1e-4
		torch.nn.init.uniform_(self.fc1.weight, -init_w, init_w)
		torch.nn.init.uniform_(self.fc2.weight, -init_w, init_w)
		torch.nn.init.uniform_(self.fc3.weight, -init_w, init_w)
		torch.nn.init.uniform_(self.fc1.bias, -init_w, init_w)
		torch.nn.init.uniform_(self.fc2.bias, -init_w, init_w)
		torch.nn.init.uniform_(self.fc3.bias, -init_w, init_w)

		util.fanin_init(self.fc4.weight)
		util.fanin_init(self.fc5.weight)
		util.fanin_init(self.fc6.weight)
		self.fc4.bias.data.fill_(0.1)
		self.fc5.bias.data.fill_(0.1)
		self.fc6.bias.data.fill_(0.1)

		init_w = 31e-3
		for i in range(self.output_heads_num):
			torch.nn.init.uniform_(self.mean_output_heads[i].weight, -init_w, init_w)
			torch.nn.init.uniform_(self.logstd_output_heads[i].weight, -init_w, init_w)
			torch.nn.init.uniform_(self.mean_output_heads[i].bias, -init_w, init_w)
			torch.nn.init.uniform_(self.logstd_output_heads[i].bias, -init_w, init_w)
		## Weight and bias init done

		self.augmentation_transform = augmentation_transform

	def set_output_head_idx(self, output_head_idx):
		self.output_head_idx = output_head_idx

	def forward(self, conv_input):
		
		x = conv_input.view(conv_input.shape[0], 3, 48, 48)
		## x.shape -> n, 3, 48, 48

		if self.training:  # Only done in training
			x = self.augmentation_transform(x)

		x = F.relu(F.max_pool2d(self.conv1(x), 2,2))  # -> n, (3,48,48 -- 16,24,24)
		x = F.relu(F.max_pool2d(self.conv2(x), 2,2))  # -> n, (16,24,24 -- 16,12,12)
		x = F.relu(self.conv3(x))  # -> n, (16,12,12 -- 16,12,12)

		x = x.view(-1, 16 * 12 * 12)  # -> n, 2304

		x = F.relu(self.fc1(x))  # -> n, 2304-1024
		x = F.relu(self.fc2(x))  # -> n, 1024-512
		x = self.fc3(x)  # -> n, 512-256

		x = F.relu(self.fc4(x)) # -> n, 256-256
		x = F.relu(self.fc5(x)) # -> n, 256-256
		x = F.relu(self.fc6(x)) # -> n, 256-256

		mean = self.mean_output_heads[self.output_head_idx](x)
		log_std = self.logstd_output_heads[self.output_head_idx](x)

		# we ouput log_std instead of std because SGD is not good at constrained optimisation
		# we want std to be positive i.e. >0 but what we do is ouput log_std and then take exponential
		# so NN output is negative also i.e. log_std can be negative but std will alwyas be positive
		return mean, log_std

	def get_action(self, x, deterministic=False):
		## This Function get mean and std value from policy NN
		## These values are then used according to policy-type to get action values
		## Here policy type is TanhGaussianPolicy
		## In case of deterministic a = tanh(mean)
		## In case of stochastic a = tanh(x), x~N(mean, std)

		mean, log_std = self.forward(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		std = log_std.exp()

		log_prob = None
		if(deterministic):
			action = torch.tanh(mean)
		else:
			normal = Normal(mean, std)  # creating normal distribution from mean, std

			x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
			x_t.requires_grad_()
			y_t = torch.tanh(x_t)  # scaling ouput from normal distribution bw (-1,1)

			## x_t is pre_tanh_value, y_t=action
			action = y_t

			log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
			# returns log of probability of x_t --> log(P(x))
			# Enforcing Action Bound {torch.log((1 - y_t.pow(2)) + 1e-6)}

			log_prob = log_prob.sum(1, keepdim=True) 
			# shape of log_prob is (n, 8) --> sum(1, keepdim=True) --> (n, 1)
			# sums along all actions, keepdim=True keeps dimension same as original i.e. 2
		
		return action, log_prob, mean, std

	def log_prob(self, states, actions):

		mean, log_std = self.forward(states)
		mean = torch.clamp(mean, self.mean_min, self.mean_max)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		std = log_std.exp()

		normal = Normal(mean, std)  # creating normal distribution from mean, std

		x_t = util.atanh(actions)
		y_t = actions

		log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
		# returns log of probability of x_t --> log(P(x))
		# Enforcing Action Bound {torch.log((1 - y_t.pow(2)) + 1e-6)}

		
		log_prob = log_prob.sum(-1)
		# shape of log_prob is (n, 8) --> sum(-1) --> (n)
		# sums along all actions, since here is no keepdim it reduces dimesion to 1 from 2

		return log_prob

	# def get_theta(self, task_id, copy=False):
	# 	shared_param_list = []
	# 	if copy:
	# 		for (name, param) in self.named_parameters():
	# 			shared_param_list.append(param.detach().clone().view(-1))
	# 	else:
	# 		for (name, param) in self.named_parameters():
	# 			shared_param_list.append(param.view(-1))

	# 	return torch.cat(shared_param_list)

	# def get_theta_grads(self, task_id):
	# 	grads = []
	# 	for (name, param) in self.named_parameters():
	# 		if param.grad is not None:
	# 		    grads.append(param.grad.detach().clone().view(-1))
	# 		else:
	# 		    grads.append(torch.zeros_like(param).view(-1))
	# 	return torch.cat(grads)

	def get_theta(self, task_id, copy=False):
		shared_param_list = []

		if copy:
			for (name, param) in self.named_parameters():
				# print('name', name)
				if (not 'output_heads' in name) or (f'output_heads.{task_id}' in name):
					shared_param_list.append(param.detach().clone().view(-1))
					# print('yes')
		else:
			for (name, param) in self.named_parameters():
				if (not 'output_heads' in name) or (f'output_heads.{task_id}' in name):
					shared_param_list.append(param.view(-1))

		return torch.cat(shared_param_list)

	def get_theta_grads(self, task_id):
		grads = []
		for (name, param) in self.named_parameters():
				if (not 'output_heads' in name) or (f'output_heads.{task_id}' in name):
					if param.grad is not None:
					    grads.append(param.grad.detach().clone().view(-1))
					else:
					    grads.append(torch.zeros_like(param).view(-1))

		return torch.cat(grads)


class QNetwork(nn.Module):
	def __init__(
			self,
			input_width,
			input_height,
			input_channels,
			output_size,

			kernel_sizes_lst,
			n_channels_lst,
			strides_lst,
			paddings_lst,
			conv_normalization_type='none',
			fc_normalization_type='none',
			pool_type='none',
			pool_sizes=None,
			pool_strides=None,
			pool_paddings=None,
			fc_hidden_sizes=None,
			init_w=1e-4,
			hidden_init_fn=nn.init.xavier_uniform_,
			hidden_activation_fn=nn.ReLU(),
			output_activation_fn=nn.Identity(),

			added_fc_input_size=0,
			output_heads_num=1,
			augmentation_transform=None,
			):

		super().__init__()

		self.output_heads_num = output_heads_num
		self.output_head_idx = 0

		self.conv1 = nn.Conv2d(3, 16, (3, 3), padding='same')
		self.conv2 = nn.Conv2d(16, 16, (3, 3), padding='same')
		self.conv3 = nn.Conv2d(16, 16, (3, 3), padding='same')

		self.fc1 = nn.Linear(16 * 12 * 12 + 8, 1024)  # 16*12*12 from previous and 8 for action
		self.fc2 = nn.Linear(1024, 512)
		self.fc3 = nn.Linear(512, 256)

		self.fc4_output_heads = nn.ModuleList()

		for i in range(self.output_heads_num):
			fc4_layer = nn.Linear(256, output_size)
			self.fc4_output_heads.append(fc4_layer)

		## weights and bias initialisation
		torch.nn.init.xavier_uniform_(self.conv1.weight)
		torch.nn.init.xavier_uniform_(self.conv2.weight)
		torch.nn.init.xavier_uniform_(self.conv2.weight)
		self.conv1.bias.data.zero_()
		self.conv2.bias.data.zero_()
		self.conv3.bias.data.zero_()

		init_w = 1e-4
		torch.nn.init.uniform_(self.fc1.weight, -init_w, init_w)
		torch.nn.init.uniform_(self.fc2.weight, -init_w, init_w)
		torch.nn.init.uniform_(self.fc3.weight, -init_w, init_w)
		torch.nn.init.uniform_(self.fc1.bias, -init_w, init_w)
		torch.nn.init.uniform_(self.fc2.bias, -init_w, init_w)
		torch.nn.init.uniform_(self.fc3.bias, -init_w, init_w)
		for i in range(self.output_heads_num):
			torch.nn.init.uniform_(self.fc4_output_heads[i].weight, -init_w, init_w)
			torch.nn.init.uniform_(self.fc4_output_heads[i].bias, -init_w, init_w)
		## Weight and bias init done

		self.augmentation_transform = augmentation_transform

	def set_output_head_idx(self, output_head_idx):
		self.output_head_idx = output_head_idx

	def forward(self, conv_input, a):

		x = conv_input.view(conv_input.shape[0], 3, 48, 48)
		# x.shape -> n, 3, 48, 48

		if self.training:  # Only done in training
			x = self.augmentation_transform(x)

		x = F.relu(F.max_pool2d(self.conv1(x), 2,2))  # -> n, 16, 24, 24
		x = F.relu(F.max_pool2d(self.conv2(x), 2,2))  # -> n, 16, 12, 12
		x = F.relu(self.conv3(x))  # -> n, 16, 12, 12

		x = x.view(-1, 16 * 12 * 12)  # -> n, 2304
		x = torch.cat([x, a], 1)  # -> n, (2304 + 8 = 2312)

		x = F.relu(self.fc1(x))  # -> n, 1024
		x = F.relu(self.fc2(x))  # -> n, 512
		x = F.relu(self.fc3(x))  # -> n, 256
		x = self.fc4_output_heads[self.output_head_idx](x)  # -> n, 1

		return x

	# def get_theta(self, task_id, copy=False):
	# 	shared_param_list = []
	# 	if copy:
	# 		for (name, param) in self.named_parameters():
	# 			shared_param_list.append(param.detach().clone().view(-1))
	# 	else:
	# 		for (name, param) in self.named_parameters():
	# 			shared_param_list.append(param.view(-1))

	# 	return torch.cat(shared_param_list)

	# def get_theta_grads(self, task_id):
	# 	grads = []
	# 	for (name, param) in self.named_parameters():
	# 		if param.grad is not None:
	# 		    grads.append(param.grad.detach().clone().view(-1))
	# 		else:
	# 		    grads.append(torch.zeros_like(param).view(-1))
	# 	return torch.cat(grads)

	def get_theta(self, task_id, copy=False):
		shared_param_list = []

		if copy:
			for (name, param) in self.named_parameters():
				# print('name', name)
				if (not 'output_heads' in name) or (f'output_heads.{task_id}' in name):
					shared_param_list.append(param.detach().clone().view(-1))
					# print('yes')
		else:
			for (name, param) in self.named_parameters():
				if (not 'output_heads' in name) or (f'output_heads.{task_id}' in name):
					shared_param_list.append(param.view(-1))

		return torch.cat(shared_param_list)

	def get_theta_grads(self, task_id):
		grads = []
		for (name, param) in self.named_parameters():
				if (not 'output_heads' in name) or (f'output_heads.{task_id}' in name):
					if param.grad is not None:
					    grads.append(param.grad.detach().clone().view(-1))
					else:
					    grads.append(torch.zeros_like(param).view(-1))

		return torch.cat(grads)
