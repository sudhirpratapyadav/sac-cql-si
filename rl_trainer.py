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
import argparse

from robo_sim import RoboverseSim
from rl_agent import RlAgent
import util

import pprint
import copy

# import gtimer as gt

class RlTrainer:
	def __init__(self, param_config):

		self.param_config = param_config
		self.dataset_config = self.param_config['dataset_config']
		self.trainer_config = self.param_config['trainer_kwargs']
		self.tasks = param_config['tasks']
		self.logger = util.Logger(param_config['log_params'])

		if torch.cuda.is_available():
			os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
			memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
			free_gpu_id =  np.argmax(memory_available)

			self.logger.log("Memory: ", memory_available)
			self.logger.log("free_gpu_id: ", free_gpu_id)

			self.device = torch.device(f'cuda:{free_gpu_id}')
		else:
			self.device = torch.device('cpu')

		self.rl_agent = RlAgent(copy.deepcopy(param_config), self.logger, device = self.device)

		self.logger.log(f"\nparam_config:{self.param_config}")
		self.logger.log(f"\nrl_agent:{self.rl_agent.__dict__}")

		self.accuracy_avg = []
		self.accuracy_mat = np.full((len(self.tasks),len(self.tasks)), 0.0)
		self.num_successful_epochs_counter = 0

	def _end_epoch(self, rl_agent, epoch, end_epoch_config, simulator):

		if(epoch % self.trainer_config['eval_epoch']==0):

			# self._eval_policy_on_training_data(rl_agent, end_epoch_config)

			stats_dict, paths = self._eval_agent_in_env(rl_agent)

			eval_dict = util.concat_dicts(
					dicts = (
							OrderedDict({'epoch':epoch}),
							stats_dict,
							self.logger.stats_dict
							),
					prefixes = (
						'',
						'evaluation',
						'train')
					)

			## Checking whether task is completed
			# print(f"eval_dict['train-min_q_pi Mean'] {eval_dict['train-min_q_pi Mean']}")
			# print(eval_dict[f"evaluation-{self.current_task['task_id']}-Accuracy"])
			
			task_completed = False
			if epoch > self.trainer_config['min_epochs_per_task']:
				if (eval_dict['train-min_q_pi Mean']>self.trainer_config['min_q_val_success'] and eval_dict[f"evaluation-{self.current_task['task_id']}-Accuracy"]>self.trainer_config['min_eval_accuracy']):
					self.num_successful_epochs_counter +=1
					if self.num_successful_epochs_counter >= self.trainer_config['num_successful_epochs']:
						task_completed = True
				else:
					self.num_successful_epochs_counter = 0


			if(epoch % self.trainer_config['save_checkpoint_epoch']==0) or task_completed:
				agent_clone = copy.deepcopy(rl_agent)
				self.logger.save_checkpoint(self.current_task['task_id'], epoch, dict(agent=agent_clone))
				del agent_clone

			if(epoch % self.trainer_config['save_gifs_epoch']==0) or task_completed:
				self.logger.save_gif(self.current_task['task_id'], epoch, paths[0:4])

			self.logger.log_eval(task_id=self.current_task['task_id'], epoch=epoch, data=eval_dict)

			self.logger.stats_dict = OrderedDict()
			rl_agent.flag_collect_stats = False

			return task_completed

	# def _eval_policy_on_training_data(self, rl_agent, end_epoch_config):
	# 	rb_size = end_epoch_config['rb_size']
	# 	reward_hist_indices = end_epoch_config['reward_hist_indices']
	# 	path_length = end_epoch_config['path_length']

	# 	selected_indices = []
	# 	reward_hist_selected_indices = [[] for i in range(len(reward_hist_indices))]
	# 	for r_indx in range(len(reward_hist_indices)):
	# 		if len(reward_hist_indices[r_indx])>0:

	# 			selected_index = random.choice(reward_hist_indices[r_indx])
	# 			selected_indices.extend(list(range(selected_index*path_length, (selected_index+1)*path_length)))
	# 			reward_hist_selected_indices[r_indx] = selected_index

	# 	batch = util.get_random_batch(end_epoch_config['replay_buffer'], device=self.device, selected_indices=selected_indices)

	# 	states = batch['obs']
	# 	actions = batch['actions']
	# 	rewards = batch['rewards']
	# 	next_states = batch['next_obs']
	# 	dones = batch['terminals']

	# 	## Setting all Neural Networks in Evaluation Mode
	# 	for net in rl_agent.networks:
	# 		net.train(False)

	# 	with torch.no_grad():
	# 		q1_pred = util.to_numpy(rl_agent.q1(states, actions).view(-1)).reshape((-1, path_length))

	# 		a_pi, _, policy_mean, policy_std = rl_agent.policy.get_action(states, deterministic=True)

	# 		a_pi = util.to_numpy(a_pi).reshape((-1, path_length, rl_agent.action_space))
	# 		policy_mean = util.to_numpy(policy_mean).reshape((-1, path_length, rl_agent.action_space))
	# 		policy_std = util.to_numpy(policy_std).reshape((-1, path_length, rl_agent.action_space))

	# 		self.logger.stats_dict['Q_values_train'] = q1_pred
	# 		self.logger.stats_dict['A_values_train'] = a_pi
	# 		self.logger.stats_dict['pi_mu_values_train'] = policy_mean
	# 		self.logger.stats_dict['pi_std_values_train'] = policy_std

	# def _eval_policy_in_env(self, rl_agent, simulator, num_eval_steps=None, max_path_length=None):

	# 	## Setting all Neural Networks in Evaluation Mode
	# 	for net in rl_agent.networks:
	# 		net.train(False)

	# 	agent_clone = copy.deepcopy(rl_agent)
	# 	simulator.setAgent(agent_clone)

	# 	if max_path_length is None:
	# 		max_path_length = self.trainer_config['max_path_length']
	# 	if num_eval_steps is None:
	# 		num_eval_steps = self.trainer_config['num_eval_steps_per_policy']
		
	# 	paths = simulator.collectPaths(
	# 		max_path_length=max_path_length,
	# 		num_eval_steps=num_eval_steps,
	# 		discard_incomplete_paths=True
	# 		)
	# 	del agent_clone

	# 	return paths

	def _eval_agent_in_env(self, rl_agent, num_eval_steps=None, max_path_length=None):

		# self.logger.log(f"Evaluating after training {self.current_task['task_id']+1} tasks")

		if max_path_length is None:
			max_path_length = self.trainer_config['max_path_length']
		if num_eval_steps is None:
			num_eval_steps = self.trainer_config['num_eval_steps_per_policy_task_eval']


		path_stats = OrderedDict()
		paths_task = None

		acc_lst = []
		with torch.no_grad():
			for net in rl_agent.networks:
				net.train(False)
			agent_clone = copy.deepcopy(self.rl_agent)

			for task in self.tasks:
				self.logger.log(f"task_num {task['task_id']+1}/{len(self.tasks)}")

				agent_clone.set_task_id(task_id=task['task_id'])
				
				simulator_temp = RoboverseSim(task['env_id'])
				simulator_temp.setAgent(agent_clone)
				
				paths = simulator_temp.collectPaths(
					max_path_length=max_path_length,
					num_eval_steps=num_eval_steps,
					discard_incomplete_paths=True
					)

				stat_prefix = str(task['task_id'])+'-'

				path_stats.update(util.get_generic_path_information(paths, stat_prefix=stat_prefix))

				acc = path_stats[stat_prefix + 'Accuracy']
				acc_lst.append(acc)
				self.accuracy_mat[task['task_id'], self.current_task['task_id']] = acc
				self.logger.log(f"Accuracy of the network after training task {self.current_task['task_id']+1} on task {task['task_id']+1}: {acc} %")

				actions = None
				states = None
				for path in paths:
					if actions is None:
						actions = path['actions']
						state = []
						for obs in path['observations']:
							state.append(obs['image'])
						states = np.array(state)
					else:
						actions = np.vstack((actions, path['actions']))
						state = []
						for obs in path['observations']:
							state.append(obs['image'])
						states = np.vstack((states, state))

				actions = torch.FloatTensor(actions).to(self.device)
				states = torch.FloatTensor(states).to(self.device)
				# actions = actions.reshape((, path_length))
				print('\n\n---------\n')
				print(states.shape, actions.shape)
				print(len(paths))

				with torch.no_grad():
					q1_pred = util.to_numpy(agent_clone.q1(states, actions))
					q2_pred = util.to_numpy(agent_clone.q2(states, actions))

					path_stats.update(util.create_stats_ordered_dict('Q1_values', q1_pred, stat_prefix=stat_prefix))
					path_stats.update(util.create_stats_ordered_dict('Q2_values', q2_pred, stat_prefix=stat_prefix))

				if task['task_id']==self.current_task['task_id']:
					paths_task = paths.copy()

			del agent_clone

		print('\nacc_list')
		pprint.pprint(acc_lst)
		print('\n')

		avg_acc = np.array(acc_lst).mean()
		self.accuracy_avg.append(avg_acc)

		self.logger.log(f"Average Accuracy of the network after training for {self.current_task['task_id']+1} tasks: {avg_acc} %")
		self.logger.log(f"Evaluating Done")

		return path_stats, paths_task
	

	def train(self):
		for task in self.tasks:

			self.current_task = task
			self.num_successful_epochs_counter = 0

			replay_buffer, rb_size, reward_hist_indices, path_length, list_lens = util.loadDataFromFile(self.current_task['dataset_path'], self.dataset_config, self.logger)

			simulator = RoboverseSim(self.current_task['env_id'])

			print(f"simulator: {self.current_task['env_id']}")

			end_epoch_config = {
					'replay_buffer':replay_buffer,
					'rb_size':rb_size,
					'reward_hist_indices':reward_hist_indices,
					'path_length':path_length,
					}
					
			eval_dict_temp = OrderedDict({'list_lens':list_lens, 'param_config':self.param_config})
			self.logger.log_eval(self.current_task['task_id'], -1, eval_dict_temp)
			del eval_dict_temp

			self.rl_agent.set_new_task(self.current_task['task_id']) #for head-selection

			print(f"self.current_task['task_id']: {self.current_task['task_id']}")

			# Timing Data
			eta = None
			epoch_time = None
			previous_epoch_time = None
			start_time = time.time()

			# gt.stamp('Training Start')
			# for epoch in gt.timed_for(range(1,self.trainer_config['total_epochs']+1)):
			for epoch in range(1,self.trainer_config['total_epochs']+1):

				self.logger.log(f"epoch:{epoch}\t steps_completed:", end='')
				if(epoch>self.trainer_config['epoch_cql_start']):
					self.rl_agent.train_cql = True
				else:
					self.rl_agent.train_cql = False

				# gt.stamp('epoch_start')
				for step in range(1,self.trainer_config['steps_per_epoch']+1):
					batch = util.get_random_batch(replay_buffer, self.device, rb_size, batch_size=self.trainer_config['batch_size'])
					# gt.stamp('random batch', unique=False)
					if (step==self.trainer_config['steps_per_epoch']) and (epoch%self.trainer_config['eval_epoch'] == 0):
						self.rl_agent.flag_collect_stats = True
					self.rl_agent.train(batch)
					# gt.stamp('train', unique=False)

					# self.logger.log(f"epoch [{epoch}/{self.trainer_config['total_epochs']}] step [{step}/{self.trainer_config['steps_per_epoch']}]")
					# print(f"epoch [{epoch}/{self.trainer_config['total_epochs']}] step [{step}/{self.trainer_config['steps_per_epoch']}]")

					self.logger.display_progress(step, self.trainer_config['steps_per_epoch'], f"[epoch:{epoch}/{self.trainer_config['total_epochs']}]")
					self.logger.log(f'{step},', end='')

				# ## evaluation policy on all tasks
				# self._eval_agent()

				task_completed = self._end_epoch(self.rl_agent, epoch, end_epoch_config, simulator)
				# gt.stamp('eval_time', unique=False)
				
				#Epoch End Timing
				t = time.time()
				if epoch_time is None:
					epoch_time = t - start_time
				else:
					epoch_time = t - previous_epoch_time
				eta = epoch_time*(self.trainer_config['total_epochs']-epoch)
				elapsed_time = t - start_time
				total_time = elapsed_time + eta
				previous_epoch_time = t

				self.logger.log(f"\ntask_{self.current_task['task_id']}_completed: {task_completed}")
				self.logger.log(f"epoch:{epoch}/{self.trainer_config['total_epochs']} epoch_time:{epoch_time:.1f}s \telapsed/total:({util.sec_to_str(elapsed_time)}/{util.sec_to_str(total_time)}) \tETA:{util.sec_to_str(eta)}\n")

				if task_completed:
					break

			# gt.stamp('epoch_end')

			## Updating Importance Variables
			print('\n\n\n\n-------------UPDTAING importance_params------------------------\n\n\n\n')
			self.rl_agent.update_importance_vars()
			# gt.stamp('update_imp_param')

			# print(gt.report())


		exp_result = dict(
			accuracy_mat = self.accuracy_mat,
			accuracy_avg = self.accuracy_avg,
		)

		exp_results = []
		
		exp_results.append((self.param_config['algorithm_kwargs']['importance_params'], exp_result))

		self.logger.log_exp_result(exp_results)

		# return exp_results