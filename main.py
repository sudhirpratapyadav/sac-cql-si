import os
import sys
import yaml
import pprint
import pathlib
import argparse


from rl_trainer import RlTrainer

import torch.nn as nn

def main():

	## Parsing Arguments ##
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset-file", type=str, required = True)
	parser.add_argument("--task-seq", type=str, required = True)
	parser.add_argument("--area", type=str, default="0.53")
	parser.add_argument("--noise", type=str, default="0.1")
	parser.add_argument("--dgx", action='store_true', default=False)
	parser.add_argument("--reg-critic", action='store_true', default=False)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--steps-per-epochs", type=int, default=1000)
	parser.add_argument("--cql-epoch", type=int, default=5)
	parser.add_argument("--cql-alpha", type=float, default=1.0)
	parser.add_argument("--cql-temp", type=float, default=1.0)
	parser.add_argument("--reward-scaling", action='store_true', default=False)
	parser.add_argument("--reward-scale", type=float, default=1.0)
	parser.add_argument("--reward-shift", type=float, default=0.0)
	parser.add_argument("-c", type=float, default=1.0)

	args = parser.parse_args()

	BASE_PATH =  pathlib.Path(__file__).parent

	TASKS = []
	with open((BASE_PATH / args.dataset_file).resolve()) as file:
		task_datasets = yaml.full_load(file)
		for task_id, task_name in enumerate(args.task_seq.split("_")):
			TASKS.append(dict(
							task_id=task_id,
							task_name=task_name,
							env_id=task_datasets[task_name][f'area_{args.area}']['env_id'],
							dataset_path = (BASE_PATH / task_datasets[task_name]['base_path'] / task_datasets[task_name][f'area_{args.area}'][f'noise_{args.noise}']).resolve(),
							description='',))


	save_folder_path = f"si_results/area_{args.area}/noise_{args.noise}/{args.task_seq}"

	if args.reg_critic:
		save_folder_path += '/actor_critic_reg'
	else:
		save_folder_path += '/actor_only_reg'
	if args.c >0:
		save_folder_path += '/results_1'
	else:
		save_folder_path += '/results_0'
	save_folder_path = (BASE_PATH / f'../../RL_RESULTS/{save_folder_path}/').resolve()

	############ Parameters #############

	CNN_PARAMS = dict(
		input_width=48,
		input_height=48,
		input_channels=3,
		output_size=8,

		kernel_sizes_lst=[3, 3, 3],
		n_channels_lst=[16, 16, 16],
		strides_lst=[1, 1, 1],
		paddings_lst=[1, 1, 1],
		conv_normalization_type='none',
		fc_normalization_type='none',
		pool_type='max2d',
		pool_sizes=[2, 2, 1],
		pool_strides=[2, 2, 1],
		pool_paddings=[0, 0, 0],
		fc_hidden_sizes=[1024, 512, 256],
		init_w=1e-4,
		hidden_init_fn=nn.init.xavier_uniform_,
		hidden_activation_fn=nn.ReLU(),
		output_activation_fn=nn.Identity(),

		added_fc_input_size=0,
		output_heads_num=len(TASKS),
		augmentation_transform=None,
		)

	POLICY_PARAMS = dict(
		**CNN_PARAMS,
		log_std_max=2,
		log_std_min=-5,
		mean_max=9.0,
		mean_min=-9.0,
		)

	CNN_PARAMS['output_size']=1
	Q_PARAMS = dict(
		**CNN_PARAMS,
		)

	PARAM_CONFIG = dict(
		algorithm="SI-CQL",
		tasks = TASKS,
		log_params=dict(
			flag_print_file= True,
			save_path=save_folder_path,
			),
		trainer_kwargs=dict(
			save_checkpoint_epoch=50,
			save_gifs_epoch=10,
			eval_epoch=1,
			epoch_cql_start=args.cql_epoch,
			total_epochs=args.epochs,
			steps_per_epoch=args.steps_per_epochs,
			batch_size=256,
			eval_paths_per_policy=4,
			max_path_length=40,
			num_eval_steps_per_policy=4*40,

			num_eval_steps_per_policy_task_eval=10*40,
			min_epochs_per_task=120,
			num_successful_epochs=3,
			min_q_val_success=120,
			min_eval_accuracy=120,
			eval_agent_epoch=1,
		),
		algorithm_kwargs=dict(
			discount=0.99,
			soft_target_tau=5e-3,
			policy_lr=1E-4,
			qf_lr=3E-4,
			use_automatic_entropy_tuning=True,

			# Target nets/ policy vs Q-function update
			policy_eval_start=10000,
			num_qs=2,

			# min Q
			temp=1.0,
			min_q_version=3,
			min_q_weight=5.0,

			# lagrange
			with_lagrange=False,  # Defaults to False
			lagrange_thresh=5.0,

			# extra params
			num_random=1,
			max_q_backup=False,
			deterministic_backup=True,

			state_space_dim=6912,
			action_space_dim=8, 
			automatic_entropy_tuning=True,
			cql_alpha=args.cql_alpha,
			cql_temp=args.cql_temp,

			regularize_critic = args.reg_critic,

			importance_params = dict(
									c = args.c,
									epsilon = 1e-3
									)
		),
		dataset_config=dict(
			reward_scaling=args.reward_scaling,
			reward_scale=args.reward_scale,
			reward_shift=args.reward_shift,
		),
		policy_params=POLICY_PARAMS,
		q_params=Q_PARAMS,
	)

	rl_trainer = RlTrainer(PARAM_CONFIG)
	rl_trainer.train()

if __name__ == "__main__":
	main()

	
