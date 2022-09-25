import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import time

import os
import sys
import torch
import datetime
import dateutil.tz
from collections import OrderedDict
from numbers import Number

import torch
import pickle
import pprint

# from torch.utils.tensorboard import SummaryWriter

def fanin_init(tensor):

	size = tensor.size()
	if len(size) == 2:
		fan_in = size[0]
	else:
		raise Exception("Shape must be have dimension of 2.")

	bound = 1. / np.sqrt(fan_in)
	return tensor.data.uniform_(-bound, bound)

def atanh(x):
	one_plus_x = (1 + x).clamp(min=1e-6)
	one_minus_x = (1 - x).clamp(min=1e-6)
	return 0.5*torch.log(one_plus_x/ one_minus_x)

def get_random_batch(replay_buffer, device, rb_size=None, batch_size=None, selected_indices=None):

	if selected_indices is None:
		indices = np.random.randint(0, rb_size, batch_size)
	else:
		indices = selected_indices

	obs = torch.from_numpy(replay_buffer['observations'][indices]).type('torch.FloatTensor').to(device)
	actions = torch.from_numpy(replay_buffer['actions'][indices]).type('torch.FloatTensor').to(device)
	rewards = torch.from_numpy(replay_buffer['rewards'][indices]).type('torch.FloatTensor').to(device)
	next_obs = torch.from_numpy(replay_buffer['next_observations'][indices]).type('torch.FloatTensor').to(device)
	terminals = torch.from_numpy(replay_buffer['terminals'][indices]).type('torch.FloatTensor').to(device)

	batch = dict(
		obs=obs/255.0,
		actions=actions,
		rewards=rewards,
		next_obs=next_obs/255.0,
		terminals=terminals)

	return batch


def compare_images(x1, x2):
	imgs1 = np.transpose(x1.detach().cpu().numpy(), [0, 2, 3, 1])
	imgs2 = np.transpose(x2.detach().cpu().numpy(), [0, 2, 3, 1])

	num_imgs = imgs1.shape[0]

	imgs = np.vstack((imgs1, imgs2))
	print(imgs.shape)

	num_rows = 2
	num_cols = min(16,num_imgs)
	frames = []
	fig,axs = plt.subplots(num_rows,num_cols)

	for r in range(num_rows):
		for c in range(num_cols):
			axs[r, c].set_title(f'{c}')
			axs[r, c].imshow(imgs[r*num_imgs+c])

	fig.tight_layout()
	plt.show()

def loadDataFromFile(task_data_path, dataset_config, logger):

		s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []


		logger.log("\n-------------------Loading Data-----------------")
	   
		logger.log("Loading Task Data from File ... ")
		raw_data = np.load(task_data_path, allow_pickle=True)

		logger.log("Adding Task Data to replay_buffer ... ")

		ep_len = len(raw_data[0]['observations'])

		returns_lst = []
		for data in raw_data:
			s, a, r, s_prime, done_mask = data['observations'], data['actions'], data['rewards'], \
										  data['next_observations'], data['terminals']
			returns_lst.append(int(np.sum(r)))
			for step in range(ep_len):
				s_lst.append(np.transpose(s[step]['image'], [2, 0, 1]).flatten())
				a_lst.append(a[step])
				if dataset_config['reward_scaling']:
					rw = r[step]*dataset_config['reward_scale']+dataset_config['reward_shift']
					r_lst.append(rw)
				else:
					r_lst.append(r[step])
				s_prime_lst.append(np.transpose(s_prime[step]['image'], [2, 0, 1]).flatten())
				done_mask_lst.append(done_mask[step])

		reward_hist_indices = [[] for i in range(ep_len+1)]
		for indx, rew in enumerate(returns_lst):
			reward_hist_indices[rew].append(indx)

		list_lens = []
		for r_indx in range(len(reward_hist_indices)):
			list_lens.append(len(reward_hist_indices[r_indx]))

		# max_rew_indx = np.argmax(list_lens)
		# reward_hist_nums = np.floor(PARAM_CONFIG['EVAL_TRAIN_PATHS']*np.array(list_lens)/np.array(list_lens).sum()).astype(int)
		# reward_hist_nums[max_rew_indx] = reward_hist_nums[max_rew_indx] + PARAM_CONFIG['EVAL_TRAIN_PATHS']-reward_hist_nums.sum()

		logger.log(f"Finalising Replay Buffer size:{len(s_lst)}")

		replay_buffer = {
			'observations': np.asarray(s_lst),
			'actions': np.asarray(a_lst),
			'rewards': np.asarray(r_lst),
			'next_observations': np.asarray(s_prime_lst),
			'terminals': np.asarray(done_mask_lst),
		}

		# print(replay_buffer['observations'].shape, replay_buffer['observations'].dtype)
		# print(replay_buffer['actions'].shape, replay_buffer['actions'].dtype)
		# print(replay_buffer['rewards'].shape, replay_buffer['rewards'].dtype)
		# print(replay_buffer['terminals'].shape, replay_buffer['terminals'].dtype)
		logger.log("---------------Done Loading Data------------------\n")

		return replay_buffer, len(s_lst), reward_hist_indices, ep_len, list_lens

def dict_of_list__to__list_of_dicts(dict, n_items):
	"""
	```
	x = {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
	ppp.dict_of_list__to__list_of_dicts(x, 3)
	# Output:
	# [
	#     {'foo': 3, 'bar': 1},
	#     {'foo': 4, 'bar': 2},
	#     {'foo': 5, 'bar': 3},
	# ]
	```
	:param dict:
	:param n_items:
	:return:
	"""
	new_dicts = [{} for _ in range(n_items)]
	for key, values in dict.items():
		for i in range(n_items):
			new_dicts[i][key] = values[i]
	return new_dicts

def list_of_dicts__to__dict_of_lists(lst):
	"""
	```
	x = [
		{'foo': 3, 'bar': 1},
		{'foo': 4, 'bar': 2},
		{'foo': 5, 'bar': 3},
	]
	ppp.list_of_dicts__to__dict_of_lists(x)
	# Output:
	# {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
	```
	"""
	if len(lst) == 0:
		return {}
	keys = lst[0].keys()
	output_dict = collections.defaultdict(list)
	for d in lst:
		assert set(d.keys()) == set(keys)
		for k in keys:
			output_dict[k].append(d[k])
	return output_dict

def to_numpy(tensor):
	return tensor.detach().cpu().numpy()

def get_generic_path_information(paths, stat_prefix=''):
	"""
	Get an OrderedDict with a bunch of statistic names and values.
	"""
	statistics = OrderedDict()
	returns = [sum(path["rewards"]) for path in paths]

	rewards = np.vstack([path["rewards"] for path in paths])

	statistics[stat_prefix + 'Accuracy'] = 100*(np.array(returns) > 0).sum()/len(paths)

	statistics.update(create_stats_ordered_dict('Rewards', rewards,
												stat_prefix=stat_prefix))
	statistics.update(create_stats_ordered_dict('Returns', returns,
												stat_prefix=stat_prefix))
	actions = [path["actions"] for path in paths]
	if len(actions[0].shape) == 1:
		actions = np.hstack([path["actions"] for path in paths])
	else:
		actions = np.vstack([path["actions"] for path in paths])
	statistics.update(create_stats_ordered_dict(
		'Actions', actions, stat_prefix=stat_prefix
	))

	statistics[stat_prefix + 'Num Paths'] = len(paths)
	statistics[stat_prefix + 'Average Returns'] = np.mean(returns)

	for info_key in ['env_infos', 'agent_infos']:
		if info_key in paths[0]:
			all_env_infos = [
				list_of_dicts__to__dict_of_lists(p[info_key])
				for p in paths
			]
			for k in all_env_infos[0].keys():
				print("k:",k)
				final_ks = np.array([info[k][-1] for info in all_env_infos])
				first_ks = np.array([info[k][0] for info in all_env_infos])
				all_ks = np.concatenate([info[k] for info in all_env_infos])
				statistics.update(create_stats_ordered_dict(
					stat_prefix + k,
					final_ks,
					stat_prefix='{}/final/'.format(info_key),
				))
				statistics.update(create_stats_ordered_dict(
					stat_prefix + k,
					first_ks,
					stat_prefix='{}/initial/'.format(info_key),
				))
				statistics.update(create_stats_ordered_dict(
					stat_prefix + k,
					all_ks,
					stat_prefix='{}/'.format(info_key),
				))

	return statistics

def concat_dicts(dicts, prefixes=None):

	dict_final = OrderedDict()
	if prefixes is not None:
		assert len(dicts)==len(prefixes)
		for dict, prefix in zip(dicts,prefixes):
			for key, value in dict.items():
				dict_final[prefix+'-'+key] = value
	else:
		for dict in dicts:
			dict_final.update(dict)

	return dict_final



def create_stats_ordered_dict(
		name,
		data,
		stat_prefix=None,
		always_show_all_stats=True,
		exclude_max_min=False,
):
	if stat_prefix is not None:
		name = "{}{}".format(stat_prefix, name)
	if isinstance(data, Number):
		return OrderedDict({name: data})

	if len(data) == 0:
		return OrderedDict()

	if isinstance(data, tuple):
		ordered_dict = OrderedDict()
		for number, d in enumerate(data):
			sub_dict = create_stats_ordered_dict(
				"{0}_{1}".format(name, number),
				d,
			)
			ordered_dict.update(sub_dict)
		return ordered_dict

	if isinstance(data, list):
		try:
			iter(data[0])
		except TypeError:
			pass
		else:
			data = np.concatenate(data)

	if (isinstance(data, np.ndarray) and data.size == 1
			and not always_show_all_stats):
		return OrderedDict({name: float(data)})

	stats = OrderedDict([
		(name + ' Mean', np.mean(data)),
		(name + ' Std', np.std(data)),
	])
	if not exclude_max_min:
		stats[name + ' Max'] = np.max(data)
		stats[name + ' Min'] = np.min(data)
	return stats

def getTimeStamp():
	now = datetime.datetime.now(dateutil.tz.tzlocal())
	timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
	return timestamp

def sec_to_str(sec):
		sec = int(sec)
		s = sec%60
		h = int(sec//3600)
		m = int((sec - h*3600)//60)
		s = int(sec - h*3600 -m*60)
		return f"{h:02d}h:{m:02d}m:{s:02d}s"



class Logger:
	def __init__(self, log_params):
		self._base_dir_path = log_params['save_path']

		self._print_file_path = os.path.join(self._base_dir_path , 'log.txt')
		self._eval_file_path = os.path.join(self._base_dir_path , 'eval.txt')
		self._eval_data_path = os.path.join(self._base_dir_path, 'eval_data.npy')
		self._exp_result_path = os.path.join(self._base_dir_path, 'exp_results.dat')
		self._gif_dir_path = os.path.join(self._base_dir_path, 'gifs/')
		self._checkpoint_dir_path = os.path.join(self._base_dir_path, 'checkpoints/')

		self.make_dir(self._base_dir_path)
		self.make_dir(self._gif_dir_path)
		self.make_dir(self._checkpoint_dir_path)
		self.make_file(self._print_file_path)
		self.make_file(self._eval_file_path)

		self.flag_print_file = log_params['flag_print_file']

		self.eval_data = []

		self.stats_dict = OrderedDict()

		# self.tb_writer = SummaryWriter(log_dir=self._base_dir_path)

	def make_dir(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def make_file(self, file_path):
		with open(file_path, 'w') as f:
			f.write(f'File creation time: {getTimeStamp()}\n\n')

	def save_checkpoint(self, task_id, epoch, data):
		file_name = os.path.join(self._checkpoint_dir_path, f'checkpoint_task_{task_id}_epoch_{epoch}.pkl')
		torch.save(data, file_name)

	def save_gif(self, task_id, epoch, paths):
		path_length = len(paths[0]['observations'])
		num_paths = len(paths)

		num_rows = 2
		num_cols = 2

		frames = []
		fig,axs = plt.subplots(num_rows,num_cols)
		fig.suptitle(f'Task:{task_id}, Epoch:{epoch}')
		for r in range(num_rows):
				for c in range(num_cols):
					indx = (c + r*num_cols)
					axs[r, c].set_title(f'trial-{indx}')

		for i in range(path_length):
			plt_frame = []
			for r in range(num_rows):
				for c in range(num_cols):
					indx = c + r*num_cols
					img = paths[indx]['observations'][i]['image'].reshape(3, 48, 48).transpose(1, 2, 0)

					frame = axs[r, c].imshow(img, animated=True)
					plt_frame.append(frame)

			frames.append(plt_frame)

		fig.tight_layout()
		ani = animation.ArtistAnimation(fig, frames, interval=1000, blit=True, repeat_delay=100)

		writergif = animation.PillowWriter(fps=30)
		gif_file_path = os.path.join(self._gif_dir_path, f'task_{task_id}_epoch_{epoch}.gif')
		ani.save(gif_file_path, writer=writergif)
		plt.close()

	def display_progress(self, step, total_steps, disp_str_pre = '', disp_str_post = '', display_width = 20):
		if self.flag_print_file:
			sys.stdout.write('\r')
			percent = int(100*(step/total_steps))
			percent_step = int(display_width*(step/total_steps))
			sys.stdout.write(f"{disp_str_pre} [{'='*percent_step}>{' '*(display_width-percent_step)}] ({percent}%, step:{step}/{total_steps})  {disp_str_post}")
			sys.stdout.flush()

	def log(self, *args, **kwargs):
		if self.flag_print_file:
			with open(self._print_file_path, 'a') as f:
				print(*args, **kwargs, file=f)
		else:
			print(*args, **kwargs)

	def log_eval(self, task_id, epoch, data, save_to_file=True):
		if save_to_file:
			with open(self._eval_file_path, 'a') as f:
				print(f'task_id:{task_id}\ttime:{getTimeStamp()}', file=f)
				print(f'epoch:{epoch}', file=f)
				print('----------------------------------------------------------------------------', file=f)
				for key, value in data.items():
					print(f'{key}:{value}', file=f)
				print('----------------------------------------------------------------------------\n\n', file=f)

		# print('data')
		# pprint.pprint(data)
		# self.tb_writer.add_scalar("", , epoch)

		self.eval_data.append(data)
		if save_to_file:
			np.save(self._eval_data_path, np.array(self.eval_data), allow_pickle=True)

	def log_exp_result(self, exp_results):
		with open(self._exp_result_path, 'wb') as f:
			pickle.dump(exp_results, f)


class RandomCrop:
	"""
	Source: # https://github.com/pratogab/batch-transforms
	Applies the :class:`~torchvision.transforms.RandomCrop` transform to
	a batch of images.
	Args:
		size (int): Desired output size of the crop.
		padding (int, optional): Optional padding on each border of the image.
			Default is None, i.e no padding.
		dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
		device (torch.device,optional): The device of tensors to which the transform will be applied.
	"""

	def __init__(self, size, padding=None, dtype=torch.float, device='cpu'):
		self.size = size
		self.padding = padding
		self.dtype = dtype
		self.device = device

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
		Returns:
			Tensor: Randomly cropped Tensor.
		"""

		if self.padding is not None:
			padded = torch.zeros((
									tensor.size(0),
									tensor.size(1),
									tensor.size(2) + self.padding * 2,
									tensor.size(3) + self.padding * 2
									),
								 dtype=self.dtype, device=self.device)

			## zero tensor of size (N, C, H+2*ps, W+2*ps) ps=padding_size
			## (32, 3, 48+4*2, 48+4*2) --> (32, 3, 56, 56)

			padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = tensor

			## set (:,:, -ps:ps, -ps:ps) = tensor
			#3 (:,:, 4:52, 4:52) = tensor (img now sits in the middle of padded)

			## Basically it creates paaded image (fills 0, size=padding)

		else:
			padded = tensor

		w, h = padded.size(2), padded.size(3)
		th, tw = self.size, self.size
		if w == tw and h == th:
			i, j = 0, 0
		else:
			i = torch.randint(0, h - th + 1, (tensor.size(0),),
							  device=self.device)
			j = torch.randint(0, w - tw + 1, (tensor.size(0),),
							  device=self.device)

		## i is random number between [0, padding_size*2)

		rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:,None]
		columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:,None]

		## create rows and column indexes from [i to i+48)

		padded = padded.permute(1, 0, 2, 3) ## (N, C, H+2*ps, W+2*ps) --> (C, N, H+2*ps, W+2*ps)
		padded = padded[
						:,
						torch.arange(tensor.size(0))[:, None, None],
						rows[:, torch.arange(th)[:, None]],
						columns[:, None]
						]
		return padded.permute(1, 0, 2, 3) ## (C, N, H, W) --> (N, C, H, W)
