import roboverse
import numpy as np
from matplotlib import pyplot as plt
import time

class RoboverseSim():
    def __init__(self, env_id):
        self._env = roboverse.make(env_id, transpose_image=True)
        self._agent = None

    def setAgent(self, agent):
        self._agent = agent

    def collectPaths(self, max_path_length=40, num_eval_steps=1, discard_incomplete_paths=True):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_eval_steps:
            max_path_length_this_loop = min(  # Do not go over num_eval_step
                max_path_length,
                num_eval_steps - num_steps_collected,
            )
            path = self.rollout(
                self._env,
                self._agent,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)

        return paths

    def rollout(self, env, agent, max_path_length):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        path_length = 0

        o = env.reset()
        next_o = None

        while path_length < max_path_length:

            # print(o['image'].shape, o['image'].min())

            a = agent.get_action(o['image'])
            next_o, r, d, env_info = env.step(a)
            # time.sleep(0.2)

            # ## displaying image
            # img_disp = o['image'].reshape(3, 48, 48).transpose(1, 2, 0)
            # plt.imshow(img_disp)
            # plt.show()
            
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            next_observations.append(next_o)
            terminals.append(d)
            path_length += 1
            if d:
                break
            o = next_o
        
        return dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards).reshape(-1, 1),
            terminals=np.array(terminals).reshape(-1, 1),
            next_observations = np.array(next_observations)
        )

    