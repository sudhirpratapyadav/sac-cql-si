# sac-cql-si
This is official code repository of the work: Yadav, Sudhir Pratap, Rajendra Nagar, and Suril V. Shah. "Learning Vision-based Robotic Manipulation Tasks Sequentially in Offline Reinforcement Learning Settings." arXiv preprint arXiv:2301.13450 (2023).

Preprint: https://arxiv.org/abs/2301.13450

This works implements sequential (or continual) learning in offline RL settings. We use SI (Synaptic Intelligence) regularisation based method for continual learning while SAC-CQL method for offline RL.

# Requirements
1. pip3 install torch pybullet==3.2.1 matplotlib gtimer gym==0.21.0
2. Install roboverse (https://github.com/sudhirpratapyadav/roboverse)

# Usage
Example command
`python3 main.py --dgx --dataset-file datasets_density_10.yaml --area 0.32 --noise 0.1 --task-seq shed_button -c 1`

Here '--dgx' is required for all experiments. We will update the code soon so this command won't be necessary. Ignore it for the moment.

--dataset-file: path of the dataset file, see the example file provided.
--area, --noise : arguments to select dataset from the dataset file
--task-seq : seq of tasks to be trained separated by '_'. Currenlty 3 tasks are there button, shed, drawer (choose 2/3 tasks in any order)
-c : Regularisation strength parameter in SI paper

# References

1. SI paper: https://arxiv.org/abs/1703.04200?context=cs
2. SAC-CQL paper: https://arxiv.org/abs/2006.04779

