import argparse

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers.gym_wrapper_notflat import GymWrapper

from spinup.algos.pytorch.comp_ppo.lifelong_ppo import LifelongPPO, StlPPO
from spinup.algos.pytorch.pandc_ppo.lifelong_ppo import LifelongPPO as PandCLifelongPPO
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, num_procs

import numpy as np
import torch
from itertools import product
import os
import sys
import pickle
from spinup.utils.logx import Logger    # only to store the env_dict at the end

import gym
gym.logger.set_level(40)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--seed', '-s', type=int, default=4)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--pi-lr', type=float, default=1e-3)
    parser.add_argument('--vf-lr', type=float, default=1e-3)
    parser.add_argument('--pi-iters', type=int, default=80)
    parser.add_argument('--vf-iters', type=int, default=80)

    parser.add_argument('--num-tasks', type=int, default=10, help='Number of tasks to train the agent on')
    parser.add_argument('--task-id', type=int, default=None, help='Which task to train, only for STL methods')
    parser.add_argument('--results-dir', type=str, default='./tmp/results/', help='/path/to/results/files')

    parser.add_argument('--algo', type=str, default='comp-ppo', choices=['comp-ppo', 'pandc-ppo', 'stl-ppo'])

    parser.add_argument('--load-checkpoint', action='store_true')
    parser.add_argument('--slurm-job', action='store_true')         # used to choose whether to exit after every task or not
    parser.add_argument('--ewc-lambda', type=float, default=1)
    parser.add_argument('--fixed-assignments', action='store_true', help='Whether the compositional method uses a manually fixed module assignment')
    parser.add_argument('--forgetting-method', type=str, default='bcq', help='Whether to use BCQ or EWC for P&C')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)   
    torch.manual_seed(args.seed)

    mpi_fork(args.cpu)  # run parallel code with mpi

    ac_kwargs = {
        'hidden_sizes': ((32,), (32, 32), (64, 64)), 
        'module_inputs': ['obstacle-state', 'object-state', ['goal-state', 'robot0_proprio-state']],
        'interface_depths': [-1, 1, 2] ,   
        'graph_structure': [[0], [1], [2]],
        'step_q': True
    }

    ppo_kwargs = {
        'seed': args.seed,
        'steps_per_epoch': args.steps,
        'epochs': args.epochs,
        'clip_ratio': args.clip,
        'pi_lr': args.pi_lr,
        'vf_lr': args.vf_lr,
        'train_pi_iters': args.pi_iters,
        'train_v_iters': args.vf_iters,
        'max_ep_len': 500,
        'target_kl': 0.05
    }


    bcq_kwargs = {
        'buffer_batch_size': 1200,  # I need a huge batch size to be able to run quickly
        'num_epochs': 100
    }


    #########
    dummy_env_kwargs = {
        'env_name': 'CompositionalEnv',
        'task': 'Lift',
        'object_type': 'milk',
        'obstacle': None,
        'robots': 'Panda',
        'controller_configs': load_controller_config(default_controller='JOINT_POSITION'),
        'has_renderer': False,
        'has_offscreen_renderer': False,
        'reward_shaping': True,
        'ignore_done': True,
        'use_camera_obs': False,
        'control_freq': 20,
        'horizon': 500
    }
    env = GymWrapper(suite.make(**dummy_env_kwargs))
    obs_dim = env.obs_dim
    action_space = env.action_space
    #########

    robots = ["IIWA", "Kinova3", "Panda", "Sawyer"]
    obstacles = ["None", "ObjectDoor", "ObjectWall"]
    objects = ["can", "milk", "bread", "cereal"]
    tasks = ["Lift"]
    all_combinations = np.array(list(product(tasks, objects, obstacles, robots)))
    task_order = np.random.permutation(all_combinations)

    initialized_obstacle = set()
    initialized_object = set()
    initialized_robot = set()
    num_initialized = 0
    for task_id in range(len(task_order)):
        task_combination = tuple(task_order[task_id])
        if (task_combination[1] not in initialized_object and
            (task_combination[2] not in initialized_obstacle or num_initialized == len(obstacles)) and    # treat obstacle specially because it's only 3, but could generalize to others too
            task_combination[3] not in initialized_robot
        ):  
            task_order[[num_initialized, task_id]] = task_order[[task_id, num_initialized]]
            initialized_object.add(task_combination[1])
            initialized_obstacle.add(task_combination[2])
            initialized_robot.add(task_combination[3])
            num_initialized += 1
            if num_initialized == max(len(robots), len(obstacles), len(objects)):
                break
    task_order = task_order[:args.num_tasks]

    env_dict = {}
    env_kwargs = {}
    gt_module_assignments = [{} for _ in range(3)]  # obstacles, objects, robots
    for task_id, task_combination in enumerate(task_order):
        if proc_id() == 0: print(task_id, task_combination)
        env_dict[task_id] = task_combination

        env_kwargs[task_id] = {
            'env_name': 'CompositionalEnv',
            'task': task_combination[0],
            'object_type': task_combination[1],
            'obstacle': task_combination[2] if task_combination[2] != "None" else None,
            'robots': task_combination[3],
            'controller_configs': load_controller_config(default_controller='JOINT_POSITION'),
            'has_renderer': False,
            'has_offscreen_renderer': False,
            'reward_shaping': True,
            'ignore_done': True,
            'use_camera_obs': False,
            'control_freq': 20,
            'horizon': 500,
            'ignore_done': False
        }
        gt_module_assignments[0][task_id] = obstacles.index(task_combination[2])
        gt_module_assignments[1][task_id] = objects.index(task_combination[1])
        gt_module_assignments[2][task_id] = robots.index(task_combination[3])


    if args.algo == 'stl-ppo':
        experiment_name = args.algo
        ac_kwargs['num_modules'] = (1, 1, 1)
        ac_kwargs['step_q'] = False     
        lifelong_learner = StlPPO(
            obs_dim,
            action_space,
            gamma=args.gamma,
            ppo_kwargs=ppo_kwargs,
            ac_kwargs=ac_kwargs,
            results_dir=args.results_dir,
            experiment_name=experiment_name,
            seed=args.seed
        )
        lifelong_learner.train(
            env_kwargs[args.task_id],
            task_id=args.task_id     # for logging
        )
    elif args.algo == 'comp-ppo':
        ac_kwargs['num_modules'] = (4, 4, 4)    # obs, obj, rob
        if args.fixed_assignments:
            ac_kwargs['module_assignments'] = gt_module_assignments
            module_selection_mode = 'ground-truth'
        else:
            module_selection_mode = 'gridsearch'
        experiment_name = args.algo
        lifelong_learner = LifelongPPO(
            obs_dim,
            action_space,
            gamma=args.gamma,
            ppo_kwargs=ppo_kwargs,
            ac_kwargs=ac_kwargs,
            bcq_kwargs=bcq_kwargs,
            results_dir=args.results_dir,
            experiment_name=experiment_name,
            seed=args.seed,
            accommodation_replay_capacity=int(100000 // num_procs()),
            module_selection_mode=module_selection_mode,
        )

        if args.load_checkpoint:
            task_0 = lifelong_learner.load_checkpoint()
        else:
            task_0 = 0
        for task_id in range(task_0, len(task_order)):
            lifelong_learner.train(
                env_kwargs[task_id],
                task_id=task_id
            )
            lifelong_learner.store_checkpoint()
            if args.slurm_job and task_id < len(task_order) - 1:    # tell slurm to requeue
                mpi_avg(np.random.rand(1))      # wait for store_checkpoint to return in all processes before exiting
                sys.exit(3) 

    elif args.algo == 'pandc-ppo':
        ac_kwargs['num_modules'] = (3, 4, 4)    # obs, obj, rob
        ac_kwargs['module_assignments'] = gt_module_assignments
        ac_kwargs['module_inputs'][0] = [ ac_kwargs['module_inputs'][0], 'obstacle-id']
        ac_kwargs['module_inputs'][1] = [ ac_kwargs['module_inputs'][1], 'object-id']
        ac_kwargs['module_inputs'][2].append('robot-id')

        obs_dim['obstacle-id'] = 3
        obs_dim['object-id'] = 4
        obs_dim['robot-id'] = 4

        # bcq_kwargs['ewc_lambda'] = args.ewc_lambda
        # bcq_kwargs['batch_size_fisher'] = 100 * bcq_kwargs['buffer_batch_size']

        experiment_name = args.algo
        lifelong_learner = PandCLifelongPPO(
            obs_dim,
            action_space,
            gamma=args.gamma,
            ppo_kwargs=ppo_kwargs,
            ac_kwargs=ac_kwargs,
            update_modules_kwargs=bcq_kwargs,
            results_dir=args.results_dir,
            experiment_name=experiment_name,
            seed=args.seed,
            accommodation_replay_capacity=int(100000 // num_procs()),
            forgetting_method=args.forgetting_method
        )

        if args.load_checkpoint:
            task_0 = lifelong_learner.load_checkpoint()
        else:
            task_0 = 0
        for task_id in range(task_0, len(task_order)):
            lifelong_learner.train(
                env_kwargs[task_id],
                task_id=task_id
            )
            lifelong_learner.store_checkpoint()
            if args.slurm_job and task_id < len(task_order) - 1:    # tell slurm to requeue
                mpi_avg(np.random.rand(1))      # wait for store_checkpoint to return in all processes before exiting
                sys.exit(3) 


    lifelong_learner.logger_kwargs['output_fname'] = 'dummy.txt'
    logger = Logger(**lifelong_learner.logger_kwargs)
    logger.setup_pytorch_saver(lifelong_learner.acmodel)
    logger.save_state(env_dict, fname='env_dict.pickle')

if __name__ == '__main__':
    main()
