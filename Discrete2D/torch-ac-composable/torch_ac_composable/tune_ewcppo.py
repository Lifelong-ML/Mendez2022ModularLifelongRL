import subprocess
import os
import sys
import torch
from itertools import cycle
import numpy as np

num_seeds = 5
gpu_use = 30
cpu_use = 800

num_gpus = torch.cuda.device_count()
num_cpus = os.cpu_count()

algo = 'ewc-ppo'
learning_rate = 1e-3
steps_per_proc = 256
batch_size = 256
procs = 16
num_tasks = 5
num_steps = 1000000
max_modules = 4
minigrid_size = 8

# possible_lambdas = [1e-3, 1e-2, 1e-1, 1e0]
possible_lambdas = [1e1, 1e2, 1e3, 1e4]

experiment_name = 'MiniGrid_1M_H64_EWC_PPO_tune'

gpu_use_total = np.zeros(num_gpus)
cpu_use_total = 0
cuda_device_dict = {}
process_gpu_use = {}
for seed in range(num_seeds):
    for ewc_lambda in possible_lambdas:
        while np.all(gpu_use_total + gpu_use > 100) or cpu_use_total > 100 * num_cpus:
            for p in cycle(process_gpu_use):
                try:
                    p.wait(1)
                    gpu_use_remove = process_gpu_use[p]
                    gpu_use_total[cuda_device_dict[p]] -= gpu_use_remove
                    cpu_use_total -= cpu_use
                    del process_gpu_use[p]
                    del cuda_device_dict[p]
                    break
                except subprocess.TimeoutExpired:
                    pass

        cuda_device = np.argmin(gpu_use_total)
        job_name = 'tune_' + experiment_name + '_seed_{}' + '_lambda_{}' 
        results_dir = os.path.join('..', '..', 'pt', experiment_name, job_name.format(seed, ewc_lambda), 'results')

        my_env = os.environ.copy()
        my_env['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        args = ['python', '-m', 'experiments.ppo_minigrid_lifelong',
                '--algo', algo,
                '--ewc-lambda', str(ewc_lambda),
                '--learning-rate', str(learning_rate),
                '--steps-per-proc', str(steps_per_proc),
                '--batch-size', str(batch_size),
                '--procs', str(procs),
                '--num-tasks', str(num_tasks),
                '--num-steps', str(num_steps),
                '--max-modules', str(max_modules),
                '--minigrid-size', str(minigrid_size),
                '--results-dir', str(results_dir),
                '--seed', str(seed)
        ]
        p = subprocess.Popen(args, env=my_env)
        process_gpu_use[p] = gpu_use
        gpu_use_total[cuda_device] += gpu_use
        cuda_device_dict[p] = cuda_device
        cpu_use_total += cpu_use



