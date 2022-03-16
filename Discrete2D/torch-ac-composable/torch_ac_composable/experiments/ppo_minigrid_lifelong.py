import torch
from models.acmodel import ACModel
from models.acmodel_modular_fixed import ACModelModularFixed
from models.acmodel_nonmodular_fixed import ACModelNonModularFixed
from models.acmodel_bcq_progresscompress import ACModelBcqProgressCompress
from models.acmodel_progresscompress import ACModelProgressCompress

from algos.compositional_er_ppo_bcq import CompositionalErPpoBcq
from algos.monolithic_pandc_ewc_ppo import MonolithicPandCPPO
from algos.monolithic_pandc_bcq_ppo import MonolithicPandCBcqPPO
from algos.agent_wrappers import EWCLearner
from algos.agent_wrappers import CLEARLearner
from algos.agent_wrappers import SingleTaskLearner
from algos.agent_wrappers import MultiTaskLearner

import gym
import argparse
import os

from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gym_minigrid.comp_wrappers import ChanneledWrapper

import pickle

from itertools import product
import numpy as np

import utils

parser = argparse.ArgumentParser(description='Lifelong ppo')
parser.add_argument('--algo', type=str, default='comp-bcq', help='Lifelong RL method',
                    choices=['comp-ppo', 'stl-ppo', 'mtl-ppo', 'pandc-ppo', 'ewc-ppo', 'clear-ppo', 'pandc-bcq-ppo'])
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='lr', help='Adam learning rate')
parser.add_argument('--steps-per-proc', type=int, default=128, help='Number of frames per process before update')
parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training PPO')
parser.add_argument('--replay-batch-size', type=int, default=256, help='Replay batch size for training CLEAR')
parser.add_argument('--procs', type=int, default=16, help='Number of processes')
parser.add_argument('--num-tasks', type=int, default=10, help='Number of tasks to train the agent on')
parser.add_argument('--num-steps', type=int, default=100000, help='Number of total steps per task')
parser.add_argument('--max-modules', type=int, default=4, help='Max number of modules')
parser.add_argument('--task-id', type=int, default=None, help='Which task to train, only for STL methods')
parser.add_argument('--minigrid-size', type=int, default=8, help='Size of the grid for MiniGrid')
parser.add_argument('--results-dir', type=str, default='./tmp/results/', help='/path/to/results/files')
parser.add_argument('--eval-episodes', type=int, default=10, help='Number of episodes for each evaluation')
parser.add_argument('--fixed-structure', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, help='Whether to use a manually fixed structure for compositional methods')
parser.add_argument('--modular', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, help='Whether to use modular model for MTL and CLEAR')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ewc-lambda', type=float, default=100, help='EWC lambda hyper-parameter')

args = parser.parse_args()

algo = args.algo
learning_rate = args.learning_rate
steps_per_proc = args.steps_per_proc
procs = args.procs
num_tasks = args.num_tasks
num_steps = args.num_steps
max_modules = args.max_modules
results_dir = args.results_dir
eval_episodes = args.eval_episodes
seed = args.seed
minigrid_size = args.minigrid_size
batch_size = args.batch_size
replay_batch_size = args.replay_batch_size
task_id_stl = args.task_id
modular = args.modular
fixed_structure = args.fixed_structure

results_dir = os.path.join(results_dir, 'MiniGrid-{0}x{0}'.format(minigrid_size), algo, 'seed_{}'.format(seed))
torch.manual_seed(seed)
np.random.seed(seed)

agents = range(4)
static_objects = ['floor', 'wall', 'lava', 'food']
target_objects = ['red', 'green', 'blue', 'purple']

all_combinations = list(product(agents, static_objects, target_objects))
all_combinations = np.array(all_combinations)

task_order = np.random.permutation(all_combinations)

initialized_agent = set()
initialized_static = set()
initialized_target = set()
num_initialized = 0
for task_id in range(len(task_order)):
    task_combination = tuple(task_order[task_id])
    if (task_combination[0] not in initialized_agent and 
            task_combination[1] not in initialized_static and 
            task_combination[2] not in initialized_target):
        print(task_combination)
        task_order[[num_initialized, task_id]] = task_order[[task_id, num_initialized]]
        initialized_agent.add(task_combination[0])
        initialized_static.add(task_combination[1])
        initialized_target.add(task_combination[2])
        num_initialized += 1
        if num_initialized == max_modules:
            break
task_order = task_order[:num_tasks]

train_env_list = []
test_env_list = []
env_dict = {}
agent_dyn_per_task = []
static_object_per_task = []
target_object_per_task = []
for task_id, task_combination in enumerate(task_order):
    env_dict[task_id] = task_combination
    agent_dyn_per_task.append(int(task_combination[0]))
    static_object_per_task.append(static_objects.index(task_combination[1]))
    target_object_per_task.append(target_objects.index(task_combination[2]))

    env_id = 'MiniGrid-FewShot-{0[0]}-{0[1]}-{0[2]}-{1}x{1}-v0'.format(task_combination, minigrid_size)
    print(env_id)

    train_env = []
    for i in range(procs):
        env = utils.make_env(env_id)
        env = ChanneledWrapper(env)
        train_env.append(env)
    train_env_list.append(train_env)

    test_env = []
    for i in range(procs):
        env = utils.make_env(env_id)
        env = ChanneledWrapper(env)
        test_env.append(env)
    test_env_list.append(test_env)

obs_space, preprocess_obss = utils.get_obss_preprocessor(train_env_list[0][0].observation_space)


os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'env_dict.pickle'), 'wb') as f:
    pickle.dump(env_dict, f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'stl' in algo:
    model = ACModel(obs_space['image'], train_env_list[0][0].action_space.n, device=device)
elif 'mtl' in algo:
    if modular:
        model = ACModelModularFixed(obs_space['image'], train_env_list[0][0].action_space.n, agent_dyn_per_task, static_object_per_task, target_object_per_task, max_modules=max_modules, device=device)
    else:
        model = ACModelNonModularFixed(obs_space['image'], train_env_list[0][0].action_space.n, agent_dyn_per_task, static_object_per_task, target_object_per_task, max_modules=max_modules, device=device)
elif 'pandc' in algo:
    if 'bcq' in algo:
        model = ACModelBcqProgressCompress(obs_space['image'], train_env_list[0][0].action_space.n, agent_dyn_per_task, static_object_per_task, target_object_per_task, max_modules=max_modules, device=device)
    else:
        model = ACModelProgressCompress(obs_space['image'], train_env_list[0][0].action_space.n, agent_dyn_per_task, static_object_per_task, target_object_per_task, max_modules=max_modules, device=device)
elif 'ewc' in algo:
    model = ACModelNonModularFixed(obs_space['image'], train_env_list[0][0].action_space.n, agent_dyn_per_task, static_object_per_task, target_object_per_task, max_modules=max_modules, device=device)
elif 'clear' in algo:
    if modular:
        model = ACModelModularFixed(obs_space['image'], train_env_list[0][0].action_space.n, agent_dyn_per_task, static_object_per_task, target_object_per_task, max_modules=max_modules, device=device)
    else:
        model = ACModelNonModularFixed(obs_space['image'], train_env_list[0][0].action_space.n, agent_dyn_per_task, static_object_per_task, target_object_per_task, max_modules=max_modules, device=device)
elif 'comp' in algo:
    if fixed_structure:
        agent_dyn_per_task = {i: x for i, x in enumerate(agent_dyn_per_task)}
        static_object_per_task = {i: x for i, x in enumerate(static_object_per_task)}
        target_object_per_task = {i: x for i, x in enumerate(target_object_per_task)}
        model = ACModelModularFixed(obs_space['image'], train_env_list[0][0].action_space.n, agent_dyn_per_task, static_object_per_task, target_object_per_task, max_modules=max_modules, device=device)
    else:
        model = ACModelModularFixed(obs_space['image'], train_env_list[0][0].action_space.n, {}, {}, {}, max_modules=max_modules, device=device)
else:
    raise NotImplementedError('Only base methods are PPO and BCQ')

learner_args = (model, )
learner_kwargs = {
    'num_steps': num_steps,
    'steps_per_proc': steps_per_proc,
    'num_procs': procs,
    'gamma': 0.99,
    'learning_rate': learning_rate,
    'gae_lambda': 0.95,
    'entropy_coef': 1e-2,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5,
    'recurrence': 1,
    'adam_eps': 1e-8,
    'clip_eps': 0.2,
    'epochs' : 4,
    'batch_size': batch_size,
    'preprocess_obss': preprocess_obss,
    'reshape_reward': None,
    'results_dir': results_dir
}

if algo == 'comp-ppo':
    learner = CompositionalErPpoBcq(*learner_args, **learner_kwargs)
elif algo == 'pandc-ppo':
    learner_kwargs['ewc_lambda'] = args.ewc_lambda
    learner_kwargs['ewc_gamma'] = 1.
    learner = MonolithicPandCPPO(*learner_args, **learner_kwargs)
elif algo == 'pandc-bcq-ppo':
    learner = MonolithicPandCBcqPPO(*learner_args, **learner_kwargs)
elif algo == 'ewc-ppo':
    learner_kwargs['ewc_lambda'] = args.ewc_lambda
    learner_kwargs['ewc_gamma'] = 1.
    learner = EWCLearner(*learner_args, **learner_kwargs) 
elif algo == 'clear-ppo':
    learner_kwargs['replay_batch_size'] = replay_batch_size
    learner = CLEARLearner(*learner_args, **learner_kwargs)
elif algo =='stl-ppo':
    learner = SingleTaskLearner(*learner_args, **learner_kwargs)
elif algo == 'mtl-ppo':
    learner = MultiTaskLearner(*learner_args, **learner_kwargs)

if 'stl' in algo:
    learner.train(
        train_env=train_env_list[task_id_stl],
        task_id=task_id_stl,
        test_env=test_env_list[task_id_stl],
        log_freq=1,
        eval_episodes=eval_episodes
    )
elif 'mtl' in algo:
    learner.train(
        train_envs = train_env_list,
        task_ids = np.arange(len(train_env_list)),
        test_envs=test_env_list,
        log_freq=1,
        eval_episodes=eval_episodes
    )
else:
    for task_id, (train_env, test_env) in enumerate(zip(train_env_list, test_env_list)):
        learner.train(
            train_env=train_env,
            task_id=task_id,
            test_env=test_env,
            log_freq=1,
            eval_episodes=eval_episodes,
        )




