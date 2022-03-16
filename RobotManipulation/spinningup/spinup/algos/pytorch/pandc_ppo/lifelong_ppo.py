from spinup.utils.logx import Logger
import spinup.algos.pytorch.pandc_ppo.core as core
import copy
import robosuite as suite
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, mpi_sum, mpi_argmax, mpi_gather, broadcast, proc_id, num_procs
from spinup.utils.run_utils import setup_logger_kwargs

from itertools import product
import torch
import numpy as np
import scipy.stats
from robosuite.wrappers.gym_wrapper_notflat import GymWrapper
from robosuite.wrappers.normalized_mpi_wrapper import NormalizedMPIWrapper
import os

from spinup.algos.pytorch.pandc_ppo.ppo import ppo
from spinup.algos.pytorch.pandc_ppo.ppo_q import ppo_q
from spinup.algos.pytorch.pandc_ppo.my_bcq import mt_bcq, ReplayBuffer
from spinup.algos.pytorch.pandc_ppo.my_ewc import ewc
import pickle
import math

class StateBuffer:
    """
    A simple FIFO experience replay buffer for BCQ agents.
    """

    def __init__(self, obs_dim, size):
        self.obs_dim = obs_dim
        self.obs_buf = {obs_key: np.zeros(core.combined_shape(size, obs_val), dtype=np.float32) for obs_key, obs_val in obs_dim.items()}
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs):
        for ob_key in obs.keys():
            self.obs_buf[ob_key][self.ptr] = obs[ob_key]
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {k: v[idxs] for k, v in self.obs_buf.items()}
        batch = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

        return batch

class BaseLearner:
    def __init__(
        self,
        obs_dim,
        action_space,
        ppo_kwargs,
        ac_kwargs,
        gamma=0.99,
        results_dir='./tmp/results/',
        experiment_name='',
        seed=None        
    ):
        setup_pytorch_for_mpi()
        self.obs_dim = obs_dim
        self.act_dim = action_space.shape
        self.acmodel = core.MLPActorCritic(obs_dim, action_space, **ac_kwargs)
        sync_params(self.acmodel)
        ppo_kwargs['gamma'] = gamma
        self.ppo_kwargs = ppo_kwargs

        self.depth = len(self.acmodel.num_modules)
        self.results_dir = results_dir
        self.test_reward = {}
        self.test_success = {}
        self.test_sparser = {}
        self.test_env_kwargs = {}
        self.logger_kwargs = setup_logger_kwargs(experiment_name, seed, results_dir)

    def evaluate(
        self,
        test_env,
        *,
        task_id,
        local_num_episodes
    ):      
       
        num_steps = 0
        ret = np.zeros(local_num_episodes)
        success = np.zeros(local_num_episodes)
        steps_at_goal = np.zeros(local_num_episodes)
        for i in range(local_num_episodes):
            obs = test_env.reset()
            obs['obstacle-id'] = np.zeros(self.acmodel.num_modules[0])
            obs['obstacle-id'][self.acmodel.module_assignments[0][task_id]] = 1
            obs['object-id'] = np.zeros(self.acmodel.num_modules[1])
            obs['object-id'][self.acmodel.module_assignments[1][task_id]] = 1
            obs['robot-id'] = np.zeros(self.acmodel.num_modules[2])
            obs['robot-id'][self.acmodel.module_assignments[2][task_id]] = 1
            done = False
            while not done:
                num_steps += 1
                a, _, _ = self.acmodel.step({o_key:  torch.as_tensor(o_val, dtype=torch.float32) for o_key, o_val in obs.items()}, task_id=task_id)
                next_obs, rew, done, _ = test_env.step(a)
                next_obs['obstacle-id'] = np.zeros(self.acmodel.num_modules[0])
                next_obs['obstacle-id'][self.acmodel.module_assignments[0][task_id]] = 1
                next_obs['object-id'] = np.zeros(self.acmodel.num_modules[1])
                next_obs['object-id'][self.acmodel.module_assignments[1][task_id]] = 1
                next_obs['robot-id'] = np.zeros(self.acmodel.num_modules[2])
                next_obs['robot-id'][self.acmodel.module_assignments[2][task_id]] = 1
                ret[i] += rew
                if rew == 1: 
                    success[i] = 1
                    steps_at_goal[i] += 1

                obs = next_obs

        self.test_reward[task_id] = ret.mean()
        self.test_success[task_id] = success.mean()
        self.test_sparser[task_id] = steps_at_goal.mean()

        return num_steps


class LifelongPPO(BaseLearner):
    def __init__(
        self,
        obs_dim,
        action_space,
        ppo_kwargs,
        ac_kwargs,
        update_modules_kwargs,
        gamma=0.99,
        results_dir='./tmp/results/',
        experiment_name='',
        accommodation_replay_capacity=int(1e5),
        seed=None,
        forgetting_method='bcq'
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_space=action_space,
            ppo_kwargs=ppo_kwargs,
            ac_kwargs=ac_kwargs,
            gamma=gamma,
            results_dir=results_dir,
            experiment_name=experiment_name,
            seed=seed     
        )
        if forgetting_method == 'bcq': update_modules_kwargs['gamma'] = gamma
        self.update_modules_kwargs = update_modules_kwargs

        self.T = 0
        self.observed_tasks = set()
        self.replay_buffer = {}
        self.first_state_buffer = {}
        self.local_accommodation_replay_capacity = accommodation_replay_capacity #  with original hyper-parameters, this stores all data (without /num_procs)// num_procs()
        self.test_env_kwargs = {}

        self.accommodation_logger = Logger(**self.logger_kwargs)
        self.ignore_first_row = False
        self.forgetting_method = forgetting_method
        
        self.cumfisher = {}
        self.param_prev = {}

    def add_task(
        self,
        env_kwargs,
        *,
        task_id,
    ):
        self.replay_buffer[task_id] = ReplayBuffer(size=self.local_accommodation_replay_capacity, obs_dim=self.obs_dim, act_dim=self.act_dim)
        self.test_env_kwargs[task_id] = env_kwargs

    def train(
        self,
        env_kwargs,
        *,
        task_id,
        eval_episodes=10
    ):
        self.add_task(
            env_kwargs,
            task_id=task_id,
        )
        steps = 0
        self.acmodel.set_use_kb(False)
        self.acmodel.reset_active()
        self.acmodel.freeze_active(False)
        self.acmodel.freeze_kb(True)

        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1

        logger_kwargs = copy.copy(self.logger_kwargs)
        logger_kwargs['output_dir'] = os.path.join(logger_kwargs['output_dir'], 'task_{}'.format(task_id))

        ppo_kwargs = copy.copy(self.ppo_kwargs)
        ppo_kwargs['logger_kwargs'] = logger_kwargs
        ppo_kwargs['epochs'] -= steps // ppo_kwargs['steps_per_epoch']
        if proc_id() == 0: print(f'Took {steps} steps, so reducing from {self.ppo_kwargs["epochs"]} to {ppo_kwargs["epochs"]} epochs')
       
        # This line is the online (data collection) phase
        if self.acmodel.step_q:
            ppo_q(lambda: GymWrapper(suite.make(**env_kwargs)), self.acmodel, self.replay_buffer[task_id], task_id=task_id, **ppo_kwargs)
        else:   
            ppo(lambda: GymWrapper(suite.make(**env_kwargs)), self.acmodel, self.replay_buffer[task_id], task_id=task_id, **ppo_kwargs)

        self.update_modules(task_id=task_id)
        
        for task in self.observed_tasks:
            self.evaluate(GymWrapper(suite.make(**self.test_env_kwargs[task])), task_id=task, local_num_episodes=max(eval_episodes // num_procs(), 1))
        
        for task in self.observed_tasks:
            rmean = mpi_avg(self.test_reward[task])
            smean = mpi_avg(self.test_success[task])
            sparsermean = mpi_avg(self.test_sparser[task])
            self.accommodation_logger.log_tabular('train task', task_id)
            self.accommodation_logger.log_tabular('task', task)
            self.accommodation_logger.log_tabular('reward_mean', rmean)
            self.accommodation_logger.log_tabular('success_mean', smean)
            self.accommodation_logger.log_tabular('rewardsparse_mean', sparsermean)
            if self.ignore_first_row: self.accommodation_logger.first_row = False       # when loading from ckpt, make sure not to re-write the header
            self.accommodation_logger.dump_tabular()
        
    def update_modules(self, *, task_id):
        if self.forgetting_method == 'bcq':
            self.acmodel.set_use_bcq(task_id=task_id, use_bcq=True)   # TODO: implement proper action selection and add any relevant parts of the model
            self.acmodel.set_use_kb(True)     
            self.acmodel.freeze_active(True)
            self.acmodel.freeze_kb(False)

            accommodation_tasks = list(self.observed_tasks)
            mt_bcq(self.acmodel, accommodation_tasks, self.replay_buffer, **self.update_modules_kwargs)
        else:
            active_model = copy.deepcopy(self.acmodel)  # need to copy so updates to KB don't affect active model
            active_model.set_use_kb(False)     # always use active column
            active_model.freeze_active(True)   # freeze all parameters
            active_model.freeze_kb(True)

            self.acmodel.set_use_kb(True)     
            self.acmodel.freeze_active(True)
            self.acmodel.freeze_kb(False)

            ewc(self.acmodel, active_model, self.replay_buffer[task_id], task_id=task_id, cumfisher=self.cumfisher, param_prev=self.param_prev, **self.update_modules_kwargs) 

    def store_checkpoint(self):
        # this whole function is somewhat redundant, because ppo already stores the model in pyt_save, but this is safer
        if proc_id() == 0:
            fname = self.accommodation_logger.output_file.name
            self.accommodation_logger.output_file.close()
            with open(fname, 'r') as f:
                logger_output = f.read()
            checkpoint = {
                'acmodel_statedict': self.acmodel.state_dict(),
                'acmodel_assignments': self.acmodel.module_assignments,
                'acmodel_usebcq': self.acmodel.use_bcq,
                'test_env_kwargs': self.test_env_kwargs,
                'observed_tasks': self.observed_tasks,      # redundant, but whatever
                'logger_output': logger_output,
                'cumfisher': self.cumfisher,
                'param_prev': self.param_prev,
            }
            torch.save(checkpoint, os.path.join(self.logger_kwargs['output_dir'], 'ckpt.pt'))
            self.accommodation_logger.output_file = open(fname, 'a')    # here I need to start appending rather than writing from scratch

        # store the replay buffer separately so I can save each separate process's buffer
        torch.save(self.replay_buffer, os.path.join(self.logger_kwargs['output_dir'], 'buff_{}.pt'.format(proc_id())))

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(os.path.join(self.logger_kwargs['output_dir'], 'ckpt.pt'))
            self.observed_tasks = checkpoint['observed_tasks']
            for task_id in self.observed_tasks:
                assignments = (checkpoint['acmodel_assignments'][0][task_id], checkpoint['acmodel_assignments'][1][task_id], checkpoint['acmodel_assignments'][2][task_id])
                self.acmodel.set_assignments(assignments, task_id=task_id)
            self.acmodel.load_state_dict(checkpoint['acmodel_statedict'])       # must do this after set_assignments, which creates the logstd for each task_id
            self.acmodel.use_bcq = checkpoint['acmodel_usebcq']
            self.test_env_kwargs = checkpoint['test_env_kwargs']
            if proc_id() == 0:
                logger_output = checkpoint['logger_output']
                self.accommodation_logger.output_file.write(logger_output)
            self.T = len(self.observed_tasks)
            self.ignore_first_row = True
            self.cumfisher = checkpoint['cumfisher']
            self.param_prev = checkpoint['param_prev']

            # load separate buffers to each process, assume num_procs is the same across runs
            if self.forgetting_method == 'bcq':
                self.replay_buffer = torch.load(os.path.join(self.logger_kwargs['output_dir'], 'buff_{}.pt'.format(proc_id())))
        except:
            self.accommodation_logger.log('Warning: could not load checkpoint', color='red')
        return self.T
