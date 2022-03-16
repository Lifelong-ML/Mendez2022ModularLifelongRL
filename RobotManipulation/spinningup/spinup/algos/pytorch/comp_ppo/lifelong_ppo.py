from spinup.utils.logx import Logger
import spinup.algos.pytorch.comp_ppo.core as core
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

from spinup.algos.pytorch.comp_ppo.ppo import ppo
from spinup.algos.pytorch.comp_ppo.ppo_q import ppo_q
from spinup.algos.pytorch.comp_ppo.my_bcq import mt_bcq, ReplayBuffer
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
        local_num_episodes,
        deterministic=False
    ):      
       
        num_steps = 0
        ret = np.zeros(local_num_episodes)
        success = np.zeros(local_num_episodes)
        steps_at_goal = np.zeros(local_num_episodes)
        for i in range(local_num_episodes):
            obs = test_env.reset()
            done = False
            while not done:
                num_steps += 1
                a, _, _ = self.acmodel.step({o_key:  torch.as_tensor(o_val, dtype=torch.float32) for o_key, o_val in obs.items()}, task_id=task_id, deterministic=deterministic)
                next_obs, rew, done, _ = test_env.step(a)
                ret[i] += rew
                if rew == 1: 
                    success[i] = 1
                    steps_at_goal[i] += 1

                obs = next_obs

        self.test_reward[task_id] = ret.mean()
        self.test_success[task_id] = success.mean()
        self.test_sparser[task_id] = steps_at_goal.mean()

        return num_steps

class StlPPO(BaseLearner):
    def train(
        self,
        env_kwargs,
        *,
        task_id,
        eval_episodes=10,
    ):
        logger_kwargs = copy.copy(self.logger_kwargs)
        logger_kwargs['output_dir'] = os.path.join(logger_kwargs['output_dir'], 'task_{}'.format(task_id))

        task_id = 0     # set this _after_ the logging has been set up
        ppo_kwargs = copy.copy(self.ppo_kwargs)
        ppo_kwargs['logger_kwargs'] = logger_kwargs
        module_assignments = tuple(0 for _ in range(len(self.acmodel.num_modules)))   # set modules to the current task idx
        print(task_id, module_assignments)
        self.acmodel.set_assignments(task_id=task_id, module_assignments=module_assignments)     
        # This line is the online (data collection) phase
        ppo(lambda: GymWrapper(suite.make(**env_kwargs)), self.acmodel, task_id=task_id, **ppo_kwargs)

class LifelongPPO(BaseLearner):
    def __init__(
        self,
        obs_dim,
        action_space,
        ppo_kwargs,
        ac_kwargs,
        bcq_kwargs,
        gamma=0.99,
        results_dir='./tmp/results/',
        experiment_name='',
        accommodation_replay_capacity=int(1e5),
        seed=None,
        module_selection_mode='gridsearch',
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
        bcq_kwargs['gamma'] = gamma
        self.bcq_kwargs = bcq_kwargs
        
        self.module_selection_mode = module_selection_mode

        self.T = 0
        self.observed_tasks = set()
        self.replay_buffer = {}
        self.local_accommodation_replay_capacity = accommodation_replay_capacity #  with original hyper-parameters, this stores all data (without /num_procs)// num_procs()
        self.test_env_kwargs = {}

        self.accommodation_logger = Logger(**self.logger_kwargs)
        self.ignore_first_row = False

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

        rescale = False
        if self.T >= self.acmodel.num_modules[0]:
            steps, max_success = self.find_optimal_structure(task_id)
            # train a copy
            self.backup_model = copy.deepcopy(self.acmodel)

            ############## CHECK IF RANDOM IS BETTER ##############
            if proc_id() == 0: print('zero-shot success:', max_success)
            if max_success < 0.1:    
                if proc_id() == 0: print('rescaling outputs')
                rescale = True
            #######################################################
        else:
            module_assignments = tuple(self.T for _ in range(len(self.acmodel.num_modules)))   # set modules to the current task idx
            self.acmodel.set_assignments(task_id=task_id, module_assignments=module_assignments)     
            self.backup_model = self.acmodel   # just a reference assignment
            steps = 0

        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1

        logger_kwargs = copy.copy(self.logger_kwargs)
        logger_kwargs['output_dir'] = os.path.join(logger_kwargs['output_dir'], 'task_{}'.format(task_id))

        ppo_kwargs = copy.copy(self.ppo_kwargs)
        ppo_kwargs['logger_kwargs'] = logger_kwargs
        ppo_kwargs['epochs'] -= steps // ppo_kwargs['steps_per_epoch']
        if proc_id() == 0: print(f'Took {steps} steps, so reducing from {self.ppo_kwargs["epochs"]} to {ppo_kwargs["epochs"]} epochs')
        
        ########## Reset the output head to random initialization but keep the rest ###########
        if rescale:
            for j in self.acmodel.graph_structure[-1]:  # only consider leafs
                mod_asgn = self.acmodel.module_assignments[j][task_id]
                for net in (self.acmodel.pi.mu_net, self.acmodel.qf1.q_net, self.acmodel.qf2.q_net):
                    for p in net._module_list[j]['post_interface'][mod_asgn][-2:].parameters():
                        p.data.mul_(0.01)
            sync_params(self.acmodel)  
        ########################################################################################

        # This line is the online (data collection) phase
        if self.acmodel.step_q:
            ppo_q(lambda: GymWrapper(suite.make(**env_kwargs)), self.acmodel, self.replay_buffer[task_id], task_id=task_id, **ppo_kwargs)
        else:   
            ppo(lambda: GymWrapper(suite.make(**env_kwargs)), self.acmodel, self.replay_buffer[task_id], task_id=task_id, **ppo_kwargs)

        self.acmodel.load_from(self.backup_model)

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
        

    def find_optimal_structure(self, task_id):
        if self.module_selection_mode == 'ground-truth':
            possible_obstacles = [self.test_env_kwargs[task]['obstacle'] for task in range(4)]
            possible_objects = [self.test_env_kwargs[task]['object_type'] for task in range(4)]
            possible_robots = [self.test_env_kwargs[task]['robots'] for task in range(4)]

            true_assignment = (
                possible_obstacles.index(self.test_env_kwargs[task_id]['obstacle']),
                possible_objects.index(self.test_env_kwargs[task_id]['object_type']),
                possible_robots.index(self.test_env_kwargs[task_id]['robots'])
            )
            self.acmodel.set_assignments(true_assignment, task_id=task_id)

            if proc_id() == 0:
                total_num_steps = self.evaluate(GymWrapper(suite.make(**self.test_env_kwargs[task_id])), task_id=task_id, local_num_episodes=10)
                print(true_assignment, np.mean(self.test_reward[task_id]))
                max_success = np.mean(self.test_success[task_id])
                del self.test_reward[task_id]
            else:
                total_num_steps = 0
                max_success = 0

            #### return the optimal success rate
            max_success = np.array(max_success)
            broadcast(max_success, root=0)     # broadcast the max reward of the model that achieved max reward
            max_success = max_success.item()
            ####


            return int(mpi_sum(total_num_steps)), max_success
        elif self.module_selection_mode == 'gridsearch':
            max_reward = -1e6
            total_num_steps = 0
            possible_assignments = list(product(*[range(num_mod) for num_mod in self.acmodel.num_modules]))

            large_partition_size = math.ceil(len(possible_assignments) / num_procs())
            small_partition_size = math.floor(len(possible_assignments) / num_procs())
            n_large = len(possible_assignments) - num_procs() * small_partition_size
            n_small = num_procs() - n_large
            
            if proc_id() < n_large:
                i_0 = proc_id() * (large_partition_size)
                i_f = i_0 + large_partition_size
            else:
                i_0 = n_large * (large_partition_size) + (proc_id() - n_large) * small_partition_size
                i_f = i_0 + small_partition_size
            local_possible_assignments = possible_assignments[i_0 : i_f]    # check that this is correct!!!!!!!!!!!
            print(f'Process {proc_id()} uses {len(local_possible_assignments)} assignments (from {i_0} to {i_f})')

            for module_assignments in local_possible_assignments:
                self.acmodel.set_assignments(module_assignments, task_id=task_id)
                total_num_steps += self.evaluate(GymWrapper(suite.make(**self.test_env_kwargs[task_id])), task_id=task_id, local_num_episodes=10)
                print(module_assignments, np.mean(self.test_reward[task_id]))
                if np.mean(self.test_reward[task_id]) > max_reward:
                    max_reward = np.mean(self.test_reward[task_id])
                    max_success = np.mean(self.test_success[task_id])
                    max_assignments = module_assignments

            max_proc_id = mpi_argmax(max_reward)
            max_assignments = np.array(max_assignments)
            broadcast(max_assignments, root=max_proc_id)      # if I'm not mistaken, this sends the value max_assignments of max_proc_id to all other processes and sets their max_assignemnts variables to this value. I'm uncertain about the last part.
            max_assignments = tuple(max_assignments)

            #### return the optimal success rate
            max_success = np.array(max_success)
            broadcast(max_success, root=max_proc_id)     # broadcast the max reward of the model that achieved max reward
            max_success = max_success.item()
            ####

            # Set model structure to the optimal one
            self.acmodel.set_assignments(max_assignments, task_id=task_id)
            del self.test_reward[task_id]

            return int(mpi_sum(total_num_steps)), max_success

    def update_modules(self, *, task_id):
        self.acmodel.set_use_bcq(task_id=task_id, use_bcq=True)  
        accommodation_tasks = self.filter_accommodation_tasks(task_id=task_id)
        mt_bcq(self.acmodel, accommodation_tasks, self.replay_buffer, **self.bcq_kwargs)

    def filter_accommodation_tasks(self, *, task_id):
        return list(self.observed_tasks)    # all seen tasks

    def store_checkpoint(self):
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

            # load separate buffers to each process, assume num_procs is the same across runs
            self.replay_buffer = torch.load(os.path.join(self.logger_kwargs['output_dir'], 'buff_{}.pt'.format(proc_id())))
        except:
            self.accommodation_logger.log('Warning: could not load checkpoint', color='red')
        return self.T
