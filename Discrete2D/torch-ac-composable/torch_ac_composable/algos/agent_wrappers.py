import torch
import os
import time
from algos.ppo import PPOAlgo
from algos.ppo_ewc import EwcPPOAlgo
from algos.ppo_vtrace import PPOvTraceAlgo
import utils
import numpy as np
from torch_ac_composable.utils import DictList, ParallelEnv

from torch.distributions.kl import kl_divergence
import copy

from itertools import product

import pickle

class Learner():
    def __init__(
        self,
        acmodel,
        num_steps,
        num_procs,
        steps_per_proc,
        gamma=0.99,
        learning_rate=1e-3,
        gae_lambda=0.95,
        entropy_coef=1e-2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        replay_capacity=100000,
        results_dir='./tmp/results/'
    ):
        self.depth = 3      # TODO: hardcoding architecture
        self.acmodel = acmodel
        self.num_steps = num_steps
        self.agent = PPOAlgo(
            acmodel,
            num_procs=num_procs,
            num_frames_per_proc=steps_per_proc,
            discount=gamma,
            lr=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward
        )
        self.results_dir = results_dir
        self.test_reward = {}
        self.T = 0
        self.observed_tasks = set()
        self.txt_logger = utils.get_txt_logger(self.results_dir)
        self.csv_assimilation_file = {}
        self.csv_assimilation_logger = {}
        self.csv_accommodation_file, self.csv_accommodation_logger = utils.get_csv_logger(self.results_dir)
        self.replay_buffer = {}
        self.replay_capacity = replay_capacity
        os.makedirs(self.results_dir, exist_ok=True)

    def add_task(self, env, task_id):
        self.agent.add_task(env, task_id)
        results_dir_task = os.path.join(self.results_dir, 'task_{}'.format(task_id))
        self.csv_assimilation_file[task_id], self.csv_assimilation_logger[task_id] = utils.get_csv_logger(results_dir_task)
        self.replay_buffer[task_id] = utils.ReplayBufferTensors(self.replay_capacity, self.acmodel.input_shape)

    def train(self, *args, **kwargs):
        raise NotImplementedError('Training loop is algorithm specific')

    def evaluate(self, task_id, env, num_episodes):
        start_time = time.time()
        returns = np.zeros(num_episodes)
        num_steps = 0

        assert num_episodes <= len(env), "Implemented assuming num_episodes >= num_procs"
        env = ParallelEnv(env[:num_episodes])

        obs = env.reset()
        mask = np.ones_like(returns)
        num_steps = np.zeros_like(returns)
        while np.any(mask):
            num_steps += np.ones_like(num_steps) * mask
            preprocessed_obs = self.agent.preprocess_obss(obs, device=self.acmodel.device)
            
            if hasattr(self.acmodel, 'use_bcq') and task_id in self.acmodel.use_bcq and self.acmodel.use_bcq[task_id]:
                action = self.acmodel.act(preprocessed_obs, 0.05, task_id)
                dist, value = self.acmodel(preprocessed_obs, task_id)
            else:
                with torch.no_grad():
                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                    else:
                        dist, value = self.acmodel(preprocessed_obs, task_id)#, verbose=(num_steps==np.ones_like(num_steps)).all())
                action = dist.sample()
            obs, reward, done, _ = env.step(action.cpu().numpy())
            returns += np.array(reward) * mask
            mask[np.array(done)] = 0

        self.test_reward[task_id] = utils.synthesize(returns)
        env.terminate()

        return num_steps.sum()

    def save_data(self, updates, steps, task_id, save_logs=False, final=False):
        if final:
            path = os.path.join(self.results_dir, 'policynet.pt')
            torch.save(self.acmodel.state_dict(), path)
            if hasattr(self.acmodel, 'static_object_dict'):
                with open(os.path.join(self.results_dir, 'selection_gridsearch.pickle'), 'wb') as f:
                    pickle.dump({'static_object_dict': self.acmodel.static_object_dict,
                                'target_object_dict': self.acmodel.target_object_dict,
                                'agent_dyn_dict': self.acmodel.agent_dyn_dict}, f)

        if save_logs:
            if final:
                first_row = self.T == 1
                for task in self.observed_tasks:
                    self.txt_logger.info('task {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f}'.format(task, *self.test_reward[task].values()) )
                    header = ["return_" + key for key in self.test_reward[task].keys()]
                    data = self.test_reward[task].values()
                    data = [task_id, task] + list(data)
                    if first_row:
                        header = ['train task', 'task'] + header
                        self.csv_accommodation_logger.writerow(header)
                        first_row = False
                    self.csv_accommodation_logger.writerow(data)
            else:
                fps = self.logs["num_frames"] / (self.update_end_time - self.update_start_time)
                duration = int(time.time() - self.start_time)
                return_per_episode = utils.synthesize(self.logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(self.logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(self.logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [updates, steps, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [self.logs["entropy"], self.logs["value"], self.logs["policy_loss"], self.logs["value_loss"], self.logs["grad_norm"]]

                self.txt_logger.info(
                    "task {} | U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                    .format(task_id, *data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if updates == 1:
                    self.csv_assimilation_logger[task_id].writerow(header)
                self.csv_assimilation_logger[task_id].writerow(data)

class SingleTaskLearner(Learner):
    def __init__(
        self,
        acmodel,
        num_steps,
        num_procs,
        steps_per_proc,
        gamma=0.99,
        learning_rate=1e-3,
        gae_lambda=0.95,
        entropy_coef=1e-2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        replay_capacity=100000,
        results_dir='./tmp/results/'
    ):
        super().__init__(
            acmodel,
            num_steps,
            num_procs,
            steps_per_proc,
            gamma=gamma,
            learning_rate=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward,
            replay_capacity=replay_capacity,
            results_dir=results_dir
        )

    def train(
        self,
        train_env,
        task_id,
        test_env,
        eval_episodes=10,
        log_freq=1e4
    ):
        self.txt_logger = utils.get_txt_logger(self.results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        self.T = 1
        self.observed_tasks = set([task_id])
        self.add_task(train_env, task_id)
        
        if log_freq == -1:
            log_freq = np.inf

        log_bool = not np.isinf(log_freq)

        updates = 0
        steps = 0
        self.start_time = time.time()
        while steps < self.num_steps:

            self.update_start_time = time.time()
            exps, logs1 = self.agent.collect_experiences(task_id)
            self.replay_buffer[task_id].push(exps)
            logs2 = self.agent.update_parameters(exps, task_id)
            self.logs = {**logs1, **logs2}
            self.update_end_time = time.time()

            steps += self.logs['num_frames']
            updates += 1

            if updates % log_freq == 0 or steps >= self.num_steps:
                self.save_data(updates, steps, task_id, save_logs=log_bool)

        self.evaluate(task_id, test_env, eval_episodes)
        # self.save_data(updates + 1, steps, task_id, save_logs=False, final=True)
        self.save_data(updates + 1, steps, task_id, save_logs=True, final=True)
        self.csv_assimilation_file[task_id].flush()

class MultiTaskLearner(Learner):
    def __init__(
        self,
        acmodel,
        num_steps,
        num_procs,
        steps_per_proc,
        gamma=0.99,
        learning_rate=1e-3,
        gae_lambda=0.95,
        entropy_coef=1e-2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        replay_capacity=100000,
        results_dir='./tmp/results/'
    ):
        super().__init__(
            acmodel,
            num_steps,
            num_procs,
            steps_per_proc,
            gamma=gamma,
            learning_rate=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward,
            replay_capacity=replay_capacity,
            results_dir=results_dir
        )

    def save_data(self, updates, steps, task_ids, save_logs=False, final=False):
        if not final:
            logs1_list, logs2 = self.logs
            for task, logs1 in zip(task_ids, logs1_list):
                self.logs = {**logs1, **logs2}
                super().save_data(updates, steps, task, save_logs, final)

            if save_logs:
                mean_rreturn_per_episode = 0.
                mean_num_frames_per_episode = 0.
                for task, logs1 in zip(task_ids, logs1_list):
                    mean_rreturn_per_episode += utils.synthesize(logs1["reshaped_return_per_episode"])['mean']
                    mean_num_frames_per_episode += utils.synthesize(logs1["num_frames_per_episode"])['mean']

                mean_rreturn_per_episode /= len(task_ids)
                mean_num_frames_per_episode /= len(task_ids)
                data = [updates, steps, mean_rreturn_per_episode, mean_num_frames_per_episode]

                self.txt_logger.info(
                    "MTL | U {} | F {:06} | rR:μ {:.2f} | F:μ {:.1f}"
                    .format(*data))
        else:
            super().save_data(updates, steps, task_ids, save_logs, final)
        

    def train(
        self,
        train_envs,
        task_ids,
        test_envs,
        eval_episodes=10,
        log_freq=1e4
    ):
        if log_freq == -1:
            log_freq = np.inf
        self.observed_tasks = set(task_ids)
        self.T = len(self.observed_tasks)

        log_bool = not np.isinf(log_freq)

        updates = 0
        steps = 0
        self.start_time = time.time()
        for task in task_ids:
            self.add_task(train_envs[task], task)
        
        while steps < self.num_steps:
            exps_list = []
            logs1_list = []
            self.update_start_time = time.time()
            for task in task_ids:
                exps, logs1 = self.agent.collect_experiences(task)
                exps_list.append(exps)
                logs1_list.append(logs1)

            logs2 = self.agent.update_parameters(exps_list, list(task_ids))
            self.update_end_time = time.time()
            steps += logs1['num_frames']
            updates += 1

            self.logs = (logs1_list, logs2)
                    
            if updates % log_freq == 0 or steps >= self.num_steps:
                self.save_data(updates, steps, task_ids, save_logs=log_bool)

        
        self.save_data(updates + 1, steps, task, save_logs=False, final=True)

        # self.writer.flush()
        for task in self.observed_tasks:
            self.csv_assimilation_file[task].flush()

class CompositionalLearner(Learner):
    def __init__(
        self,
        acmodel,
        num_steps,
        num_procs,
        steps_per_proc,
        gamma=0.99,
        learning_rate=1e-3,
        gae_lambda=0.95,
        entropy_coef=1e-2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        replay_capacity=100000,
        results_dir='./tmp/results/'
    ):
        super().__init__(
            acmodel,
            num_steps,
            num_procs,
            steps_per_proc,
            gamma=gamma,
            learning_rate=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward,
            replay_capacity=replay_capacity,
            results_dir=results_dir
        )

        self.test_env = {}

    def add_task(self, train_env, task_id, test_env):
        super().add_task(train_env, task_id)
        self.test_env[task_id] = test_env

    def train(
            self, 
            train_env, 
            task_id,
            test_env, 
            eval_episodes=10,
            log_freq=1e4, 
            component_update_freq=100,
            use_pcgrad=False
    ):

        if log_freq == -1:
            log_freq = np.inf
        
        if self.T >= self.acmodel.max_modules[0]:
            if task_id in self.acmodel.static_object_dict:  # if we're giving ground-truth assignments
                steps = 0   # assume the model structure is fixed
                self.acmodel.add_task(task_id, self.acmodel.static_object_dict[task_id], self.acmodel.target_object_dict[task_id], self.acmodel.agent_dyn_dict[task_id])    # re-add it to set BCQ
            else:
                steps = self.find_optimal_structure(task_id, test_env)
            # train copy
            self.backup_model = copy.deepcopy(self.acmodel)
        else:
            if task_id in self.acmodel.static_object_dict:  # if we're giving ground-truth assignments
                self.acmodel.add_task(task_id, self.acmodel.static_object_dict[task_id], self.acmodel.target_object_dict[task_id], self.acmodel.agent_dyn_dict[task_id])    # re-add it to set BCQ
            else:
                static = target = agent = self.T
                self.acmodel.add_task(task_id, static, target, agent)
            # train original model
            steps = 0
            self.backup_model = self.acmodel

        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1

        self.add_task(train_env, task_id, test_env)            # add new modules and structures if still not initialized
        
        log_bool = not np.isinf(log_freq)

        updates = 0
        self.start_time = time.time()

        while steps < self.num_steps:
            self.update_start_time = time.time()
            exps, logs1 = self.agent.collect_experiences(task_id)
            self.replay_buffer[task_id].push(exps)
            logs2 = self.agent.update_parameters(exps, task_id)
            self.logs = {**logs1, **logs2}
            self.update_end_time = time.time()

            
            steps += self.logs['num_frames']
            updates += 1    
            if updates % log_freq == 0 or steps >= self.num_steps:
                self.save_data(updates, steps, task_id, save_logs=log_bool)
        
        # Close all running processes from current env
        self.agent.env[task_id].terminate()

        # Recover original parameters before update_modules
        self.acmodel.load_state_dict(self.backup_model.state_dict())

        self.txt_logger.info('Assimilation, task {}. Trained:'.format(task_id))
        for name, param in self.acmodel.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.txt_logger.info('\t' + name)

        self.update_modules(task_id, use_pcgrad=use_pcgrad)
        for task in self.observed_tasks:
            self.evaluate(task, self.test_env[task], eval_episodes)
        self.save_data(updates + 1, steps, task_id, save_logs=True, final=True)

        self.csv_assimilation_file[task_id].flush()
        self.csv_accommodation_file.flush()

    def find_optimal_structure(self, task_id, test_env):
        max_reward = -100
        total_num_steps = 0
        for static, target, agent in product(range(self.acmodel.max_modules[0]), range(self.acmodel.max_modules[1]), range(self.acmodel.max_modules[2])):
            self.acmodel.add_task(task_id, static, target, agent)
            total_num_steps += self.evaluate(task_id, test_env, 10)
            if self.test_reward[task_id]['mean'] > max_reward:
                max_reward = self.test_reward[task_id]['mean']
                max_static = static
                max_target = target
                max_agent = agent

        # Set model structure to the optimal one
        self.acmodel.add_task(task_id, max_static, max_target, max_agent)
        del self.test_reward[task_id]

        return total_num_steps

    def update_modules(self, task_id, use_pcgrad=False):
        raise NotImplementedError('Update modules is algorithm specific')

class PandCLearner(Learner):
    def __init__(
        self,
        acmodel,
        num_steps,
        num_procs,
        steps_per_proc,
        gamma=0.99,
        learning_rate=1e-3,
        gae_lambda=0.95,
        entropy_coef=1e-2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        replay_capacity=100000,
        results_dir='./tmp/results/',
        use_replay=False,
        ewc_lambda=1.,
        ewc_gamma=1.,
        batch_size_fisher=3200
    ):
        super().__init__(
            acmodel,
            num_steps,
            num_procs,
            steps_per_proc,
            gamma=gamma,
            learning_rate=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward,
            replay_capacity=replay_capacity,
            results_dir=results_dir
        )

        self.test_env = {}
        self.use_replay = use_replay
        self.ewc_lambda = ewc_lambda
        self.ewc_gamma = ewc_gamma
        self.cumfisher = {}
        self.param_prev = {}
        self.replay_batch_size = batch_size
        self.batch_size_fisher = batch_size_fisher
        
    def add_task(self, train_env, task_id, test_env):
        super().add_task(train_env, task_id)
        self.test_env[task_id] = test_env

    def train(
            self, 
            train_env, 
            task_id,
            test_env, 
            eval_episodes=10,
            log_freq=1e4, 
            component_update_freq=100,
            use_pcgrad=False
    ):

        if log_freq == -1:
            log_freq = np.inf
        
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1

        self.add_task(train_env, task_id, test_env)            # add new modules and structures if still not initialized
        
        log_bool = not np.isinf(log_freq)

        updates = 0
        self.start_time = time.time()

        steps = 0
        self.acmodel.set_use_kb(False)
        self.acmodel.reset_active()
        self.agent.restart_optimizer()
        self.acmodel.freeze_active(False)
        self.acmodel.freeze_kb(True)
        
        while steps < self.num_steps:
            self.update_start_time = time.time()
            exps, logs1 = self.agent.collect_experiences(task_id)
            self.replay_buffer[task_id].push(exps)
            logs2 = self.agent.update_parameters(exps, task_id)
            self.logs = {**logs1, **logs2}
            self.update_end_time = time.time()

            
            steps += self.logs['num_frames']
            updates += 1    
            if updates % log_freq == 0 or steps >= self.num_steps:
                self.save_data(updates, steps, task_id, save_logs=log_bool)
        
        self.txt_logger.info('Assimilation, task {}. Trained:'.format(task_id))
        for name, param in self.acmodel.named_parameters():
            if param.requires_grad and param.grad is not None and torch.norm(param.grad) > 0:
                self.txt_logger.info('\t' + name)

        self.update_modules(task_id) 

        # Close all running processes from current env (after update_modules for P&C)
        self.agent.env[task_id].terminate()
        for task in self.observed_tasks:
            self.evaluate(task, self.test_env[task], eval_episodes)
        self.save_data(updates + 1, steps, task_id, save_logs=True, final=True)

        self.csv_assimilation_file[task_id].flush()
        self.csv_accommodation_file.flush()

    def update_modules(self, task_id, use_pcgrad=False):
        raise NotImplementedError('Update modules is algorithm specific')

class EWCLearner(Learner):
    def __init__(
        self,
        acmodel,
        num_steps,
        num_procs,
        steps_per_proc,
        gamma=0.99,
        learning_rate=1e-3,
        gae_lambda=0.95,
        entropy_coef=1e-2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        replay_capacity=100000,
        results_dir='./tmp/results/',
        use_replay=False,
        ewc_lambda=1.,
        ewc_gamma=1.,
        batch_size_fisher=3200
    ):
        super().__init__(
            acmodel,
            num_steps,
            num_procs,
            steps_per_proc,
            gamma=gamma,
            learning_rate=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward,
            replay_capacity=replay_capacity,
            results_dir=results_dir
        )

        # overwrite the agent with the EWC agent
        self.agent = EwcPPOAlgo(
            acmodel,
            num_procs=num_procs,
            num_frames_per_proc=steps_per_proc,
            discount=gamma,
            lr=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward,
            ewc_lambda=ewc_lambda,

        )

        self.test_env = {}
        self.use_replay = use_replay
        self.ewc_lambda = ewc_lambda
        self.ewc_gamma = ewc_gamma
        self.cumfisher = {}
        self.param_prev = {}
        self.replay_batch_size = batch_size
        self.batch_size_fisher = batch_size_fisher
        
    def add_task(self, train_env, task_id, test_env):
        super().add_task(train_env, task_id)
        self.test_env[task_id] = test_env

    def train(
            self, 
            train_env, 
            task_id,
            test_env, 
            eval_episodes=10,
            log_freq=1e4, 
            component_update_freq=100,
            use_pcgrad=False
    ):

        if log_freq == -1:
            log_freq = np.inf
        
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1

        self.add_task(train_env, task_id, test_env)            # add new modules and structures if still not initialized
        
        log_bool = not np.isinf(log_freq)

        updates = 0
        self.start_time = time.time()

        steps = 0
        self.agent.restart_optimizer()
        
        while steps < self.num_steps:
            self.update_start_time = time.time()
            exps, logs1 = self.agent.collect_experiences(task_id)
            old_model = copy.deepcopy(self.acmodel)   # for the Fisher
            logs2 = self.agent.update_parameters(exps, task_id)
            self.logs = {**logs1, **logs2}
            self.update_end_time = time.time()

            steps += self.logs['num_frames']
            updates += 1    
            if updates % log_freq == 0 or steps >= self.num_steps:
                self.save_data(updates, steps, task_id, save_logs=log_bool)
        
        self.txt_logger.info('Assimilation, task {}. Trained:'.format(task_id))
        for name, param in self.acmodel.named_parameters():
            if param.requires_grad and param.grad is not None and torch.norm(param.grad) > 0:
                self.txt_logger.info('\t' + name)

        self.agent.update_fisher(old_model, exps, task_id)

        # Close all running processes from current env (after update_modules for P&C)
        self.agent.env[task_id].terminate()
        for task in self.observed_tasks:
            self.evaluate(task, self.test_env[task], eval_episodes)
        self.save_data(updates + 1, steps, task_id, save_logs=True, final=True)

        self.csv_assimilation_file[task_id].flush()
        self.csv_accommodation_file.flush()

    def update_modules(self, task_id, use_pcgrad=False):
        raise NotImplementedError('Update modules is algorithm specific')

class CLEARLearner(Learner):
    def __init__(
        self,
        acmodel,
        num_steps,
        num_procs,
        steps_per_proc,
        gamma=0.99,
        learning_rate=1e-3,
        gae_lambda=0.95,
        entropy_coef=1e-2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        replay_batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        replay_capacity=100000,
        results_dir='./tmp/results/',
    ):
        super().__init__(
            acmodel,
            num_steps,
            num_procs,
            steps_per_proc,
            gamma=gamma,
            learning_rate=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward,
            replay_capacity=replay_capacity,
            results_dir=results_dir
        )

        # overwrite the agent with the EWC agent
        self.agent = PPOvTraceAlgo(
            acmodel,
            num_procs=num_procs,
            num_frames_per_proc=steps_per_proc,
            discount=gamma,
            lr=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size,
            replay_batch_size=replay_batch_size,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward,

        )

        self.test_env = {}
        self.replay_batch_size = batch_size
        
    def add_task(self, train_env, task_id, test_env):
        super().add_task(train_env, task_id)
        self.test_env[task_id] = test_env
        self.replay_buffer[task_id] = utils.ReplayBufferCLEAR(self.replay_capacity, self.acmodel.input_shape)

    def train(
            self, 
            train_env, 
            task_id,
            test_env, 
            eval_episodes=10,
            log_freq=1e4, 
            component_update_freq=100,
            use_pcgrad=False
    ):

        if log_freq == -1:
            log_freq = np.inf
        
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1

        self.add_task(train_env, task_id, test_env)            # add new modules and structures if still not initialized
        
        log_bool = not np.isinf(log_freq)

        updates = 0
        self.start_time = time.time()

        steps = 0
        self.agent.restart_optimizer()
        
        while steps < self.num_steps:
            self.update_start_time = time.time()
            exps, logs1 = self.agent.collect_experiences(task_id)
            self.replay_buffer[task_id].push(exps)
            logs2 = self.agent.update_parameters(exps, task_id, self.replay_buffer)
            self.logs = {**logs1, **logs2}
            self.update_end_time = time.time()

            steps += self.logs['num_frames']
            updates += 1    
            if updates % log_freq == 0 or steps >= self.num_steps:
                self.save_data(updates, steps, task_id, save_logs=log_bool)
        
        self.txt_logger.info('Assimilation, task {}. Trained:'.format(task_id))
        for name, param in self.acmodel.named_parameters():
            if param.requires_grad and param.grad is not None and torch.norm(param.grad) > 0:
                self.txt_logger.info('\t' + name)

        # Close all running processes from current env (after update_modules for P&C)
        self.agent.env[task_id].terminate()
        for task in self.observed_tasks:
            self.evaluate(task, self.test_env[task], eval_episodes)
        self.save_data(updates + 1, steps, task_id, save_logs=True, final=True)

        self.csv_assimilation_file[task_id].flush()
        self.csv_accommodation_file.flush()


