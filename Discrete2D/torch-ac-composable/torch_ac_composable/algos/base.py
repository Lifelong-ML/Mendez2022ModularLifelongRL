from abc import ABC, abstractmethod
import torch

from torch_ac_composable.format import default_preprocess_obss
from torch_ac_composable.utils import DictList, ParallelEnv

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, acmodel, num_procs, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.acmodel = acmodel
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.train()

        # Store helpers values

        self.num_procs = num_procs
        self.num_frames = self.num_frames_per_proc * self.num_procs

        self.env = {}
        self.obs = {}
        self.obss = {}
        self.next_obss = {}     # for DQN/BCQ
        self.done = {}          # for DQN/BCQ
        if self.acmodel.recurrent:
            self.memory = {}
            self.memories = {}
        self.mask = {}
        self.masks = {}
        self.actions = {}
        self.values = {}
        self.rewards = {}
        self.advantages = {}
        self.log_probs = {}

        self.dists = {}     # for CLEAR

        # Initialize log values

        self.log_episode_return = {}
        self.log_episode_reshaped_return = {}
        self.log_episode_num_frames = {}

        self.log_done_counter = {}
        self.log_return = {}
        self.log_reshaped_return = {}
        self.log_num_frames = {}

    def add_task(self, envs, task_id):
        self.restart_optimizer()
        self.reset(envs, task_id)

        # Initialize experience values
        shape = (self.num_frames_per_proc, self.num_procs)
        self.obss[task_id] = [None]*(shape[0])
        self.next_obss[task_id] = [None]*(shape[0])
        if self.acmodel.recurrent:
            self.memory[task_id] = torch.zeros(shape[1], self.acmodel.memory_size, device=self.acmodel.device)
            self.memories[task_id] = torch.zeros(*shape, self.acmodel.memory_size, device=self.acmodel.device)
        self.mask[task_id] = torch.ones(shape[1], device=self.acmodel.device)
        self.masks[task_id] = torch.zeros(*shape, device=self.acmodel.device)
        self.done[task_id] = torch.zeros(*shape, device=self.acmodel.device)
        self.actions[task_id] = torch.zeros(*shape, device=self.acmodel.device, dtype=torch.long)
        self.values[task_id] = torch.zeros(*shape, device=self.acmodel.device)
        self.rewards[task_id] = torch.zeros(*shape, device=self.acmodel.device)
        self.advantages[task_id] = torch.zeros(*shape, device=self.acmodel.device)
        self.log_probs[task_id] = torch.zeros(*shape, device=self.acmodel.device)

        # Initialize log values

        self.log_episode_return[task_id] = torch.zeros(self.num_procs, device=self.acmodel.device)
        self.log_episode_reshaped_return[task_id] = torch.zeros(self.num_procs, device=self.acmodel.device)
        self.log_episode_num_frames[task_id] = torch.zeros(self.num_procs, device=self.acmodel.device)

        self.log_done_counter[task_id] = 0
        self.log_return[task_id] = [0] * self.num_procs
        self.log_reshaped_return[task_id] = [0] * self.num_procs
        self.log_num_frames[task_id] = [0] * self.num_procs

    def restart_optimizer(self):
        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.lr, eps=self.adam_eps)

    def reset(self, envs, task_id):
        '''
        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        '''
        self.env[task_id] = ParallelEnv(envs)
        self.obs[task_id] = self.env[task_id].reset()

    def collect_experiences(self, task_id):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs[task_id], device=self.acmodel.device)
            
            if hasattr(self.acmodel, 'use_bcq') and task_id in self.acmodel.use_bcq and self.acmodel.use_bcq[task_id]:
                action = self.acmodel.act(preprocessed_obs, 0.01, task_id)
                dist, value = self.acmodel(preprocessed_obs, task_id)
            else:
                with torch.no_grad():
                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                    else:
                        dist, value = self.acmodel(preprocessed_obs, task_id)
                action = dist.sample()
            obs, reward, done, _ = self.env[task_id].step(action.cpu().numpy())
            # Update experiences values

            self.obss[task_id][i] = self.obs[task_id]
            self.next_obss[task_id][i] = obs
            self.obs[task_id] = obs
            if self.acmodel.recurrent:
                self.memories[task_id][i] = self.memory[task_id]
                self.memory[task_id] = memory
            self.masks[task_id][i] = self.mask[task_id]
            self.mask[task_id] = 1 - torch.tensor(done, device=self.acmodel.device, dtype=torch.float)
            self.done[task_id][i] = torch.tensor(done, device=self.acmodel.device, dtype=torch.float)
            self.actions[task_id][i] = action
            self.values[task_id][i] = value
            if self.reshape_reward is not None:
                self.rewards[task_id][i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.acmodel.device)
            else:
                self.rewards[task_id][i] = torch.tensor(reward, device=self.acmodel.device)
            self.log_probs[task_id][i] = dist.log_prob(action)
            # Update log values

            self.log_episode_return[task_id] += torch.tensor(reward, device=self.acmodel.device, dtype=torch.float)
            self.log_episode_reshaped_return[task_id] += self.rewards[task_id][i]
            self.log_episode_num_frames[task_id] += torch.ones(self.num_procs, device=self.acmodel.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter[task_id] += 1
                    self.log_return[task_id].append(self.log_episode_return[task_id][i].item())
                    self.log_reshaped_return[task_id].append(self.log_episode_reshaped_return[task_id][i].item())
                    self.log_num_frames[task_id].append(self.log_episode_num_frames[task_id][i].item())

            self.log_episode_return[task_id] *= self.mask[task_id]
            self.log_episode_reshaped_return[task_id] *= self.mask[task_id]
            self.log_episode_num_frames[task_id] *= self.mask[task_id]
        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs[task_id], device=self.acmodel.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory[task_id] * self.mask[task_id].unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs, task_id)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[task_id][i+1] if i < self.num_frames_per_proc - 1 else self.mask[task_id]
            next_value = self.values[task_id][i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[task_id][i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[task_id][i] + self.discount * next_value * next_mask - self.values[task_id][i]
            self.advantages[task_id][i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[task_id][i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.state = [self.obss[task_id][i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.next_state = [self.next_obss[task_id][i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories[task_id].transpose(0, 1).reshape(-1, *self.memories[task_id].shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks[task_id].transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.done = self.done[task_id].transpose(0, 1).reshape(-1)
        exps.action = self.actions[task_id].transpose(0, 1).reshape(-1)
        exps.value = self.values[task_id].transpose(0, 1).reshape(-1)
        exps.reward = self.rewards[task_id].transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages[task_id].transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs[task_id].transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.acmodel.device)
        # pass to CPU for replay buffer (for BCQ)
        exps.state = self.preprocess_obss(exps.state, device='cpu')
        exps.next_state = self.preprocess_obss(exps.next_state, device='cpu')

        # Log some values

        keep = max(self.log_done_counter[task_id], self.num_procs)

        logs = {
            "return_per_episode": self.log_return[task_id][-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[task_id][-keep:],
            "num_frames_per_episode": self.log_num_frames[task_id][-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter[task_id] = 0
        self.log_return[task_id] = self.log_return[task_id][-self.num_procs:]
        self.log_reshaped_return[task_id] = self.log_reshaped_return[task_id][-self.num_procs:]
        self.log_num_frames[task_id] = self.log_num_frames[task_id][-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
