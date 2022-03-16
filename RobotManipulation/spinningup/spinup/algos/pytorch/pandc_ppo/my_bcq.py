import copy
import torch
from torch.optim import Adam
from spinup.utils.mpi_tools import mpi_avg, num_procs, proc_id
from spinup.utils.mpi_pytorch import mpi_avg_grads
import numpy as np
import spinup.algos.pytorch.pandc_ppo.core as core
import time
import torch.nn.functional as F

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for BCQ agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = {obs_key: np.zeros(core.combined_shape(size, obs_val), dtype=np.float32) for obs_key, obs_val in obs_dim.items()}
        self.obs2_buf = {obs_key: np.zeros(core.combined_shape(size, obs_val), dtype=np.float32) for obs_key, obs_val in obs_dim.items()}
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        for ob_key in obs.keys():
            self.obs_buf[ob_key][self.ptr] = obs[ob_key]
            self.obs2_buf[ob_key][self.ptr] = next_obs[ob_key]
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs={k: v[idxs] for k, v in self.obs_buf.items()},
                     obs2={k: v[idxs] for k, v in self.obs2_buf.items()},
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        tensor_dict = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items() if k != 'obs' and k != 'obs2'}
        tensor_dict['obs'] = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch['obs'].items()}
        tensor_dict['obs2'] = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch['obs2'].items()}

        return tensor_dict

def mt_bcq(ac, task_list, replay_buffer, *,
            buffer_batch_size=128,
            gamma=0.99,
            soft_target_tau=5e-3,
            pi_lr=1e-3,
            qf_lr=1e-3,
            reward_scale=1.,
            num_epochs=10):
    
    if proc_id() == 0: print('(BCQ) accommodation tasks: ', task_list)

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    qf1_optimizer = Adam(ac.qf1.parameters(), lr=qf_lr)
    qf2_optimizer = Adam(ac.qf2.parameters(), lr=qf_lr)
    target_qf1_net = copy.deepcopy(ac.qf1)
    target_qf2_net = copy.deepcopy(ac.qf2)

    first_buff = next(iter(replay_buffer.values()))
    local_batch_size = buffer_batch_size // num_procs()
    steps_per_epoch = first_buff.size // local_batch_size  # assume all buffers are the same size

    start = time.time()
    for epoch in range(num_epochs):
        if proc_id() == 0: print('epoch', epoch, time.time() - start)
        for step in range(steps_per_epoch):
            pi_loss = 0.
            qf1_loss = 0.
            qf2_loss = 0.

            for task in task_list:
                samples = replay_buffer[task].sample_batch(local_batch_size)

                rewards = samples['rew']
                dones = samples['done']
                obs = samples['obs']
                actions = samples['act']
                next_obs = samples['obs2']

                """
                Behavior clone a policy
                """
                dist,_ = ac.pi(obs, task_id=task)      # Normal distribution
                pi_loss += F.gaussian_nll_loss(dist.mean, actions, dist.variance)        # match the distribution of inputs

                """
                Critic Training
                """
                with torch.no_grad():
                    # Duplicate state 10 times (10 is a hyperparameter chosen by BCQ)
                    state_rep = {o_key: o.unsqueeze(1).repeat(1, 10, 1).view(o.shape[0] * 10, o.shape[1]) for o_key, o in next_obs.items()}

                    # Compute value of perturbed actions sampled from the actor
                    action_dist, _ = ac.pi(state_rep, task_id=task) 
                    action_rep = action_dist.sample()
                    target_qf1 = target_qf1_net(state_rep, action_rep, task_id=task)
                    target_qf2 = target_qf2_net(state_rep, action_rep, task_id=task)

                    # Soft Clipped Double Q-learning 
                    target_Q = 0.75 * torch.min(target_qf1, target_qf2) + 0.25 * torch.max(target_qf1, target_qf2)
                    target_Q = target_Q.view(actions.shape[0], -1).max(1)[0].view(-1, 1)
                    target_Q = reward_scale * rewards + (1.0 - dones) * gamma * target_Q
                qf1_pred = ac.qf1(obs, actions, task_id=task)
                qf2_pred = ac.qf2(obs, actions, task_id=task)

                qf1_loss += (qf1_pred - target_Q.detach()).pow(2).mean()
                qf2_loss += (qf2_pred - target_Q.detach()).pow(2).mean()

            pi_loss /= len(task_list)
            qf1_loss /= len(task_list)
            qf2_loss /= len(task_list)
            pi_optimizer.zero_grad(),   qf1_optimizer.zero_grad(),  qf2_optimizer.zero_grad()
            pi_loss.backward(),         qf1_loss.backward(),        qf2_loss.backward()
            mpi_avg_grads(ac.pi),       mpi_avg_grads(ac.qf1),      mpi_avg_grads(ac.qf2)
            pi_optimizer.step(),        qf1_optimizer.step(),       qf2_optimizer.step()

            _update_targets(ac.qf1, ac.qf2, target_qf1_net, target_qf2_net, soft_target_tau)

def _update_targets(qf1, qf2, target_qf1, target_qf2, soft_target_tau):
    """Update parameters in the target q-functions."""
    target_qfs = [target_qf1, target_qf2]
    qfs = [qf1, qf2]
    for target_qf, qf in zip(target_qfs, qfs):
        for t_param, param in zip(target_qf.parameters(), qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - soft_target_tau) +
                               param.data * soft_target_tau)
