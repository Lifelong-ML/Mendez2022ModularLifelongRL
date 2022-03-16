import copy
import torch
from torch.optim import Adam
from spinup.utils.mpi_tools import mpi_avg, num_procs, proc_id
from spinup.utils.mpi_pytorch import mpi_avg_grads
import numpy as np
import spinup.algos.pytorch.pandc_ppo.core as core
import time
import torch.nn.functional as F

def ewc(kb_model, active_model, replay_buffer, *,
            task_id,
            cumfisher,
            param_prev,
            buffer_batch_size=128,
            pi_lr=1e-3,
            num_epochs=10,
            max_grad_norm=0.5,
            batch_size_fisher=12800,
            ewc_lambda=1.,
            ewc_gamma=1.,
            ):
    
    pi_optimizer = Adam(kb_model.pi.parameters(), lr=pi_lr)

    local_batch_size = buffer_batch_size // num_procs()
    steps_per_epoch = replay_buffer.size // local_batch_size 

    start = time.time()
    for epoch in range(num_epochs):
        if proc_id() == 0: print('epoch', epoch, time.time() - start)
        for step in range(steps_per_epoch):
            samples = replay_buffer.sample_batch(local_batch_size)

            rewards = samples['rew']
            dones = samples['done']
            obs = samples['obs']
            actions = samples['act']
            next_obs = samples['obs2']

            """
            Behavior clone a policy
            """
            kb_dist,_ = kb_model.pi(obs, task_id=task_id)      # Normal distribution
            active_dist,_ = active_model.pi(obs, task_id=task_id)
            
            pi_loss = torch.distributions.kl.kl_divergence(active_dist, kb_dist).mean()


            pi_optimizer.zero_grad()
            pi_loss.backward()
            mpi_avg_grads(kb_model.pi)

            for name, param in kb_model.pi.named_parameters():
                if param.grad is not None and name in cumfisher:
                    param.grad.data.add_(ewc_lambda * cumfisher[name] * (param.data - param_prev[name].data))
            
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in kb_model.parameters() if p.grad is not None) ** 0.5
            torch.nn.utils.clip_grad_norm_(kb_model.pi.parameters(), max_grad_norm)
            pi_optimizer.step()

    update_fisher(kb_model=kb_model, active_model=active_model, replay_buffer=replay_buffer, batch_size_fisher=batch_size_fisher, task_id=task_id, cumfisher=cumfisher, ewc_gamma=ewc_gamma, param_prev=param_prev)

def update_fisher(kb_model, active_model, replay_buffer, *, batch_size_fisher, task_id, cumfisher, ewc_gamma, param_prev):
    local_batch_size_fisher = batch_size_fisher // num_procs()
    batch = replay_buffer.sample_batch(local_batch_size_fisher)
    obs = batch['obs']

    # active
    active_dist, _ = active_model.pi(obs, task_id=task_id)
    # kb
    kb_dist, _ = kb_model.pi(obs, task_id=task_id)

    kl = torch.distributions.kl.kl_divergence(active_dist, kb_dist).mean()
    
    for param in kb_model.pi.parameters():
        if param.grad is not None: param.grad.data.zero_()
    kl.backward()
    mpi_avg_grads(kb_model.pi)
    for name, param in kb_model.pi.named_parameters():
        if param.grad is not None:
            if name in cumfisher:
                cumfisher[name] = ewc_gamma * cumfisher[name] + param.grad.detach() ** 2
            else:
                cumfisher[name] = param.grad.detach() ** 2
            param_prev[name] = param.detach().clone()
