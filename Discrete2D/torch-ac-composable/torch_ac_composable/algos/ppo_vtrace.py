import numpy
import torch
import torch.nn.functional as F

from torch_ac_composable.algos.base import BaseAlgo

class PPOvTraceAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, acmodel, num_procs=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, replay_batch_size=256):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(acmodel, num_procs, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.replay_batch_size = replay_batch_size

        assert self.batch_size % self.recurrence == 0

        self.batch_num = 0
        self.lr = lr
        self.adam_eps = adam_eps

    def update_parameters(self, exps, task_id, replay_buffer):
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            capacity_others = len(replay_buffer[0]) if 0 in replay_buffer and task_id != 0 else 0
            num_others = len(replay_buffer) - (1 if task_id in replay_buffer else 0)
            inds_list = self._get_batches_starting_indexes()
            for inds in inds_list:
                mtl_entropy = 0
                mtl_value = 0
                mtl_policy_loss = 0
                mtl_value_loss = 0
                mtl_loss = 0

                # Current task / online data
                # Initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory
                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask, task_id)
                    else:
                        dist, value = self.acmodel(sb.obs, task_id)

                    entropy = dist.entropy().mean()

                    behavior_logprob = sb.log_prob
                    agent_logprob = dist.log_prob(sb.action)
                    discounts = self.discount * (1 - sb.done)
                    ratio = torch.exp(agent_logprob - behavior_logprob)   
                    v_trace_advantage, v_trace_vs = self.v_trace(behavior_logprob, agent_logprob, actions=sb.action,
                                                discounts=discounts,
                                                rewards=sb.reward,
                                                values=value,
                                                done=sb.done)
                    surr1 = ratio * v_trace_advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * v_trace_advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (value - v_trace_vs).pow(2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss 

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                mtl_entropy += batch_entropy
                mtl_value += batch_value
                mtl_policy_loss += batch_policy_loss
                mtl_value_loss += batch_value_loss
                mtl_loss += batch_loss

                for task, exps_other in replay_buffer.items():
                    if task == task_id:     # in case it's also in the buffer
                        continue
                    # For every batch sampled for the current task, sample a batch from memory across past tasks of the same size
                    batch_size_others = self.replay_batch_size // self.recurrence // (len(replay_buffer) - (1 if task_id in replay_buffer else 0))

                    batch_entropy = 0
                    batch_value = 0
                    batch_policy_loss = 0
                    batch_value_loss = 0
                    batch_loss = 0

                    # Initialize memory

                    if self.acmodel.recurrent:
                        memory = exps_other.memory[inds_replay]

                    for i in range(self.recurrence):
                        # Create a sub-batch of experience

                        # sb = exps[inds_replay + i]
                        sb = exps_other.sample_in_order(batch_size_others)
                        obs, action, reward, done, value_replay, log_prob = sb     # from ReplayBufferCLEAR
                        obs = obs.to(self.acmodel.device)
                        action = action.to(self.acmodel.device)
                        reward = reward.to(self.acmodel.device)
                        done = done.to(self.acmodel.device)
                        value_replay = value_replay.to(self.acmodel.device)
                        log_prob = log_prob.to(self.acmodel.device)
                        obs.image = obs

                        # Compute loss

                        if self.acmodel.recurrent:
                            raise ValueError
                            dist, value, memory = self.acmodel(sb.obs, memory * sb.mask, task)
                        else:
                            dist, value = self.acmodel(obs, task)

                        entropy = dist.entropy().mean()

                        behavior_logprob = log_prob
                        agent_logprob = dist.log_prob(action)
                        discounts = self.discount * (1 - done)
                        ratio = torch.exp(agent_logprob - behavior_logprob)   
                        v_trace_advantage, v_trace_vs = self.v_trace(behavior_logprob, agent_logprob, actions=action,
                                                    discounts=discounts,
                                                    rewards=reward,
                                                    values=value,
                                                    done=done)
                        surr1 = ratio * v_trace_advantage
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * v_trace_advantage
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = (value - v_trace_vs).pow(2).mean()

                        ### Behavior cloning
                        policy_replay_loss = F.nll_loss(dist.logits, action, reduction='mean')
                        value_replay_loss = (value - value_replay).pow(2).mean()

                        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss + policy_replay_loss + value_replay_loss

                        # Update batch values

                        batch_entropy += entropy.item()
                        batch_value += value.mean().item()
                        batch_policy_loss += policy_loss.item()
                        batch_value_loss += value_loss.item()
                        batch_loss += loss

                        # Update memories for next epoch

                        if self.acmodel.recurrent and i < self.recurrence - 1:
                            exps_other.memory[inds_replay + i + 1] = memory.detach()

                    # Update batch values

                    batch_entropy /= self.recurrence
                    batch_value /= self.recurrence
                    batch_policy_loss /= self.recurrence
                    batch_value_loss /= self.recurrence
                    batch_loss /= self.recurrence

                    mtl_entropy += batch_entropy
                    mtl_value += batch_value
                    mtl_policy_loss += batch_policy_loss
                    mtl_value_loss += batch_value_loss
                    mtl_loss += batch_loss

                mtl_entropy /= (len(replay_buffer) + 1)
                mtl_value /= (len(replay_buffer) + 1)
                mtl_policy_loss /= (len(replay_buffer) + 1)
                mtl_value_loss /= (len(replay_buffer) + 1)
                mtl_loss /= (len(replay_buffer) + 1)
                # Update actor-critic

                self.optimizer.zero_grad()
                # batch_loss.backward()
                mtl_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(mtl_entropy)
                log_values.append(mtl_value)
                log_policy_losses.append(mtl_policy_loss)
                log_value_losses.append(mtl_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        '''
        For V-trace, I need things to be in order (due to the n-step returns), so no shuffling or shifting
        '''

        indexes = numpy.arange(0, self.num_frames, self.recurrence)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def v_trace(self, behaviour_policy_logits, target_policy_logits, actions,
                discounts, rewards, values, done,
                clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0,):
        '''
        Adapted from: https://github.com/deepmind/scalable_agent/blob/master/vtrace.py
        '''
        try: 
            with torch.no_grad():       # these are targets for losses, so they shouldn't have grads
                # need to check the shapes and such


                target_action_log_probs = target_policy_logits
                behaviour_action_log_probs = behaviour_policy_logits
                rhos = torch.exp(target_action_log_probs - behaviour_action_log_probs)
                clipped_rhos = torch.clamp(rhos, 0., clip_rho_threshold)
                cs = torch.clamp(rhos, 0., 1.)

                values_t_plus_1 = torch.cat([values[1:], values[-1:]])      # first, use the last value at the end for truncated trajectory
                values_t_plus_1[done == 1.] = values[done == 1.]                              # next, for last steps in each trajectory, replace with last value of that trajectory

                deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

                sequences = (discounts, cs, deltas)

                # I have no idea how to vectorize this, and the V-trace authors didn't either. For now, this might just be fine
                vs_minus_v_xs = torch.zeros_like(values)
                acc = 0
                for t in range(-1, -(done.shape[0] + 1), -1):   # reverse for loop
                    vs_minus_v_xs[t] = deltas[t] + discounts[t] * cs[t] * acc
                    if done[t] == 1.:
                        acc = 0
                    else:
                        acc = vs_minus_v_xs[t]

                # Add V(x_s) to get v_s.
                vs = vs_minus_v_xs + values

                # Advantage for policy gradient.
                vs_t_plus_1 = torch.cat([vs[1:], values[-1:]])      # bootstrap just like above
                vs_t_plus_1[done == 1.] = values[done == 1.]

                clipped_pg_rhos = torch.clamp(rhos, 0., clip_pg_rho_threshold)
                pg_advantages = (
                    clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))

            return pg_advantages, vs
        except:
            import pdb
            pdb.set_trace()
            raise
