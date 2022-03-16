import numpy
import torch
import torch.nn.functional as F

from torch_ac_composable.algos.ppo import PPOAlgo

class EwcPPOAlgo(PPOAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, acmodel, num_procs=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, ewc_lambda=1., ewc_gamma=1.):
        

        super().__init__(acmodel, num_procs, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence,
                         adam_eps, clip_eps, epochs, batch_size,  preprocess_obss, reshape_reward)

        self.ewc_lambda = ewc_lambda
        self.ewc_gamma = ewc_gamma
        self.cumfisher = {}
        self.param_prev = {}

    def update_parameters(self, exps, task_id):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
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

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

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

                # Update actor-critic

                self.optimizer.zero_grad()
                # batch_loss.backward()
                batch_loss.backward()

                for name, param in self.acmodel.named_parameters():
                    if param.grad is not None and name in self.cumfisher:
                        param.grad.data.add_(self.ewc_lambda * self.cumfisher[name] * (param.data - self.param_prev[name].data))


                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
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

    def update_fisher(self, old_model, exps, task_id):

        # base distribution
        with torch.no_grad():
            base_dist, _ = old_model(exps.obs, task_id)    # this should be the last one used to collect data
        # differentiable distribution
        dist, _ = self.acmodel(exps.obs, task_id)

        kl = torch.distributions.kl.kl_divergence(base_dist, dist).mean()   # keep this sign because we're treating this as a LOSS and not an OBJECTIVE
        self.optimizer.zero_grad()
        kl.backward()
        for name, param in self.acmodel.named_parameters():
            if param.grad is not None:
                if name in self.cumfisher:
                    self.cumfisher[name] = self.ewc_gamma * self.cumfisher[name] + param.grad.detach() ** 2
                else:
                    self.cumfisher[name] = param.grad.detach() ** 2
                self.param_prev[name] = param.detach().clone()
