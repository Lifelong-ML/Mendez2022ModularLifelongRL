import torch
import torch.nn.functional as F
from algos.agent_wrappers import PandCLearner
import random 
import copy

import time
import numpy

class MonolithicPandCPPO(PandCLearner):
    def update_modules(self, task_id):

        active_model = copy.deepcopy(self.acmodel)  # need to copy so updates to KB don't affect active model
        active_model.set_use_kb(False)     # always use active column
        active_model.freeze_active(True)   # freeze all parameters
        active_model.freeze_kb(True)

        self.acmodel.set_use_kb(True)     
        self.acmodel.freeze_active(True)
        self.acmodel.freeze_kb(False)

        for name, param in self.acmodel.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.txt_logger.info('\t' + name)

        loader = torch.utils.data.DataLoader(
            self.replay_buffer[task_id],
            batch_size=self.replay_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        for epoch in range(10):
            print('epoch: {}'.format(epoch))
            for epoch_i, batch in enumerate(loader):
                state = batch[0].to(self.acmodel.device, non_blocking=True)
                state.image = state

                # Compute loss
                kb_dist, _ = self.acmodel(state, task_id)

                active_dist, value = active_model(state, task_id)

                loss = torch.distributions.kl.kl_divergence(active_dist, kb_dist).mean()
                
                self.agent.optimizer.zero_grad()
                loss.backward()

                for name, param in self.acmodel.named_parameters():
                    if param.grad is not None and name in self.cumfisher:
                        param.grad.data.add_(self.ewc_lambda * self.cumfisher[name] * (param.data - self.param_prev[name].data))
                
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.agent.max_grad_norm)
                
                self.agent.optimizer.step()

        self.txt_logger.info('Accommodation task {}. Trained:'.format(task_id))
        for name, param in self.acmodel.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.txt_logger.info('\t' + name)

        # Add latest Fisher
        self.update_fisher(active_model, task_id)

        self.txt_logger.info('Fisher updated task {}. Keys:'.format(task_id))
        self.txt_logger.info('\t' + str(self.cumfisher.keys()))

        self.acmodel.set_use_kb(True)      # make sure this is used for evaluation post-accommodation    


    def update_fisher(self, active_model, task_id):
        loader = torch.utils.data.DataLoader(
            self.replay_buffer[task_id],
            batch_size=self.batch_size_fisher,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        batch = next(iter(loader))
        state = batch[0].to(self.acmodel.device, non_blocking=True)
        state.image = state

        # active
        active_dist, _ = active_model(state, task_id)
        # kb
        kb_dist, _ = self.acmodel(state, task_id)

        kl = torch.distributions.kl.kl_divergence(active_dist, kb_dist).mean()
        self.agent.optimizer.zero_grad()
        kl.backward()
        for name, param in self.acmodel.named_parameters():
            if param.grad is not None:
                if name in self.cumfisher:
                    self.cumfisher[name] = self.ewc_gamma * self.cumfisher[name] + param.grad.detach() ** 2
                else:
                    self.cumfisher[name] = param.grad.detach() ** 2
                self.param_prev[name] = param.detach().clone()
