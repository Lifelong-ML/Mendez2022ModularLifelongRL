import torch
import torch.nn.functional as F
from algos.agent_wrappers import PandCLearner
import random 
import copy

import time
import numpy

class MonolithicPandCBcqPPO(PandCLearner):
    def update_modules(self, task_id):
        target_update = 1000
        self.bcq_batch_size = 256
        bcq_learning_rate = 1e-3
        ppo_learning_rate = self.agent.lr 
        self.agent.lr = bcq_learning_rate

        self.acmodel.set_use_bcq(task_id, True) 

        self.acmodel.set_use_kb(True)     
        self.acmodel.freeze_active(True)
        self.acmodel.freeze_kb(False)

        self.target_acmodel = copy.deepcopy(self.acmodel)

        self.agent.restart_optimizer()

        for name, param in self.acmodel.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.txt_logger.info('\t' + name)

        accommodation_tasks = list(self.observed_tasks) # all seen tasks

        loaders = {task: torch.utils.data.DataLoader(
                                            dataset, 
                                            batch_size=self.bcq_batch_size, 
                                            shuffle=True, 
                                            num_workers=0,
                                            pin_memory=True)
                        for task, dataset in self.replay_buffer.items() if task in accommodation_tasks
                    }

        self._naive_epoch(loaders, task_id, target_update)

        self.txt_logger.info('Accommodation task {}. Trained:'.format(task_id))
        for name, param in self.acmodel.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.txt_logger.info('\t' + name)

        self.agent.lr = ppo_learning_rate

    def update_target(self):
        self.target_acmodel.load_state_dict(self.acmodel.state_dict())

    def _naive_epoch(self, loaders, task_id, target_update):
        iter_num = 0
        
        for i in range(10):      # a single epoch
            loaders_iter = {task: iter(l) for task, l in loaders.items()}
            done = False
            while not done:     # assume same memory sizes
                loss = 0.
                n = 0
                for task, l in loaders_iter.items():
                    try:
                        batch = next(l)
                        loss += self.compute_loss(batch, task, use_bcq=True)
                        n += self.bcq_batch_size
                        # self.gradient_step(batch, task_id)
                    except StopIteration:
                        done = True
                        break
                if not done:
                    loss /= n
                    self.agent.optimizer.zero_grad()
                    loss.backward()
                    self.agent.optimizer.step()

                    if (iter_num + 1) % target_update == 0:
                        self.update_target()
                    iter_num += 1

        self.acmodel.set_use_kb(True)      # make sure this is used for evaluation post-accommodation    

    def compute_loss(self, batch, task_id, use_bcq=False):
        if not use_bcq:
            return super().compute_loss(batch, task_id)
        state, action, reward, next_state, done = batch

        state = state.to(self.acmodel.device, non_blocking=True)
        action = action.to(self.acmodel.device, non_blocking=True)
        reward = reward.to(self.acmodel.device, non_blocking=True)
        next_state = next_state.to(self.acmodel.device, non_blocking=True)
        done = done.to(self.acmodel.device, non_blocking=True)

        # hacky, ugly, to avoid changing code too much...
        state.image = state
        next_state.image = next_state

        with torch.no_grad():
            next_q_values, bc_prob, _ = self.acmodel(next_state, task_id, return_bc=True)
            bc_prob = bc_prob.exp()
            bc_prob = (bc_prob / bc_prob.max(1, keepdim=True)[0] > self.acmodel.threshold).float()
            next_action = (bc_prob * next_q_values + (1 - bc_prob) * -1e8).argmax(1, keepdim=True)

            next_q_state_values, _, _ = self.target_acmodel(next_state, task_id, return_bc=True) 
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            expected_q_value = reward + self.agent.discount * next_q_value * (1 - done)
        
        q_values, bc_prob, bc_original = self.acmodel(state, task_id, return_bc=True)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        
        loss = 1e-3 * (q_value - expected_q_value).pow(2).sum()
        loss += F.nll_loss(bc_prob, action, reduction='sum') 
        loss += 1e-2 * bc_original.pow(2).sum()   
        
        return loss
