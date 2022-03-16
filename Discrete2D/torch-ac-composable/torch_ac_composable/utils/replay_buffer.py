from collections import deque
import numpy as np
import random
import torch

class ReplayBufferTensors(torch.utils.data.Dataset):
    def __init__(self, capacity, state_shape):
        self.state_buffer = torch.empty((capacity, *state_shape), dtype=torch.float32)
        self.action_buffer = torch.empty(capacity, dtype=torch.long)
        self.reward_buffer = torch.empty(capacity, dtype=torch.float32)
        self.next_state_buffer = torch.empty_like(self.state_buffer, dtype=torch.float32)
        self.done_buffer = torch.empty(capacity, dtype=torch.float32)
        self.capacity = capacity
        self.index = 0
        self.len = 0

    def push(self, exps):
        
        idxs = np.arange(self.index, self.index + len(exps)) % self.capacity
        
        self.state_buffer.data[idxs] = exps.state.image
        self.action_buffer.data[idxs] = exps.action.cpu()
        self.reward_buffer.data[idxs] = exps.reward.cpu()
        self.next_state_buffer.data[idxs] = exps.next_state.image
        self.done_buffer.data[idxs] = exps.done.cpu()


        self.index = (self.index + len(exps)) % self.capacity
        self.len = min(self.len + len(exps), self.capacity)

    def sample(self, batch_size):
        idx_sample = random.sample(range(self.len), batch_size)
        return (self.state_buffer[idx_sample], 
                self.action_buffer[idx_sample], 
                self.reward_buffer[idx_sample], 
                self.next_state_buffer[idx_sample],
                self.done_buffer[idx_sample])

    def __getitem__(self, index):
        return (self.state_buffer[index], 
                self.action_buffer[index], 
                self.reward_buffer[index], 
                self.next_state_buffer[index],
                self.done_buffer[index] )

    def __len__(self):
        return self.len

class ReplayBufferCLEAR(torch.utils.data.Dataset):
    def __init__(self, capacity, state_shape):
        self.state_buffer = torch.empty((capacity, *state_shape), dtype=torch.float32)
        self.action_buffer = torch.empty(capacity, dtype=torch.long)
        self.reward_buffer = torch.empty(capacity, dtype=torch.float32)
        self.done_buffer = torch.empty(capacity, dtype=torch.float32)
        self.value_buffer = torch.empty(capacity, dtype=torch.float32)
        self.logprob_buffer = torch.empty(capacity, dtype=torch.float32)

        self.capacity = capacity
        self.index = 0
        self.len = 0
        self.prev_idx_sample = 0

    def push(self, exps):
        
        idxs = np.arange(self.index, self.index + len(exps)) % self.capacity
        
        self.state_buffer.data[idxs] = exps.state.image
        self.action_buffer.data[idxs] = exps.action.cpu()
        self.reward_buffer.data[idxs] = exps.reward.cpu()
        self.done_buffer.data[idxs] = exps.done.cpu()
        self.value_buffer.data[idxs] = exps.value.cpu()
        self.logprob_buffer.data[idxs] = exps.log_prob.cpu()

        self.index = (self.index + len(exps)) % self.capacity
        self.len = min(self.len + len(exps), self.capacity)

    def sample(self, batch_size):
        idx_sample = random.sample(range(self.len), batch_size)
        return (self.state_buffer[idx_sample], 
                self.action_buffer[idx_sample], 
                self.reward_buffer[idx_sample], 
                self.done_buffer[idx_sample],
                self.value_buffer[idx_sample],
                self.logprob_buffer[idx_sample])

    def sample_in_order(self, batch_size):
        idx_sample = np.arange(self.prev_idx_sample, self.prev_idx_sample + batch_size) % self.len
        self.prev_idx_sample = (idx_sample[-1] + 1) % self.len

        return (self.state_buffer[idx_sample], 
                self.action_buffer[idx_sample], 
                self.reward_buffer[idx_sample], 
                self.done_buffer[idx_sample],
                self.value_buffer[idx_sample],
                self.logprob_buffer[idx_sample])

    def __getitem__(self, index):
        return (self.state_buffer[index], 
                self.action_buffer[index], 
                self.reward_buffer[index], 
                self.done_buffer[index],
                self.value_buffer[index],
                self.logprob_buffer[index])

    def __len__(self):
        return self.len

class ReplayBufferSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield random.sample(range(len(self.data_source)), self.batch_size)

