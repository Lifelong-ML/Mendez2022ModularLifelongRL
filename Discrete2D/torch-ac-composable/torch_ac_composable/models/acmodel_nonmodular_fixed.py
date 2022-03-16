'''
This version uses modules to process the input, but not
split across tasks
'''
import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random
import numpy as np

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)



class ACModelNonModularFixed(nn.Module):
    def __init__(
        self, 
        input_shape, 
        num_actions,
        agent_dyn_dict,
        static_object_dict,
        target_object_dict,
        max_modules=0,
        threshold=0.3,
        device=torch.device('cuda'),
    ):
        super().__init__()

        self.threshold = threshold

        self.device = device
        if isinstance(max_modules, (int, float)):
            max_modules = max_modules if max_modules != 0 else np.inf
            max_modules = [max_modules] * 4

        # List of selections of modules per task
        self.static_object_dict = static_object_dict
        self.target_object_dict = target_object_dict
        self.agent_dyn_dict = agent_dyn_dict

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.recurrent = False
        self.recurrence = 1

        self.max_modules = max_modules
        self.num_modules = max_modules
        self.sizes = [8, 16, 32, 64]
        self.num_tasks = 0
        
        # Static object (conv0 and 1)
        self.static = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU()
        ).to(self.device)

        # Target object (conv2)
        self.target_pre = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU()
        ).to(self.device)
        self.target_post = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2),
            nn.ReLU()
        ).to(self.device)
        
        # Agent dynamics (actor, critic)
        self.agent_pre = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU()
        ).to(self.device)

        self.actor = nn.Sequential(
            nn.Linear(self.feature_size() + sum(self.max_modules), self.sizes[3]),
            nn.Tanh(),
            nn.Linear(self.sizes[3], self.num_actions)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(self.feature_size() + sum(self.max_modules), self.sizes[3]),
            nn.Tanh(),
            nn.Linear(self.sizes[3], 1)
        ).to(self.device)

        # Initialize parameters correctly
        self.apply(init_params)

        self.to(self.device)


    def features(self, x, task_id):
        n = x.shape[0]
        x_static = x[:, :5, :, :]
        x_target = x[:, 5:6, :, :]
        x_agent = x[:, 6:, :, :]

        x_static = self.static(x_static)

        x_target = self.target_pre(x_target)
        x_target = torch.cat((x_static, x_target), dim=1)
        x_target = self.target_post(x_target)

        x_agent = self.agent_pre(x_agent)
        x_agent = torch.cat((x_target, x_agent), dim=1)

        return x_agent

    def forward(self, obs, task_id, return_bc=False, verbose=False):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.features(x, task_id)
        features = x.view(x.size(0), -1)

        task_descriptor = torch.zeros(features.shape[0], sum(self.max_modules), device=self.device, dtype=features.dtype)
        task_descriptor.data[:, self.static_object_dict[task_id]] = 1
        task_descriptor.data[:, sum(self.max_modules[:1]) + self.target_object_dict[task_id]] = 1
        task_descriptor.data[:, sum(self.max_modules[:2]) + self.agent_dyn_dict[task_id]] = 1
        
        if verbose:
            print(task_descriptor)
            # print(self.max_modules)
            # print(self.agent_dyn_dict[task_id])
            # print(self.static_object_dict[task_id])
            # print(self.target_object_dict[task_id])
            # exit()
            pass
        features = torch.cat((features, task_descriptor), dim=1)

        x = self.actor(features)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(features)
        value = x.squeeze(1)

        return dist, value

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape, device=self.device).transpose(1, 3).transpose(2, 3)), 0).view(1, -1).size(1)

    def add_task(self, task_id, _):
        pass
