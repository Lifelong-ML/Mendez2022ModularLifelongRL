import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd 
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


class ACModel(nn.Module):
    def __init__(self, input_shape, num_actions, device=torch.device('cuda')):
        super().__init__()
        
        self.device = device
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.recurrent = False
        self.recurrence = 1

        self.static = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU()
        )

        self.target_pre = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU()
        )
        self.target_post = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2),
            nn.ReLU()
        )

        self.agent_pre = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size(), 64),
            nn.Tanh(),
            nn.Linear(64, self.num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size(), 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

        self.to(self.device)

    def features(self, x):
        x_static = x[:, :5, :, :]
        x_target = x[:, 5:6, :, :]
        x_agent = x[:, 6:, :, :]

        x_static = self.static(x_static)

        x_target = self.target_pre(x_target)
        x_target = torch.cat((x_static, x_target), dim=1)
        x_target = self.target_post(x_target)

        x_agent = self.agent_pre(x_agent)
        x_agent = torch.cat((x_target, x_agent), dim=1)

        return(x_agent)
        
    def forward(self, obs, _=None):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.features(x)
        features = x.view(x.size(0), -1)

        x = self.actor(features)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(features)
        value = x.squeeze(1)

        return dist, value
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape).transpose(1, 3).transpose(2, 3))).view(1, -1).size(1)
    
    # def act(self, state, epsilon, task_id=None):
    #     # task_id is added for compatibility with lifelong methods, but is ignored as this class deals with a single task
    #     if random.random() > epsilon:
    #         with torch.no_grad():
    #             state   = autograd.Variable(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))#, volatile=True)
    #             q_value = self.forward(state)
    #             action  = q_value.max(1)[1].data[0]
    #     else:
    #         action = random.randrange(self.num_actions)
    #     return action

    def add_task(self, _):
        pass
