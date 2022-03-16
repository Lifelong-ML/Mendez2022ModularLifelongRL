'''
This version uses a Q function for PPO, the same that is 
later used for BCQ
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



class ACModelModularFixed(nn.Module):
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

        self.use_bcq = {}
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
        self.static = nn.ModuleList()
        for i in range(max_modules[0]):
            self.static.append(nn.Sequential(
                nn.Conv2d(5, 8, kernel_size=2),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(8, 16, kernel_size=2),
                nn.ReLU()
            ).to(self.device))

        # Target object (conv2)
        self.target_pre = nn.ModuleList()
        self.target_post = nn.ModuleList()
        for i in range(max_modules[1]):
            self.target_pre.append(nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=2),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(8, 16, kernel_size=2),
                nn.ReLU()
            ).to(self.device))
            self.target_post.append(nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=2),
                nn.ReLU()
            ).to(self.device))
        
        # Agent dynamics (actor, critic)
        self.agent_pre = nn.ModuleList()
        for i in range(max_modules[2]):
            self.agent_pre.append(nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=2),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(8, 16, kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=2),
                nn.ReLU()
            ).to(self.device))

        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        for i in range(max_modules[2]):
            self.actor_layers.append(nn.Sequential(
                nn.Linear(self.feature_size(), self.sizes[3]),
                nn.Tanh(),
                nn.Linear(self.sizes[3], self.num_actions)
            ).to(self.device))

            self.critic_layers.append(nn.Sequential(
                nn.Linear(self.feature_size(), self.sizes[3]),
                nn.Tanh(),
                nn.Linear(self.sizes[3], self.num_actions)
            ).to(self.device))

        # Initialize parameters correctly
        self.apply(init_params)

        self.to(self.device)


    def features(self, x, task_id):
        n = x.shape[0]
        x_static = x[:, :5, :, :]
        x_target = x[:, 5:6, :, :]
        x_agent = x[:, 6:, :, :]

        x_static = self.static[self.static_object_dict[task_id]](x_static)

        x_target = self.target_pre[self.target_object_dict[task_id]](x_target)
        x_target = torch.cat((x_static, x_target), dim=1)
        x_target = self.target_post[self.target_object_dict[task_id]](x_target)

        x_agent = self.agent_pre[self.agent_dyn_dict[task_id]](x_agent)
        x_agent = torch.cat((x_target, x_agent), dim=1)

        return x_agent

    def fc(self, x, task_id, return_bc=False):
        if return_bc:
            x_q = self.critic_layers[self.agent_dyn_dict[task_id]](x)
            x_bc = self.actor_layers[self.agent_dyn_dict[task_id]](x)
            return x_q, F.log_softmax(x_bc, dim=1), x_bc

        x_actor = self.actor_layers[self.agent_dyn_dict[task_id]](x)
        x_critic = self.critic_layers[self.agent_dyn_dict[task_id]](x).max(dim=1, keepdim=True)[0]


        return x_actor, x_critic

    def forward(self, obs, task_id, return_bc=False):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.features(x, task_id)
        features = x.view(x.size(0), -1)

        x = self.fc(features, task_id, return_bc)

        if not return_bc:
            x_actor, x_critic = x
            dist = Categorical(logits=F.log_softmax(x_actor, dim=1))
            value = x_critic.squeeze(1)
            return dist, value

        return x

    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape, device=self.device).transpose(1, 3).transpose(2, 3))
        x_static = x[:, :5, :, :]
        x_target = x[:, 5:6, :, :]
        x_agent = x[:, 6:, :, :]

        x_static = self.static[0](x_static)
        x_target = self.target_pre[0](x_target)
        x_target = torch.cat((x_static, x_target), dim=1)
        x_target = self.target_post[0](x_target)  
        x_agent = self.agent_pre[0](x_agent)
        x_agent = torch.cat((x_target, x_agent), dim=1)

        return x_agent.reshape(1, -1).size(1)

    def act(self, state, epsilon, task_id):
        # with torch.no_grad():
        #     q_value, bc_prob, _ = self.forward(state, task_id, return_bc=True)
        #     bc_prob = bc_prob.exp()
        #     bc_prob = (bc_prob / bc_prob.max(1, keepdim=True)[0] > self.threshold).float()

        #     q_value  = (bc_prob * q_value + (1 - bc_prob) * -1e8)

        #     dist = Categorical(logits=F.log_softmax(q_value, dim=1))

        #     action = dist.sample()

        # return action
        with torch.no_grad():
            q_value, bc_prob, _ = self.forward(state, task_id, return_bc=True)
            bc_prob = bc_prob.exp()
            bc_prob = (bc_prob / bc_prob.max(1, keepdim=True)[0] > self.threshold).float()

            q_value  = (bc_prob * q_value + (1 - bc_prob) * -1e8)

            dist = Categorical(logits=F.log_softmax(q_value, dim=1))

            action = dist.sample()

        return action
        
    def add_task(self, task_id, static_object, target_object, agent_dyn):
        self.static_object_dict[task_id] = static_object
        self.target_object_dict[task_id] = target_object
        self.agent_dyn_dict[task_id] = agent_dyn
        self.set_use_bcq(task_id, False)

    def set_use_bcq(self, task_id, use_bcq=False):
        self.use_bcq[task_id] = use_bcq

    def anneal_tau(*args, **kwargs):
        pass