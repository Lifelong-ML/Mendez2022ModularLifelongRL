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
    if classname.find("Linear") != -1 and classname.find("Active") == -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ActiveConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, activation):
        super().__init__()
        self.conv_lateral = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            activation,
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, bias=False)
        )
        self.gates = nn.parameter.Parameter(torch.rand(c_out, 1, 1) * 0.1)
        self.conv_active = nn.Conv2d(c_in, c_out, kernel_size=kernel_size)
        self.activation = activation

    def forward(self, x, x_kb):
        return self.activation(self.conv_active(x) + self.gates * self.conv_lateral(x_kb))
        # return self.activation(self.conv_active(x))

class ActiveLinear(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.linear_lateral = nn.Sequential(
            nn.Linear(n_in, n_out, bias=False),
            activation,
            nn.Linear(n_out, n_out, bias=False)
        )
        self.gates = nn.parameter.Parameter(torch.rand(n_out) * 0.1)
        self.linear_active = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x, x_kb):
        return self.activation(self.linear_active(x) + self.gates * self.linear_lateral(x_kb))
        # return self.activation(self.linear_active(x))



class ACModelProgressCompress(nn.Module):
    def __init__(
        self,
        input_shape,
        num_actions,
        agent_dyn_dict,
        static_object_dict,
        target_object_dict,
        max_modules=0,
        threshold=0.3,
        device=torch.device('cuda')
    ):
        super().__init__()

        # self.use_bcq = {}
        self.use_kb = True
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

        self.create_kb()
        self.create_active()

        # Initialize parameters correctly
        self.apply(init_params)

        self.to(self.device)

    def create_kb(self):
        ############ Static ############
        self.static_kb_conv0 = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.static_kb_conv1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU()
        )
        self.static_kb = nn.Sequential(     # for compress
            self.static_kb_conv0,
            self.static_kb_conv1
        )

        ############ Target ############
        self.target_pre_kb_conv0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )  
        self.target_pre_kb_conv1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU()
        )
        self.target_post_kb_conv0 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2),
            nn.ReLU()
        )
        self.target_pre_kb = nn.Sequential(
            self.target_pre_kb_conv0,
            self.target_pre_kb_conv1,
        )
        self.target_post_kb = nn.Sequential(
            self.target_post_kb_conv0
        )

        ############ Agent ############
        self.agent_pre_kb_conv0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.agent_pre_kb_conv1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2),
            nn.ReLU()
        )
        self.agent_pre_kb_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU()
        )
        self.agent_pre_kb = nn.Sequential(
            self.agent_pre_kb_conv0,
            self.agent_pre_kb_conv1,
            self.agent_pre_kb_conv2
        )

        # odd placing to make things work with CUDA without much hacking
        self.feature_size = self.features(autograd.Variable(torch.zeros(1, *self.input_shape).transpose(1, 3).transpose(2, 3))).view(1, -1).size(1)
        

        self.actor_kb_linear0 = nn.Sequential(
            nn.Linear(self.feature_size + sum(self.max_modules), 64),
            nn.Tanh()
        )   
        self.actor_kb_linear1 = nn.Linear(64, self.num_actions)
        self.critic_kb_linear0 = nn.Sequential(
            nn.Linear(self.feature_size + sum(self.max_modules), 64),
            nn.Tanh()
        )
        self.critic_kb_linear1 = nn.Linear(64, 1)
        self.actor_kb = nn.Sequential(
            self.actor_kb_linear0,
            self.actor_kb_linear1
        )
        self.critic_kb = nn.Sequential(
            self.critic_kb_linear0,
            self.critic_kb_linear1
        )

    def create_active(self):
        ############ Static ############
        self.static_active_conv0 = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.static_active_conv1 = ActiveConv2d(8, 16, kernel_size=2, activation=nn.ReLU())
        
        ############ Target ############
        self.target_pre_active_conv0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )  
        self.target_pre_active_conv1 = ActiveConv2d(8, 16, kernel_size=2, activation=nn.ReLU())
        self.target_post_active_conv0 = ActiveConv2d(32, 32, kernel_size=2, activation=nn.ReLU())

        ############ Agent ############
        self.agent_pre_active_conv0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.agent_pre_active_conv1 = ActiveConv2d(8, 16, kernel_size=2, activation=nn.ReLU())
        self.agent_pre_active_conv2 = ActiveConv2d(16, 32, kernel_size=2, activation=nn.ReLU())
        self.actor_active_linear0 = ActiveLinear(self.feature_size + sum(self.max_modules), 64, activation=nn.Tanh())
        self.actor_active_linear1 = ActiveLinear(64, self.num_actions, activation=nn.Identity())
        self.critic_active_linear0 = ActiveLinear(self.feature_size + sum(self.max_modules), 64, activation=nn.Tanh())
        self.critic_active_linear1 = ActiveLinear(64, 1, activation=nn.Identity())

    def features(self, x):
        x_static = x[:, :5, :, :]
        x_target = x[:, 5:6, :, :]
        x_agent = x[:, 6:, :, :]
        if self.use_kb:
            x_static = self.static_kb(x_static)

            x_target = self.target_pre_kb(x_target)
            x_target = torch.cat((x_static, x_target), dim=1)
            x_target = self.target_post_kb(x_target)

            x_agent = self.agent_pre_kb(x_agent)
            x_agent = torch.cat((x_target, x_agent), dim=1)
        else:
            x_static_active = self.static_active_conv0(x_static)
            x_static_kb = self.static_kb_conv0(x_static)
            #
            # x_static_kb = torch.zeros_like(x_static_kb)
            #
            # print(x_static_active.shape, x_static_kb.shape)
            x_static_active = self.static_active_conv1(x_static_active, x_static_kb)
            x_static_kb = self.static_kb_conv1(x_static_kb)
            #
            # x_static_kb = torch.zeros_like(x_static_kb)
            #
            # print(x_static_active.shape, x_static_kb.shape)

            x_target_active = self.target_pre_active_conv0(x_target)
            x_target_kb = self.target_pre_kb_conv0(x_target)
            #
            # x_target_kb = torch.zeros_like(x_target_kb)
            #
            # print(x_target_active.shape, x_target_kb.shape)
            x_target_active = self.target_pre_active_conv1(x_target_active, x_target_kb)
            x_target_kb = self.target_pre_kb_conv1(x_target_kb)
            #
            # x_target_kb = torch.zeros_like(x_target_kb)
            #
            # print(x_target_active.shape, x_target_kb.shape)
            x_target_active = torch.cat((x_static_active, x_target_active), dim=1)
            x_target_kb = torch.cat((x_static_kb, x_target_kb), dim=1)
            #
            # x_target_kb = torch.zeros_like(x_target_kb)
            #
            # print(x_target_active.shape, x_target_kb.shape)
            x_target_active = self.target_post_active_conv0(x_target_active, x_target_kb)
            x_target_kb = self.target_post_kb_conv0(x_target_kb)
            #
            # x_target_kb = torch.zeros_like(x_target_kb)
            #
            # print(x_target_active.shape, x_target_kb.shape)

            x_agent_active = self.agent_pre_active_conv0(x_agent)
            x_agent_kb = self.agent_pre_kb_conv0(x_agent)
            #
            # x_agent_kb = torch.zeros_like(x_agent_kb)
            #
            # print(x_agent_active.shape, x_agent_kb.shape)
            x_agent_active = self.agent_pre_active_conv1(x_agent_active, x_agent_kb)
            x_agent_kb = self.agent_pre_kb_conv1(x_agent_kb)
            #
            # x_agent_kb = torch.zeros_like(x_agent_kb)
            #
            # print(x_agent_active.shape, x_agent_kb.shape)
            x_agent_active = self.agent_pre_active_conv2(x_agent_active, x_agent_kb)
            x_agent_kb = self.agent_pre_kb_conv2(x_agent_kb)
            #
            # x_agent_kb = torch.zeros_like(x_agent_kb)
            #
            # print(x_agent_active.shape, x_agent_kb.shape)
            x_agent_active = torch.cat((x_target_active, x_agent_active), dim=1)
            x_agent_kb = torch.cat((x_target_kb, x_agent_kb), dim=1)
            #
            # x_agent_kb = torch.zeros_like(x_agent_kb)
            #
            # print(x_agent_active.shape, x_agent_kb.shape)
            x_agent = x_agent_active, x_agent_kb

        return(x_agent)

    def fc(self, x, task_id):
        if self.use_kb:
            x = x.view(x.size(0), -1)
            
            task_descriptor = torch.zeros(x.shape[0], sum(self.max_modules), device=self.device, dtype=x.dtype)
            task_descriptor.data[:, self.static_object_dict[task_id]] = 1
            task_descriptor.data[:, sum(self.max_modules[:1]) + self.target_object_dict[task_id]] = 1
            task_descriptor.data[:, sum(self.max_modules[:2]) + self.agent_dyn_dict[task_id]] = 1
            x = torch.cat((x, task_descriptor), dim=1)

            x_actor = self.actor_kb(x)
            x_critic = self.critic_kb(x)
        else:
            x_active, x_kb = x
            x_active = x_active.view(x_active.size(0), -1)
            x_kb = x_kb.view(x_kb.size(0), -1)
            
            task_descriptor = torch.zeros(x_kb.shape[0], sum(self.max_modules), device=self.device, dtype=x_kb.dtype)
            task_descriptor.data[:, self.static_object_dict[task_id]] = 1
            task_descriptor.data[:, sum(self.max_modules[:1]) + self.target_object_dict[task_id]] = 1
            task_descriptor.data[:, sum(self.max_modules[:2]) + self.agent_dyn_dict[task_id]] = 1
            
            x_active = torch.cat((x_active, task_descriptor), dim=1)
            x_kb = torch.cat((x_kb, task_descriptor), dim=1)
            #
            # x_kb = torch.zeros_like(x_kb)
            #

            x_actor_active = self.actor_active_linear0(x_active, x_kb)
            x_actor_kb = self.actor_kb_linear0(x_kb)
            #
            # x_actor_kb = torch.zeros_like(x_actor_kb)
            #
            # print(x_actor_active.shape, x_actor_kb.shape)
            x_actor = self.actor_active_linear1(x_actor_active, x_actor_kb)

            x_critic_active = self.critic_active_linear0(x_active, x_kb)
            x_critic_kb = self.critic_kb_linear0(x_kb)
            #
            # x_critic_kb = torch.zeros_like(x_critic_kb)
            #
            # print(x_critic_active.shape, x_critic_kb.shape)
            x_critic = self.critic_active_linear1(x_critic_active, x_critic_kb)
            # exit()
        return x_actor, x_critic


    def forward(self, obs, task_id):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.features(x)

        x_actor, x_critic = self.fc(x, task_id)
        dist = Categorical(logits=F.log_softmax(x_actor, dim=1))
        value = x_critic.squeeze(1)

        return dist, value

    def freeze_kb(self, freeze=True):
        all_kb_parameters = (
                list(self.static_kb.parameters()) + 
                list(self.target_pre_kb.parameters()) +
                list(self.target_post_kb.parameters()) +
                list(self.agent_pre_kb.parameters()) + 
                list(self.actor_kb.parameters()) +
                list(self.critic_kb.parameters())
        )
        for param in all_kb_parameters:
            param.requires_grad = not freeze
            if freeze:
                param.grad = None

    def freeze_active(self, freeze=True):
        all_active_parameters = (
            list(self.static_active_conv0.parameters()) +
            list(self.static_active_conv1.parameters()) +
            list(self.target_pre_active_conv0.parameters()) +
            list(self.target_pre_active_conv1.parameters()) +
            list(self.target_post_active_conv0.parameters()) +
            list(self.agent_pre_active_conv0.parameters()) +
            list(self.agent_pre_active_conv1.parameters()) +
            list(self.agent_pre_active_conv2.parameters()) +
            list(self.actor_active_linear0.parameters()) +
            list(self.actor_active_linear1.parameters()) +
            list(self.critic_active_linear0.parameters()) +
            list(self.critic_active_linear1.parameters())
        )

        for param in all_active_parameters:
            param.requires_grad = not freeze
            if freeze:
                param.grad = None

    def set_use_kb(self, use_kb):
        self.use_kb = use_kb

    def reset_active(self):
        self.create_active()
        self.actor_active_linear0.apply(init_params)
        self.actor_active_linear1.apply(init_params)
        self.critic_active_linear0.apply(init_params)
        self.critic_active_linear1.apply(init_params)
        self.to(self.device)
