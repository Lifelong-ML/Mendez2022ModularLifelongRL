import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import copy


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)

class mlp(nn.Module):
    def __init__(
        self,
        sizes,
        activation,
        num_modules, 
        module_assignments, 
        module_inputs, 
        interface_depths,
        graph_structure,
        output_activation=nn.Identity
    ):
        super().__init__()
        self._num_modules = num_modules
        self.module_assignments = module_assignments
        self._module_inputs = module_inputs         # keys in a dict
        self._interface_depths = interface_depths
        self._graph_structure = graph_structure     # [[0], [1,2], 3] or [[0], [1], [2], [3]]   


        self._module_list = nn.ModuleList() # e.g., object, robot, task...
        
        for graph_depth in range(len(graph_structure)): # root -> children -> ... leaves 
            for j in graph_structure[graph_depth]:          # loop over all module types at this depth
                self._module_list.append(nn.ModuleDict())   # pre, post
                self._module_list[j]['pre_interface'] = nn.ModuleList()
                self._module_list[j]['post_interface'] = nn.ModuleList()
                
                for k in range(num_modules[j]):                 # loop over all modules of this type
                    layers_pre = []
                    layers_post = []
                    for i in range(len(sizes[j]) - 1):              # loop over all depths in this module
                        act = activation if graph_depth < len(graph_structure) - 1 or i < len(sizes[j])-2 else output_activation

                        if i == interface_depths[j]:
                            input_size = sum(sizes[j_prev][-1] for j_prev in graph_structure[graph_depth - 1])
                            input_size += sizes[j][i]
                        else:
                            input_size = sizes[j][i]

                        new_layer = [nn.Linear(input_size, sizes[j][i+1]), act()]
                        if i < interface_depths[j]:
                            layers_pre += new_layer
                        else:
                            layers_post += new_layer
                    if layers_pre:
                        self._module_list[j]['pre_interface'].append(nn.Sequential(*layers_pre))
                    else:   # it's either a root or a module with no preprocessing
                        self._module_list[j]['pre_interface'].append(nn.Identity())
                    self._module_list[j]['post_interface'].append(nn.Sequential(*layers_post))

    def forward(self, input_val, *, task_id):
        x = None
        for graph_depth in range(len(self._graph_structure)):     # root -> children -> ... -> leaves
            x_post = []
            for j in self._graph_structure[graph_depth]:          # nodes (modules) at this depth
                if isinstance(self._module_inputs[j], list):
                    x_pre = torch.cat(tuple(input_val[each_input] for each_input in self._module_inputs[j]), dim=-1)
                else:
                    x_pre = input_val[self._module_inputs[j]]
                x_pre = self._module_list[j]['pre_interface'][self.module_assignments[j][task_id]](x_pre)
                if x is not None: x_pre = torch.cat((x, x_pre), dim=-1)
                x_post.append(self._module_list[j]['post_interface'][self.module_assignments[j][task_id]](x_pre))
            x = torch.cat(x_post, dim=-1)
        return x


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs, *, task_id):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act, *, task_id):
        raise NotImplementedError

    def forward(self, obs, *, task_id, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, task_id=task_id)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, 
        hidden_sizes,
        num_modules,
        module_assignments,
        module_inputs,
        interface_depths,
        graph_structure,
        activation):

        super().__init__()

        sizes = list(hidden_sizes)
        for j in range(len(sizes)):
            if isinstance(module_inputs[j], list):
                input_size = sum(obs_dim[each_input] for each_input in module_inputs[j])
            else:
                input_size = obs_dim[module_inputs[j]]
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [act_dim]

        self.logits_net = mlp(sizes=sizes,
            activation=activation,
            num_modules=num_modules,
            module_assignments=module_assignments,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure)

    def _distribution(self, obs, *, task_id):
        logits = self.logits_net(obs, task_id=task_id)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, 
        hidden_sizes,
        num_modules,
        module_assignments,
        module_inputs,
        interface_depths,
        graph_structure,
        activation):

        super().__init__()

        sizes = list(hidden_sizes)
        for j in range(len(sizes)):
            if isinstance(module_inputs[j], list):
                input_size = sum(obs_dim[each_input] for each_input in module_inputs[j])
            else:
                input_size = obs_dim[module_inputs[j]]
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [act_dim]

        self.shared_log_std = torch.tensor(np.zeros(act_dim, dtype=np.float32))
        self.shared_log_std[-1] = -0.5
        self.log_std = {}
        self.mu_net = mlp(sizes=sizes,
            activation=activation, 
            output_activation=nn.Tanh,
            num_modules=num_modules,
            module_assignments=module_assignments,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure)

    def _distribution(self, obs, *, task_id):
        mu = self.mu_net(obs, task_id=task_id)
        std = torch.exp(self.log_std[str(task_id)])
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def new_logstd(self, *, task_id):
        self.log_std[str(task_id)] = self.shared_log_std    # as_tensor doesn't copy data, tensor does

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, 
        hidden_sizes,
        num_modules,
        module_assignments,
        module_inputs,
        interface_depths,
        graph_structure,
        activation):

        super().__init__()

        sizes = list(hidden_sizes)
        for j in range(len(sizes)):
            if isinstance(module_inputs[j], list):
                input_size = sum(obs_dim[each_input] for each_input in module_inputs[j])
            else:
                input_size = obs_dim[module_inputs[j]]
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [1]

        self.v_net = mlp(sizes=sizes,
            activation=activation,
            num_modules=num_modules,
            module_assignments=module_assignments,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure)

    def forward(self, obs, *, task_id):
        return torch.squeeze(self.v_net(obs, task_id=task_id), -1) # Critical to ensure v has right shape.

class MLPQCritic(nn.Module):

    def __init__(self, obs_dim, act_dim,
        hidden_sizes,
        num_modules,
        module_assignments,
        module_inputs,
        interface_depths,
        graph_structure,
        activation):

        super().__init__()
        self._graph_structure = graph_structure
        self._module_inputs = module_inputs

        sizes = list(hidden_sizes)
        for j in range(len(sizes)):
            if isinstance(module_inputs[j], list):
                input_size = sum(obs_dim[each_input] for each_input in module_inputs[j])
            else:
                input_size = obs_dim[module_inputs[j]]
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [1]
                sizes[j][0] += act_dim

        self.q_net = mlp(sizes=sizes,
            activation=activation,
            num_modules=num_modules,
            module_assignments=module_assignments,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure)

    def forward(self, obs, act, *, task_id):
        obs = copy.copy(obs)    # shallow copy
        for j in self._graph_structure[-1]:
            if isinstance(self._module_inputs[j], list):
                obs[self._module_inputs[j][-1]] = torch.cat((obs[self._module_inputs[j][-1]], act), dim=-1)
            else:
                obs[self._module_inputs[j]] = torch.cat((obs[self._module_inputs[j]], act), dim=-1)
        return torch.squeeze(self.q_net(obs, task_id=task_id), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, action_space, 
                 hidden_sizes=None,
                 num_modules=None,
                 module_assignments=None,
                 module_inputs=None,
                 interface_depths=None,
                 graph_structure=None, 
                 activation=nn.Tanh,
                 step_q=False):
        super().__init__()
        # policy builder depends on action space
        if module_assignments is None:
            module_assignments = [{} for _ in range(len(num_modules))]
        self.seen_tasks = set(module_assignments[0].keys())    
        self.module_assignments = module_assignments
        self.num_modules = num_modules
        self.module_inputs = module_inputs
        self.graph_structure = graph_structure
        self.step_q = step_q
        self.use_bcq = {}
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], 
                hidden_sizes=hidden_sizes,
                num_modules=num_modules,
                module_assignments=module_assignments,
                module_inputs=module_inputs,
                interface_depths=interface_depths,
                graph_structure=graph_structure,
                activation=activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, 
                hidden_sizes=hidden_sizes,
                num_modules=num_modules,
                module_assignments=module_assignments,
                module_inputs=module_inputs,
                interface_depths=interface_depths,
                graph_structure=graph_structure,
                activation=activation)

        if self.step_q:
            # build q functions
            self.qf1 = MLPQCritic(obs_dim, action_space.shape[0], 
                hidden_sizes=hidden_sizes,
                num_modules=num_modules,
                module_assignments=module_assignments,
                module_inputs=module_inputs,
                interface_depths=interface_depths,
                graph_structure=graph_structure,
                activation=activation)

            self.qf2 = MLPQCritic(obs_dim, action_space.shape[0], 
                hidden_sizes=hidden_sizes,
                num_modules=num_modules,
                module_assignments=module_assignments,
                module_inputs=module_inputs,
                interface_depths=interface_depths,
                graph_structure=graph_structure,
                activation=activation)
        else:
            # build value function
            self.v  = MLPCritic(obs_dim, 
                hidden_sizes=hidden_sizes,
                num_modules=num_modules,
                module_assignments=module_assignments,
                module_inputs=module_inputs,
                interface_depths=interface_depths,
                graph_structure=graph_structure, 
                activation=activation)

        self.apply(init_weights)
        for j in graph_structure[-1]:          # loop over all module types at the last depth
            for k in range(num_modules[j]):                 # loop over all modules of this type
                for p in self.pi.mu_net._module_list[j]['post_interface'][k][-2:].parameters():
                    p.data.mul_(0.01)   

    def set_use_bcq(self, *, task_id, use_bcq=True):
        self.use_bcq[task_id] = use_bcq

    def step(self, obs, *, task_id, deterministic=False):
        with torch.no_grad():
            if task_id in self.use_bcq and self.use_bcq[task_id]:
                pi = self.pi._distribution(obs, task_id=task_id)
                if deterministic:
                    a = pi.loc
                else:
                    a = pi.sample()
                return a.numpy(), None, None
            else:
                pi = self.pi._distribution(obs, task_id=task_id)
                if deterministic:
                    a = pi.loc
                else:
                    a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                if self.step_q:
                    a_samples = pi.sample((10,))
                    obs = {o_k: o_v.unsqueeze(0).expand(10, -1) for o_k, o_v in obs.items()}
                    v1 = self.qf1(obs, a_samples, task_id=task_id).mean(axis=0)
                    v2 = self.qf2(obs, a_samples, task_id=task_id).mean(axis=0)
                    v = (v1 + v2) / 2
                else:
                    v = self.v(obs, task_id=task_id)
                return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs, *, task_id, deterministic=False):
        return self.step(obs, task_id=task_id, deterministic=deterministic)[0]

    def load_from(self, other):
        self.pi.load_state_dict(other.pi.state_dict())
        if self.step_q:
            self.qf1.load_state_dict(other.qf1.state_dict())
            self.qf2.load_state_dict(other.qf2.state_dict())
        else:
            self.v.load_state_dict(other.v.state_dict())

    def set_assignments(self, module_assignments, *, task_id):
        for i, asgn in enumerate(module_assignments):
            self.module_assignments[i][task_id] = asgn
        self.pi.new_logstd(task_id=task_id)
        self.seen_tasks.add(task_id)
        

