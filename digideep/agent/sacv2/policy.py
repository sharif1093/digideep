"""
This implementation is mainly adopted from `pytorch-soft-actor-critic <https://github.com/pranz24/pytorch-soft-actor-critic>`__.
"""

import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
# ---
from digideep.utility.toolbox import get_class
# ---
from digideep.agent.policy_base import PolicyBase
from digideep.agent.policy_common import Averager
# ---
from copy import deepcopy


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class Policy(PolicyBase):
    """Implementation of a stochastic actor-critic policy for the SAC method.

    Args:
        obs_space: The observation space of the environment.
        act_space: The action space of the environment.
        actor_args (dict): A dictionary of arguments for the :class:`ActorNetwork`.
        softq_args (dict): A dictionary of arguments for the :class:`SoftQNetwork`.
        value_args (dict): A dictionary of arguments for the :class:`ValueNetwork`.
        average_args (dict): A dictionary of arguments for the :class:`Averager`.
    
    Todo:
        Override the base class ``state_dict`` and ``load_state_dict`` to also save the state of ``averager``.

    """
    def __init__(self, device, **params):
        super(Policy, self).__init__(device)

        self.params = params

        assert len(self.params["obs_space"]["dim"]) == 1, "We only support 1d observations for the SACv2 policy for now."
        assert self.params["act_space"]["typ"] == "Box", "We only support continuous actions in SACv2 policy for now."

        state_size  = self.params["obs_space"]["dim"][0]
        action_size = self.params["act_space"]["dim"] if np.isscalar(self.params["act_space"]["dim"]) else self.params["act_space"]["dim"][0]
        hidden_size = self.params["hidden_size"]
        
        self.model["critic1"] = QNetwork(state_size, action_size, hidden_size, **self.params["critic_args"])
        self.model["critic1_target"] = deepcopy(self.model["critic1"])

        self.model["critic2"] = QNetwork(state_size, action_size, hidden_size, **self.params["critic_args"])
        self.model["critic2_target"] = deepcopy(self.model["critic2"])

        self.model["actor"] = GaussianPolicy(state_size, action_size, hidden_size, **self.params["actor_args"])

        self.averager = {}
        self.averager["critic1"] = Averager(self.model["critic1"], self.model["critic1_target"], **self.params["average_args"])
        self.averager["critic2"] = Averager(self.model["critic2"], self.model["critic2_target"], **self.params["average_args"])
        
        self.post_init()

    
    def generate_actions(self, inputs, deterministic=False):
        """
        This function generates actions from the "actor" model.
        """
        # inputs = torch.FloatTensor(inputs).unsqueeze(0).to(device)
        with torch.no_grad():
            self.model.eval()

            mean, log_std = self.model["actor"](inputs)
            std = log_std.exp()
            
            dist = distributions.Normal(mean, std)

            # If not deterministic sample the distribution, otherwise use mean or median.
            if not deterministic:
                z = dist.sample()
            else:
                z = dist.mean
            action = torch.tanh(z)
            
            self.model.train()
            return action

    def evaluate_actions(self, state):
        mean, log_std = self.model["actor"](state)
        std = log_std.exp()
        dist = distributions.Normal(mean, std) # validate_args=True
        z = dist.rsample() # For reparameterization trick (mean + std * N(0,1))
        
        # NOTE: We are assuming normalized action altogether. So no need to scale the log_prob and the action here.
        action = torch.tanh(z)
    
        log_prob = dist.log_prob(z)
        # Correction term to enforce action bound. See Haarnoja et al. (2018): http://arxiv.org/abs/1801.01290.
        log_prob -= torch.log(1 - action.pow(2) + self.params["epsilon"])
        # log_prob = log_prob.sum(-1, keepdim=True)
        log_prob = log_prob.sum(1, keepdim=True)
    
        return action, log_prob
        # , z, mean, log_std


    # def sample(self, state):
        #     mean, log_std = self.forward(state)
        #     std = log_std.exp()
        #     normal = Normal(mean, std)
        #     x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        #     y_t = torch.tanh(x_t)
        #     action = y_t * self.action_scale + self.action_bias
        #     log_prob = normal.log_prob(x_t)
        #     # Enforcing Action Bound
        #     log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        #     log_prob = log_prob.sum(1, keepdim=True)
        #     return action, log_prob, torch.tanh(mean)

    
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x = F.relu(self.linear1(xu))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std



# class DeterministicPolicy(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
#         super(DeterministicPolicy, self).__init__()
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)

#         self.mean = nn.Linear(hidden_dim, num_actions)
#         self.noise = torch.Tensor(num_actions)

#         self.apply(weights_init_)

#         # action rescaling
#         if action_space is None:
#             self.action_scale = 1.
#             self.action_bias = 0.
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)

#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
#         return mean

#     def sample(self, state):
#         mean = self.forward(state)
#         noise = self.noise.normal_(0., std=0.1)
#         noise = noise.clamp(-0.25, 0.25)
#         action = mean + noise
#         return action, torch.tensor(0.), mean

#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         return super(GaussianPolicy, self).to(device)


