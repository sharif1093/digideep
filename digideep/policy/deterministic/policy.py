import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from digideep.utility.toolbox import get_class

from digideep.policy.base import PolicyBase

from copy import deepcopy

class Policy(PolicyBase):
    """Implementation of a deterministic actor-critic policy for the DDPG method.

    Args:
        obs_space: The observation space of the environment.
        act_space: The action space of the environment.
        actor_args (dict): A dictionary of arguments for the :class:`ActorModel`.
        critic_args (dict): A dictionary of arguments for the :class:`CriticModel`.
        average_args (dict): A dictionary of arguments for the :class:`Averager`.
    
    Todo:
        Override the base class ``state_dict`` and ``load_state_dict`` to also save the state of ``averager``.

    """
    def __init__(self, device, **params):
        super(Policy, self).__init__(device)

        self.params = params

        assert len(self.params["obs_space"].dim) == 1, "We only support 1d observations for the DDPG policy for now."
        assert self.params["act_space"].typ == "Box", "We only support continuous actions in DDPG policy for now."

        state_size  = self.params["obs_space"].dim[0]
        action_size = self.params["act_space"].dim if np.isscalar(self.params["act_space"].dim) else self.params["act_space"].dim[0]
        action_gain = self.params["act_space"].lim[1][0]

        self.model["actor"] = ActorModel(state_size=state_size, action_size=action_size, action_gain=action_gain, **self.params["actor_args"])
        self.model["actor_target"] = deepcopy(self.model["actor"])

        self.model["critic"] = CriticModel(state_size=state_size, action_size=action_size, **self.params["critic_args"])
        self.model["critic_target"] = deepcopy(self.model["critic"])

        self.averager = {}
        self.averager["actor"]  = Averager(self.model["actor"],  self.model["actor_target"],  **self.params["average_args"])
        self.averager["critic"] = Averager(self.model["critic"], self.model["critic_target"], **self.params["average_args"])
        
        self.model_to_gpu()
    
    def generate_actions(self, inputs, deterministic=False):
        """
        This function generates actions from the "actor" model.
        """

        with torch.no_grad():
            self.model.eval()
            action = self.model["actor"](inputs)
            # value  = self.model["critic"]()
            self.model.train()

            return action
    

class Averager:
    """A helper class to update the target models in the DDPG method. It supports a ``hard`` mode and a ``soft`` mode.

    Args:
        model: The reference model.
        model_target: The target model (which will be a lagging version of the main model for stability).
        
        mode (str): The mode of updating which can be ``soft`` | ``hard``.
        polyak_factor (float): A number in :math:`[0,1]` which indicates the forgetting factor in the ``soft`` mode.
            The smaller this value the less it will forget the previous models and so is less sensitive to noise.
        interval (int): The interval of updating the target factor in the ``hard`` mode.
    """
    def __init__(self, model, model_target, **params):
        self.model = model
        self.model_target = model_target
        self.params = params

        self.state = {}
        # This will hold the counter for the hard mode updates
        self.state['update_counter'] = 0

    def update_target(self):
        """Upon calling this function the target model will be updated based on its ``mode``.
        """
        mode = self.params["mode"]
        if mode == 'soft':
            polyak_factor = self.params["polyak_factor"]
            # y = TAU*x + (1 - TAU)*y
            for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - polyak_factor) + param.data * polyak_factor)
        elif mode == 'hard':
            self.state['update_counter'] += 1
            interval = self.params["interval"]
            if self.state['update_counter'] % interval == 0:
                self.model_target.load_state_dict(self.model.state_dict())
                self.state['update_counter'] = 0
    def state_dict(self):
        return {"state":self.state}
    def load_state_dict(self, state_dict):
        self.state.update(state_dict["state"])


def fanin_init(data, gain=None):
    size = data.size()
    fanin = gain or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# def init(module, weight_init=None, gain=1, bias_init=None, bias=0):
#     if weight_init:
#         weight_init(module.weight.data, gain=gain)
#     if bias_init:
#         bias_init(module.bias.data, bias=0)
#     return module

# def init_easy(gains=1):
#     def _f(module):
#         return init(module=module, weight_init=fanin_init, gain=gain, bias_init=None, bias=None)
#     return _f


class CriticModel(nn.Module):
    """The model for the critic in the DDPG method. The input to this model would be both the observations and actions.
    Then this model will estimate the action-values function, :math:`Q(s,a)`, based on those.

    Args:
        state_size (int): Dimension of input state.
        action_size (int): Dimension of input action.
        eps (float): The initialization range of the third layer.
    """
    def __init__(self, **params):
        super(CriticModel, self).__init__()
        self.params = params

        # init_ = init_easy()
        # self.bn1 = nn.BatchNorm1d(num_features=self.params['state_size'])

        self.fcs1 = nn.Linear(self.params['state_size'], 256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data)

        self.fcs2 = nn.Linear(256,128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data)

        self.fca1 = nn.Linear(self.params['action_size'], 128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data)

        # self.fc2 = nn.Linear(256,256)
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data)
        
        # self.fc3 = nn.Linear(256, 1)
        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data.uniform_(-self.params['eps'], self.params['eps'])

    def forward(self, state, action):
        """ Returns Value function :math:`Q(s,a)` obtained from critic network

        Args:
            state (:obj:`torch.Tensor`): Input state with shape ``(batch_size,*state_size)``
            action (:obj:`torch.Tensor`): Input Action with shape ``(batch_size,*action_size)``
        
        Returns:
            :obj:`torch.Tensor`: Action-Value function with shape ``(batch_size,1)``
        """
        # state =  self.bn1(state)

        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))

        a1 = F.relu(self.fca1(action))

        x = torch.cat((s2,a1),dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ActorModel(nn.Module):
    def __init__(self, **params):
        """
        Provides the policy function :math:`a=\pi(s)`.
        Args:
            state_size (int): Dimension of input state
            action_size (int): Dimension of output action (int)
            action_gain (float): Used to scale the output action. The output action range will be in ``[-action_gain,+action_gain]``.
            eps: A constant used in initializations.
        """
        super(ActorModel, self).__init__()
        self.params = params

        # self.bn1 = nn.BatchNorm1d(num_features=self.params['state_size'])

        self.fc1 = nn.Linear(self.params['state_size'], 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data)

        # self.fc2 = nn.Linear(256,256)
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data)

        # self.fc3 = nn.Linear(256,256)
        self.fc3 = nn.Linear(128,64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data)

        # self.fc4 = nn.Linear(256, self.params['action_size'])
        self.fc4 = nn.Linear(64, self.params['action_size'])
        self.fc4.weight.data.uniform_(-self.params['eps'], self.params['eps'])
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        Args:
            state (:obj:`torch.Tensor`): The input state in the shape: ``(batch_size,state_size)``
        Returns:
            :obj:`torch.Tensor`: The action from the actor network in the shape: ``(batch_size,action_size)``
        """
        # state = self.bn1(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        action = self.tanh(self.fc4(x))
        action = action * float(self.params['action_gain'])
        return action
