"""This module is highly inspired by `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr>`__.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .policy_utils import Bernoulli, Categorical, DiagGaussian
from .policy_utils import init_easy, init_rnn
from .policy_utils import MLPBlock, RNNBlock, CNNBlock

from digideep.agent.policy_base import PolicyBase

from digideep.utility.toolbox import get_class #, get_module
from digideep.utility.logging import logger

class Policy(PolicyBase):
    """The stochastic policy to be used with PPO algorithm. This policy supports three different action
    distributions:

    * ``Categorical``: For ``gym.spaces.Discrete`` action spaces.
    * ``DiagGaussian``: For ``gym.spaces.Box`` action spaces.
    * ``Bernoulli``: For ``gym.spaces.MultiBinary`` action spaces.

    Args:
        device: The device for the PyTorch computations. Either of CPU or GPU.
        obs_space: Observation space of the environment.
        act_space: Action space of the environment.
        modelname (str): The model to be used within the policy. CURRENTLY THIS OPTION IS NOT USED INSIDE THE CLASS
            AND MODEL IS DECIDED BY THE SHAPE OF OBSERVATION SPACE.
        modelargs (dict): A dictionary of arguments for the model.
    
    """
    def __init__(self, device, obs_space, act_space, modelname, modelargs):
        super(Policy, self).__init__(device)

        self.recurrent = modelargs["recurrent"]
        #######
        # modelclass = get_class(modelname)
        #######
        if len(obs_space["dim"]) == 3: # It means we have images as observations
            # obs_space["dim"][0] is the channels of the input
            self.model["base"] = CNNModel(num_inputs=obs_space["dim"][0], **modelargs)
        elif len(obs_space["dim"]) == 1: # It means we have vectors as observation
            self.model["base"] = MLPModel(num_inputs=obs_space["dim"][0], **modelargs)
        else:
            raise NotImplementedError

        # TODO: For discrete actions, `act_space["dim"][0]` works. It works for constinuous actions as well.
        #       Even for discrete actions `np.isscalar(act_space["dim"])` returns False.
        num_outputs = act_space["dim"] if np.isscalar(act_space["dim"]) else act_space["dim"][0]
        # num_outputs = act_space["dim"].item() if len(act_space["dim"].shape)==0 else act_space["dim"][0]
        if act_space["typ"] == "Discrete":
            print("Discrete is recognized and num_outputs=", num_outputs)
            self.model["dist"] = Categorical(num_inputs=modelargs["output_size"], num_outputs=num_outputs)
        elif act_space["typ"] == "Box":
            self.model["dist"] = DiagGaussian(num_inputs=modelargs["output_size"], num_outputs=num_outputs)
        elif act_space["typ"] == "MultiBinary":
            # TODO: Is the following necessary?
            num_outputs = act_space["dim"][0]
            self.model["dist"] = Bernoulli(num_inputs=modelargs["output_size"], num_outputs=num_outputs)
        else:
            raise NotImplementedError("The action_space of the environment is not supported!")
        
        self.model_to_gpu()
        logger("Number of parameters:\n>>>>>>", self.count_parameters())


    def generate_actions(self, inputs, hidden, masks, deterministic=False):
        """This function is used by :func:`~digideep.agent.ppo.PPO.action_generator` to generate the actions while simulating in the environments.
        
        Args:
            inputs: The observations.
            hidden: The hidden states of the policy models.
            masks: The masks indicates the status of the environment in the last state, either it was finished ``0``, or still continues ``1``.
            deterministic (bool): The flag indicationg whether to sample from the action distribution (when ``False``) or choose the best (when ``True``).
        """
        with torch.no_grad():
            self.model.eval()
            value, actor_features, hidden_ = self.model["base"](inputs, hidden, masks)
            dist = self.model["dist"](actor_features)
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            action_log_probs = dist.log_probs(action)
            # dist_entropy = dist.entropy().mean() # NOTE: Why do we calculate it here? Probably it's not a big deal!
            # return value.detach(), action.detach(), action_log_probs.detach(), hidden.detach()
            self.model.train()
            return value, action, action_log_probs, hidden_

    def evaluate_actions(self, inputs, hidden, masks, action):
        """Evaluates a given action in the PPO method.

        Args:
            inputs: The observations.
            hidden: The hidden states of the policy models.
            masks: The masks indicates the status of the environment in the last state, either it was finished ``0``,
              or still continues ``1``.
            action: The actions to be evaluated with the current policy model.
        """

        value, actor_features, hidden = self.model["base"](inputs, hidden, masks)
        dist = self.model["dist"](actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, hidden

    @property
    def is_recurrent(self):
        """Indicates whether or not the policy uses a recurrent policy in its base method.
        """
        return self.recurrent


##############################################################
########################### MODELS ###########################
##############################################################
class MLPModel(nn.Module):
    """An MLP model of an actor-critic. It may use a recurrent unit or not.
    
    Args:
        num_inputs: The dimension of the input observation.
        output_size: The dimension of the output action feature vector.
        recurrent (bool): An indicator whether or not a base recurrent module.
    """

    def __init__(self, num_inputs, output_size, recurrent=False):
        super(MLPModel, self).__init__()

        self.recurrent = recurrent
        if recurrent:
            self.rnnblock = RNNBlock(num_inputs=num_inputs, hidden_size=output_size)
            num_inputs = output_size

        self.actor  = MLPBlock(num_inputs=num_inputs, output_size=output_size)
        self.critic = MLPBlock(num_inputs=num_inputs, output_size=output_size)
        
        init_ = init_easy(gain=np.sqrt(2), bias=0)
        self.critic_linear = init_(nn.Linear(output_size, 1))
    
    def forward(self, inputs, hidden, masks):
        """
        Args:
            inputs (:obj:`torch.Tensor`): The input to the model that includes the observations.
            hidden (:obj:`torch.Tensor`): The hidden state of last step used in recurrent policies.
            masks (:obj:`torch.Tensor`): The mask indicator to be used with recurrent policies.
        
        Returns:
            tuple: A tuple of ``(values, feature_actor, hidden)``, which are action-value, the features that
            will form the action distribution probability, and the hidden state of the recurrent unit.
        """

        x = inputs
        if self.recurrent:
            x, hidden = self.rnnblock(x, hidden, masks)
        
        hidden_critic = self.critic(x)
        feature_actor = self.actor(x)

        values = self.critic_linear(hidden_critic)
        return values, feature_actor, hidden



class CNNModel(nn.Module):
    """
    A CNN model of an actor-critic model. It may use a recurrent unit or not.

    Args:
        num_inputs: The dimension of the input observation.
        output_size: The dimension of the output action feature vector.
        recurrent (bool): An indicator whether or not a base recurrent module.
    """

    def __init__(self, num_inputs, output_size, recurrent=False):
        super(CNNModel, self).__init__()

        self.cnnblock = CNNBlock(num_inputs= num_inputs, output_size=output_size)
        
        self.recurrent = recurrent
        if recurrent:
            self.rnnblock = RNNBlock(num_inputs=output_size, hidden_size=output_size)

        init_ = init_easy(gain=1, bias=0)
        self.critic_linear = init_(nn.Linear(output_size, 1))

        self.train() # TODO: Is it necessary????

    def forward(self, inputs, hidden, masks):
        """
        Args:
            inputs (:obj:`torch.Tensor`): The observations tensor. The observation must be an image.
            hidden (:obj:`torch.Tensor`): The hidden state of last step used in recurrent policies.
            masks (:obj:`torch.Tensor`): The mask indicator to be used with recurrent policies.
        
        Returns:
            tuple: A tuple of ``(values, feature_actor, hidden)``.
        """

        x = self.cnnblock(inputs/255.) # Normalizing inputs
        
        if self.recurrent:
            x, hidden = self.rnnblock(x, hidden, masks)
        
        values = self.critic_linear(x)

        feature_actor = x

        return values, feature_actor, hidden
