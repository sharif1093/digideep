from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np
from digideep.utility.logging import logger

class PolicyBase(object):
    """The base class for all policy classes. Policy is a model inside the agent which generates
    actions. A policy must implement :func:`generate_actions` function to generate actions based
    on its own architecture.

    Args:
        device: The target device for the model (i.e. CPU or GPU).
    
    Attributes:
        model (:obj:`nn.ModuleDict`): A dictionary containing all different models of the policy.
        This helps handling all models inside the policy to happen more easily. For instance, it
        helps in simplifying (de-)serialization of models or transferring them to GPU/CPU.

    Caution:
        "Resuming" training from a checkpoint is not fully implemented yet.
    
    """
    def __init__(self, device):
        self.device = device
        self.model = nn.ModuleDict()
    
    def post_init(self):
        self.model_to_gpu()
        logger("Number of parameters: <", self.count_parameters(), '>')
        # Summary writer of PyTorch goes here.

    def model_to_gpu(self):
        """Function to transfer ``self.model`` to the GPU. It will use PyTorch's ``nn.DataParallel``
        if there is one or more GPUs available.

        Note:
            'torch.nn.DataParallel' is a wrapper and hides many internal methods for a good reason.
        """
        # Multi-GPU
        if torch.cuda.device_count() >= 1:
            gpu_count = torch.cuda.device_count()
            for mname in self.model:
                self.model[mname] = nn.DataParallel(self.model[mname])
        else:
            gpu_count = 0
        self.model.to(self.device) # dtype=model_type

    def count_parameters(self):
        """
        Counts the number of parameters in a PyTorch model.
        """
        return np.sum(p.numel() for p in list(self.model.parameters()) if p.requires_grad)

    @abstractmethod
    def state_dict(self):
        """Returns state dict of the policy. It is ``model.state_dict`` by default, but child classes
        can override it to include more states in the dictionary.
        """
        return {'model':self.model.state_dict()}
    @abstractmethod
    def load_state_dict(self, state_dict):
        """The function to load the state dictionary on the components that have states.
        Override this function in the child class if there is more states to be loaded.
        """
        self.model.load_state_dict(state_dict['model'])
        
    
    # @abstractmethod
    def generate_actions(self, *args, deterministic=False):
        """This function will generate actions by using the policy models. This is overrided in all policies.
        
        This function should be called from :func:`~digideep.agent.base.AgentBase.action_generator` function.

        Note:
            Note the difference between this function and :func:`~digideep.agent.base.AgentBase.action_generator`.
            The former will directly work with policy models to generate actions. The latter will use this function
            with appropriate arguments, and add extra noise to the actions if necessary. For example, look at
            :func:`~digideep.agent.ddpg.DDPG.action_generator` in ``DDPG`` and :func:`~digideep.agent.ppo.PPO.action_generator`
            in ``PPO``.

        """
        raise NotImplementedError
    def evaluate_actions(self, *args):
        """This function will evaluate (for on-policy methods) or generate (for off-policy methods) an action in retrospect. 
        `generate_actions` functions works without gradients. However, gradients are important in `evaluate_actions`.
        """
        raise NotImplementedError
    
    