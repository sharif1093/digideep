import numpy as np
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from digideep.memory.sampler import sampler_ff, sampler_rn
# from digideep.memory.sampler import check_shape

from digideep.utility.toolbox import get_class
from digideep.utility.logging import logger
from digideep.utility.profiling import KeepTime
from digideep.utility.monitoring import monitor

from .base import AgentBase


# def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
#     """Decreases the learning rate linearly"""
#     lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


class PPO(AgentBase):
    """The implementation of the Proximal Policy Optimization (`PPO <https://arxiv.org/abs/1707.06347>`_) method.

    Args:
        name: The name of the agent
        type: The type of the agent.
        methodargs: The arguments of the PPO method.
        sampler (dict): Sampler arguments.
        policyname (str): The name of the policy used.
        policyargs (dict): Arguments for the policy.
        optimname (str): The name of the optimizer used.
        optimargs (dict): The parameters for the optimizer.
    
    The elements in the ``methodargs`` are:

    * ``n_steps``: Number of steps to run the simulation before each update.
    * ``n_update``: Number of times to perform PPO step.
    * ``clip_param``: PPO clip parameter
    * ``value_loss_coef``: Value loss coefficient
    * ``entropy_coef``: Entropy term coefficient
    * ``max_grad_norm``: Max norm of gradients
    * ``use_clipped_value_loss``: Whether to clip value loss or not.
        
    """
    def __init__(self, session, memory, **params):
        super(PPO, self).__init__(session, memory, **params)

        self.device = self.session.get_device()

        # Set the model
        policyclass = get_class(self.params["policyname"])
        self.policy = policyclass(device=self.device, **self.params["policyargs"])
        
        # Set the optimizer (+ schedulers if any)
        optimclass = get_class(self.params["optimname"])
        self.optimizer = optimclass(self.policy.model.parameters(),  **self.params["optimargs"])

        self.state["i_step"] = 0

    ###############
    ## SAVE/LOAD ##
    ###############
    def state_dict(self):
        return {'policy':self.policy.model.state_dict()}
    def load_state_dict(self, state_dict):
        self.policy.model.load_state_dict(state_dict['policy'])
    ############################################################

    def reset_hidden_state(self, num_workers):
        hidden_size = self.params["policyargs"]["modelargs"]["output_size"]
        return np.zeros((num_workers, hidden_size), dtype=np.float32)
    def action_generator(self, observations, hidden_state, masks, deterministic=False):
        """The function that is called by :class:`~digideep.environment.explorer.Explorer`.

        Args:
            deterministic (bool): If ``True``, the best action from the optimal action will be computed. If ``False``,
                the action will be sampled from the action probability distribution.

        Returns:
            dict: ``{"actions":...,"hidden_state":...,"artifacts":{"values":...,"action_log_p":...}}``
        
        """

        with KeepTime("/explore/step/prestep/gen_action/to_torch"):
            observations_ = torch.from_numpy(observations).to(self.device)
            hidden_state_ = torch.from_numpy(hidden_state).to(self.device)
            masks_ = torch.from_numpy(masks).to(self.device)
        
        with KeepTime("/explore/step/prestep/gen_action/compute_func"):
            values, action, action_log_p, hidden_state_ = \
                self.policy.generate_actions(observations_, hidden_state_, masks_, deterministic=deterministic)
        
        with KeepTime("/explore/step/prestep/gen_action/to_numpy"):
            artifacts = dict(values=values.cpu().data.numpy(),
                             action_log_p=action_log_p.cpu().data.numpy())
            
            # actions and hidden_state is something every agent should produce.
            results = dict(actions=action.cpu().data.numpy(),
                           hidden_state=hidden_state_.cpu().data.numpy(),
                           artifacts=artifacts)
        return results


    def step(self):
        """This function is inspired by `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr>`_.

        This function needs the following keys to be in the input batch:

        * ``/observations``
        * ``/masks``
        * ``/agents/agent_name/hidden_state``
        * ``/agents/<agent_name>/actions``
        * ``/agents/<agent_name>/artifacts/action_log_p``
        * ``/agents/<agent_name>/artifacts/values``
        * ``/agents/<agent_name>/artifacts/advantages``
        * ``/agents/<agent_name>/artifacts/returns``

        The last two keys are added by the :mod:`digideep.memory.sampler`, while the rest are added at
        :class:`~digideep.environment.explorer.Explorer`.

        """
        with KeepTime("/update/step/samples"):
            info = deepcopy(self.params["sampler"])
            if self.policy.is_recurrent:
                data_sampler = sampler_rn(data=self.memory, info=info)
            else:
                data_sampler = sampler_ff(data=self.memory, info=info)

        with KeepTime("/update/step/batches"):
            for batch in data_sampler:
                with KeepTime("/update/step/batches/to_torch"):
                    # Environment
                    observations = torch.from_numpy(batch["/observations"]).to(self.device)
                    masks = torch.from_numpy(batch["/masks"]).to(self.device)
                    # Agent
                    hidden_state = torch.from_numpy(batch["/agents/"+self.params["name"]+"/hidden_state"]).to(self.device)
                    actions = torch.from_numpy(batch["/agents/"+self.params["name"]+"/actions"]).to(self.device)
                    # Agent Artifacts
                    old_action_log_p = torch.from_numpy(batch["/agents/"+self.params["name"]+"/artifacts/action_log_p"]).to(device=self.device)
                    value_preds  = torch.from_numpy(batch["/agents/"+self.params["name"]+"/artifacts/values"]).to(self.device)
                    advantages = torch.from_numpy(batch["/agents/"+self.params["name"]+"/artifacts/advantages"]).to(self.device)
                    returns = torch.from_numpy(batch["/agents/"+self.params["name"]+"/artifacts/returns"]).to(self.device)

                with KeepTime("/update/step/batches/eval_action"):
                    values, action_log_p, dist_entropy, _ = \
                        self.policy.evaluate_actions(observations,
                                                        hidden_state,
                                                        masks,
                                                        actions)

                with KeepTime("/update/step/batches/loss_function"):
                    ratio = torch.exp(action_log_p - old_action_log_p)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.params["methodargs"]["clip_param"],
                                            1.0 + self.params["methodargs"]["clip_param"]) * advantages
                    action_loss = -torch.min(surr1, surr2).mean()

                    if self.params["methodargs"]["use_clipped_value_loss"]:
                        value_pred_clipped = value_preds + \
                            (values - value_preds).clamp(-self.params["methodargs"]["clip_param"], self.params["methodargs"]["clip_param"])
                        value_losses = (values - returns).pow(2)
                        value_losses_clipped = (value_pred_clipped - returns).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        # value_loss = 0.5 * F.mse_loss(returns, values)
                        value_loss = 0.5 * (returns - values).pow(2).mean()
                    
                    Loss = value_loss * self.params["methodargs"]["value_loss_coef"] \
                        + action_loss \
                        - dist_entropy * self.params["methodargs"]["entropy_coef"]
                
                with KeepTime("/update/step/batches/backprop"):
                    self.optimizer.zero_grad()
                    Loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.model.parameters(),
                                            self.params["methodargs"]["max_grad_norm"])
                
                with KeepTime("/update/step/batches/optimstep"):
                    self.optimizer.step()

                # Monitoring values
                monitor("/update/loss", Loss.item())
                monitor("/update/value_loss", value_loss.item())
                monitor("/update/action_loss", action_loss.item())
                monitor("/update/dist_entropy", dist_entropy.item())
                
                ## Candidates for monitoring
                # ratio.item()
        self.state["i_step"] += 1

    def update(self):
        # Update the networks for n times
        for i in range(self.params["methodargs"]["n_update"]):
            self.step()
        

        
