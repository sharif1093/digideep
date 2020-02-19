"""
This implementation is mainly adopted from `pytorch-soft-actor-critic <https://github.com/pranz24/pytorch-soft-actor-critic>`__.
"""

import numpy as np
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from digideep.utility.toolbox import get_class
from digideep.utility.logging import logger
from digideep.utility.profiling import KeepTime
from digideep.utility.monitoring import monitor

# from digideep.agent.samplers.ddpg import sampler_re
from digideep.agent.sampler_common import Compose
from digideep.agent.agent_base import AgentBase
from .policy import Policy

# torch.utils.backcompat.broadcast_warning.enabled = True

class Agent(AgentBase):
    """This is an implementation of the Soft Actor Critic (`SAC <https://arxiv.org/abs/1801.01290>`_) method.
    Here the modified version of `SAC https://arxiv.org/abs/1812.05905`_ is not considered.
    
    Args:
        name: The agent's name.
        type: The type of this class which is ``digideep.agent.SAC``.
        methodargs (dict): The parameters of the SAC method.
        sampler:
        
        
    #     policyname: The name of the policy which can be ``digideep.agent.policy.soft_stochastic.Policy`` for normal SAC.
    #     policyargs: The arguments for the policy.
    #     noisename: The noise model name.
    #     noiseargs: The noise model arguments.
    #     optimname: The name of the optimizer.
    #     optimargs: The arguments of the optimizer.
        
    # The elements in the ``methodargs`` are:

    # * ``n_update``: Number of times to perform SAC step.
    # * ``gamma``: Discount factor :math:`\gamma`.
    # * ``clamp_return``: The clamp factor. One option is :math:`1/(1-\gamma)`.
    

    """

    def __init__(self, session, memory, **params):
        super(Agent, self).__init__(session, memory, **params)

        self.device = self.session.get_device()

        # policy_type: Gaussian | Deterministic. Only "Gaussian" for now.

        # Set the Policy
        # policyclass = get_class(self.params["policyname"])
        self.policy = Policy(device=self.device, **self.params["policyargs"])
        
        # Set the optimizer (+ schedulers if any)
        optimclass_critic = get_class(self.params["optimname_critic"])
        optimclass_actor  = get_class(self.params["optimname_actor"])
        
        self.optimizer = {}
        self.optimizer["critic1"] = optimclass_critic(self.policy.model["critic1"].parameters(), **self.params["optimargs_critic"])
        self.optimizer["critic2"] = optimclass_critic(self.policy.model["critic2"].parameters(), **self.params["optimargs_critic"])
        self.optimizer["actor"]   = optimclass_actor(self.policy.model["actor"].parameters(), **self.params["optimargs_actor"])

        self.criterion = {}
        # self.criterion["critic"] = nn.MSELoss()
        # self.criterion["actor"]  = nn.MSELoss()
        
        # Build the sampler from sampler list:
        sampler_list = [get_class(k) for k in self.params["sampler_list"]]
        self.sampler = Compose(sampler_list)

        # if self.params["methodargs"]["automatic_entropy_tuning"] == True:
        #     # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        #     self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        #     self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        #     self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        # noiseclass = get_class(self.params["noisename"])
        # self.noise = noiseclass(**self.params["noiseargs"])

        self.state["i_step"] = 0



    # if self.policy_type == "Gaussian":
    # elif "Deterministic":
    #     self.alpha = 0
    #     self.automatic_entropy_tuning = False






    ###############
    ## SAVE/LOAD ##
    ###############
    # TODO: Also states of optimizers, noise, etc.
    def state_dict(self):
        return {'state':self.state, 'policy':self.policy.model.state_dict()}
    def load_state_dict(self, state_dict):
        self.policy.model.load_state_dict(state_dict['policy'])
        self.state.update(state_dict['state'])
    ############################################################
    
    def action_generator(self, observations, hidden_state, masks, deterministic=False):
        """This function computes the action based on observation, and adds noise to it if demanded.

        Args:
            deterministic (bool): If ``True``, the output would be merely the output from the actor network.
            Otherwise, noise will be added to the output actions.
        
        Returns:
            dict: ``{"actions":...,"hidden_state":...}``

        """
        observation_path = self.params.get("observation_path", "/agent")
        observations_ = observations[observation_path].astype(np.float32)
        
        observations_ = torch.from_numpy(observations_).to(self.device)
        action = self.policy.generate_actions(observations_, deterministic=deterministic)
        action = action.cpu().numpy()

        # if not deterministic:
        #     action = self.noise(action)

        results = dict(actions=action, hidden_state=hidden_state)
        return results


    def update(self):
        # Update the networks for n times
        for i in range(self.params["methodargs"]["n_update"]):
            with KeepTime("step"):
                self.step()
        
        with KeepTime("targets"):
            # Update value target
            for key in self.policy.averager:
                self.policy.averager[key].update_target()
        
        # ## For debugging
        # # for p, ptar in zip(self.policy.model["actor"].parameters(), self.policy.model["actor_target"].parameters()):
        # #     print(p.mean(), ptar.mean())
    
        # # for p, ptar in zip(self.policy.model["actor"].parameters(), self.policy.model["critic"].parameters()):
        # #     print(p.mean(), ptar.mean())


    def step(self):
        """This function needs the following key values in the batch of memory:

        * ``/observations``
        * ``/rewards``
        * ``/agents/<agent_name>/actions``
        * ``/observations_2``

        The first three keys are generated by the :class:`~digideep.environment.explorer.Explorer`
        and the last key is added by the sampler.
        """
        alpha = self.params["methodargs"]["alpha"]
        gamma = self.params["methodargs"]["gamma"]


        with KeepTime("sampler"):
            info = deepcopy(self.params["sampler_args"])
            batch = self.sampler(data=self.memory, info=info)
            if batch is None:
                return

        with KeepTime("loss"):
            with KeepTime("to_torch"):
                # ['/obs_with_key', '/masks', '/agents/agent/actions', '/agents/agent/hidden_state', '/rewards', '/obs_with_key_2', ...]
                state      = torch.from_numpy(batch["/observations"+ self.params["observation_path"]]).to(self.device).float()
                action     = torch.from_numpy(batch["/agents/"+self.params["name"]+"/actions"]).to(self.device).float()
                reward     = torch.from_numpy(batch["/rewards"]).to(self.device).float()
                next_state = torch.from_numpy(batch["/observations"+self.params["observation_path"]+"_2"]).to(self.device).float()
                masks      = torch.from_numpy(batch["/masks"]).to(self.device)
                # masks      = torch.from_numpy(batch["/masks"]).to(self.device).view(-1)

            #     reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            #     masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_state_log_prob = self.policy.evaluate_actions(next_state)
                qf1_next_target = self.policy.model["critic1_target"](next_state, next_state_action)
                qf2_next_target = self.policy.model["critic2_target"](next_state, next_state_action)


                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_prob
                next_q_value = reward + masks * gamma * (min_qf_next_target)


            # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1 = self.policy.model["critic1"](state, action)
            qf2 = self.policy.model["critic2"](state, action)

            # JQ = 𝔼(st,at)~D[0.5(Q(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            
            pi, log_pi = self.policy.evaluate_actions(state)

            qf1_pi = self.policy.model["critic1"](state, pi)
            qf2_pi = self.policy.model["critic2"](state, pi)

            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

            
            self.optimizer["critic1"].zero_grad()
            qf1_loss.backward()
            self.optimizer["critic1"].step()

            self.optimizer["critic2"].zero_grad()
            qf2_loss.backward()
            self.optimizer["critic2"].step()

            self.optimizer["actor"].zero_grad()
            actor_loss.backward()
            self.optimizer["actor"].step()

        #     if self.automatic_entropy_tuning:
        #         alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        #         self.alpha_optim.zero_grad()
        #         alpha_loss.backward()
        #         self.alpha_optim.step()

        #         self.alpha = self.log_alpha.exp()
        #         alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        #     else:
        #         alpha_loss = torch.tensor(0.).to(self.device)
        #         alpha_tlogs = torch.tensor(alpha) # For TensorboardX logs


        monitor("/update/loss/actor", actor_loss.item())
        monitor("/update/loss/critic1", qf1_loss.item())
        monitor("/update/loss/critic2", qf2_loss.item())

        self.session.writer.add_scalar('loss/actor', actor_loss.item())
        self.session.writer.add_scalar('loss/critic1', qf1_loss.item())
        self.session.writer.add_scalar('loss/critic2', qf2_loss.item())

        # 'loss/entropy_loss', ent_loss:        alpha_loss.item()
        # 'entropy_temprature/alpha', alpha:    alpha_tlogs.item()
        self.state["i_step"] += 1




        ###################
        ### OLD VERSION ###
        ###################
        # expected_q_value = self.policy.model["softq"](state, action)
        # expected_value = self.policy.model["value"](state)
        # new_action, log_prob, z, mean, log_std = self.policy.evaluate_actions(state)

        # target_value = self.policy.model["value_target"](next_state)
        # next_q_value = reward + masks * float(self.params["methodargs"]["gamma"]) * target_value
        # softq_loss = self.criterion["softq"](expected_q_value, next_q_value.detach())

        # expected_new_q_value = self.policy.model["softq"](state, new_action)
        # next_value = expected_new_q_value - log_prob
        # value_loss = self.criterion["value"](expected_value, next_value.detach())

        # log_prob_target = expected_new_q_value - expected_value
        # actor_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        
        # mean_loss = float(self.params["methodargs"]["mean_lambda"]) * mean.pow(2).mean()
        # std_loss  = float(self.params["methodargs"]["std_lambda"])  * log_std.pow(2).mean()
        # z_loss    = float(self.params["methodargs"]["z_lambda"])    * z.pow(2).sum(1).mean()

        # actor_loss += mean_loss + std_loss + z_loss

        # self.optimizer["softq"].zero_grad()
        # softq_loss.backward()
        # self.optimizer["softq"].step()

        # self.optimizer["value"].zero_grad()
        # value_loss.backward()
        # self.optimizer["value"].step()

        # self.optimizer["actor"].zero_grad()
        # actor_loss.backward()
        # self.optimizer["actor"].step()

