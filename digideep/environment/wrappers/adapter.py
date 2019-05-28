import numpy as np
from collections import OrderedDict

import gym
from gym import spaces
from digideep.environment.common.vec_env import VecEnvWrapper


from digideep.environment.data_helpers import list_of_dicts_to_flattened_dict_of_lists, flattened_dict_of_lists_to_dict_of_numpy
from digideep.environment.data_helpers import dict_of_lists_to_list_of_dicts
from digideep.environment.data_helpers import flatten_dict, unflatten_dict, join_keys


def _flatten_space(s, sep="/", prefix=""):
    res = OrderedDict()
    for key, value in s.spaces.items():
        if isinstance(value, spaces.Dict):
            tmp = _flatten_space(value, sep=sep, prefix=join_keys(prefix,key,sep))
            res.update(tmp)
        else:
            res[join_keys(prefix,key,sep)] = value
    return res

##############################################################################
### Wrappers that change the list of dicts to dict of lists and vice versa ###
##############################################################################

class VecObsRewInfoActWrapper(VecEnvWrapper):
    """
    This environment converts list of dicts (which appear in the output obs and infos
    of the VecEnvs) to dict of lists (which is more favorable for programming). Also,
    it converts the dict of lists in the input actions to list of dicts favorable for
    VecEnv.
    """
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        # NOTE: This wrapper also flattens the observation space (nested observation spaces will be flattened).
        self.observation_space = spaces.Dict(_flatten_space(self.observation_space))
    
    def step_async(self, actions):
        # NOTE: No assumption is put here on the dict to be flattened or not.
        actions = dict_of_lists_to_list_of_dicts(actions, self.num_envs)
        self.venv.step_async(actions)

    def step_wait(self):
        # Here obs is a list of Dicts
        obs, rew, dones, infos = self.venv.step_wait()
        obs = self._convert(obs)
        infos = self._convert(infos)
        rew = rew.reshape((-1,1))
        return obs, rew, dones, infos
    
    def _convert(self, list_of_dicts):
        # assert len(list_of_dicts) == self.num_envs
        dict_of_lists = list_of_dicts_to_flattened_dict_of_lists(list_of_dicts, length=self.num_envs)
        result = flattened_dict_of_lists_to_dict_of_numpy(dict_of_lists)
        return result
        
    def reset(self):
        obs = self.venv.reset()
        return self._convert(obs)
##############################################################################


##########################################
### Wrappers that flatten a Dict space ###
##########################################

## NOTE: Do we need such a wrapper, given that it is easier for agents to work with nested action
## dicts rather than flattened actions?

# class WrapperFlattenActDict(gym.ActionWrapper):
#     """
#     This wrapper assumes a Dict action space, and then flattens the space (unflattens the actions.)
#     """
#     def __init__(self, env):
#         super(WrapperFlattenActDict, self).__init__(env)
#         self.action_space = spaces.Dict(_flatten_space(self.action_space))
#     def action(self, action):
#         return unflatten_dict(action)


class WrapperFlattenObsDict(gym.ObservationWrapper):
    """
    This wrapper assumes a Dict observation space, and then flattens that.
    """
    def __init__(self, env):
        super(WrapperFlattenObsDict, self).__init__(env)
        self.observation_space = spaces.Dict(_flatten_space(self.observation_space))

    def observation(self, observation):
        return flatten_dict(observation)
##########################################




##########################################
### Dummy wrappers to adapt Box fields ###
##########################################
class WrapperDummyMultiAgent(gym.ActionWrapper):
    """
    This wrapper is required to make the regular classes compatible with
    multi-agent architecture of Digideep. Use this wrapper for each
    environment that is not multi-agent.
    """
    def __init__(self, env, agent_name="agent"):
        print("WrapperDummyMultiAgent is called.")
        # We do not assume a path-like key for the Dummy wrappers.
        super(WrapperDummyMultiAgent, self).__init__(env)
        self.agent_name = agent_name
        act_space = self.action_space
        self.action_space = spaces.Dict({self.agent_name:act_space})

    def action(self, action):
        assert isinstance(action, dict), "The provided action is not a dictionary."
        assert self.agent_name in action, "The provided agent name ({}) does not exist in the action.".format(self.agent_name)
        return action[self.agent_name]

class WrapperDummyDictObs(gym.ObservationWrapper):
    """
    This wrapper is required to make the regular classes compatible with
    multi-agent architecture of Digideep. Use this wrapper for each
    environment that is not multi-agent.
    """
    def __init__(self, env, observation_key="agent"):
        print("WrapperDummyDictObs is called.")
        # We do not assume a path-like key for the Dummy wrappers.
        super(WrapperDummyDictObs, self).__init__(env)
        self.observation_key = observation_key
        
        obs_space = self.observation_space
        self.observation_space = spaces.Dict({self.observation_key:obs_space})

    def observation(self, observation):
        # assert isinstance(observation, dict), "The provided action is not a dictionary."
        # assert self.agent_name in action, "The provided agent name ({}) does not exist in the action.".format(self.agent_name)
        obs = OrderedDict()
        obs[self.observation_key] = observation
        return obs
##########################################


