import gym
import numpy as np

from gym import spaces
from collections import OrderedDict


class WrapperLevelDictObs(gym.ObservationWrapper):
    """
    The code is inspired from ``dm_control``.
    We assume flattened observation Dict in this wrapper.

    Input: spaces.Dict({"/obs1/pos":spaces.Box, "/obs1/vel":spaces.Box, "/obs2/image1":spaces.Box, "/obs2/sensor2":spaces.Discrete})
    Path: "/obs1"
    Output: spaces.Dict({"/obs1":spaces.Box, "/obs2/image1":spaces.Box, "/obs2/sensor2":spaces.Discrete})
    """
    def __init__(self, env, mode, path):
        super(WrapperLevelDictObs, self).__init__(env)
        self.path = path
        self.keys = []

        self.observation_space = self._level_observation_space(self.observation_space)

    def _level_observation_space(self, observation_space):
        size = 0
        res_space = OrderedDict()
        for key, space in observation_space.spaces.items():
            if key.startswith(self.path+"/"):
                # Reserve this space in the correct order the first time we catch an instance.
                if self.path not in res_space:
                    res_space[self.path] = None
                self.keys += [key]
                size += self._get_size_space(space)
            else:
                res_space[key] = space
        
        # TODO: A better way would be to ravel the low and high tensors and concatenate them.
        low  = np.full(shape=(size,), fill_value=-np.inf, dtype=np.float32)
        high = np.full(shape=(size,), fill_value=+np.inf, dtype=np.float32)
        res_space[self.path] = spaces.Box(low=low, high=high, dtype=np.float32)
        output_observation_space = spaces.Dict(res_space)
        return output_observation_space
        
    def _get_size_space(self, space):
        size = 1
        for num in space.shape:
            size *= num
        return size
    
    def _level_observation(self, observation):
        obs = OrderedDict()

        # Create an empty Dict with the true order.
        for key in self.observation_space.spaces:
            obs[key] = None
        # Copy non-involved keys
        for key in observation:
            if not key in self.keys:
                obs[key] = observation[key]
        # Ravel the involved keys
        observation_arrays = [observation[key].ravel() for key in self.keys]
        obs[self.path] = np.concatenate(observation_arrays)
        return obs

    def observation(self, observation):
        return self._level_observation(observation)


########################################

class WrapperTransposeImage(gym.ObservationWrapper):
    """
    Inspired by `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py>`__.
    
    If the input has shape ``(W,H,3)``, the output will become ``(3,W,H)``, which is suitable for PyTorch convolution layers.
    """
    def __init__(self, env, mode, path, op=[2, 0, 1]):
        super(WrapperTransposeImage, self).__init__(env)
        
        # assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        assert len(op) == 3, "Error: Operation, {}, must be dim3".format(str(op))

        self.path = path
        self.op = op

        obs_shape = self.observation_space.spaces[self.path].shape

        # Only change the observation_space related to the given path.
        self.observation_space.spaces[self.path] = spaces.Box(
            self.observation_space.spaces[self.path].low[0, 0, 0],
            self.observation_space.spaces[self.path].high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.spaces[self.path].dtype)
        
    def observation(self, observation):
        obs = OrderedDict()
        for key in observation:
            if key == self.path:
                obs[key] = observation[key].transpose(self.op[0], self.op[1], self.op[2])
            else:
                obs[key] = observation[key]
        return obs












# ######################################
# ### Base Classes ###
# ####################
### Normal wrappers
# class BaseObservationWrapperDict(gym.ObservationWrapper):
#     def __init__(self, env, keys_list):
#         self.keys_list = keys_list
#         for key in self.observation_space.spaces.keys():
#             if key in self.keys_list:
#                 self.observation_space.spaces[key] = self._modify_observation_space(self.observation_space.spaces[key])    
#     def observation(self, observation):
#         for key in self.observation_space.spaces.keys():
#             if key in self.keys_list:
#                 observation[key] = self._modify_observation(observation[key])
#         return observation
#     def _modify_observation_space(self, observation_space):
#         raise NotImplementedError
#     def _modify_observation(self, observation):
#         raise NotImplementedError
# ######################################

# ###################
# ## Example Class ##
# ###################
# class WrapperAddConstantDict(BaseObservationWrapperDict):
#     def __init__(self, env, mode, keys_list, constant):
#         super(WrapperAddConstantDict, self).__init__(env, keys_list)
#         self.constant = constant

#     def _modify_observation_space(self, observation_space):
#         obs_space = spaces.Box(observation_space.low[0],
#                                observation_space.high[0],
#                                [observation_space.shape[0] + 1],
#                                dtype=observation_space.dtype)
#         return obs_space
    
#     def _modify_observation(self, observation):
#         return np.concatenate((observation, [self.constant]))


##################################################
















################################################
# This adds the timestep as an observation to the environment. With this the agent will know the time.
# if self.params["wrappers"]["add_time_step"]:
#     assert len(env.observation_space.shape) == 1, "AddTimeStep supports 1d observations."
#     assert str(env).find('TimeLimit') > -1, "AddTimeStep need environment to be a TimeLimit."
#     # NOTE: dm_control environments will not be wrapped by the TimeLimit environment.
#     env = WrapperAddTimestep(env)

# if self.params["wrappers"]["add_image_transpose"]:
#     obs_shape = env.observation_space.shape
#     assert len(obs_shape) == 3
#     assert obs_shape[2] in [1, 3]
#     env = WrapperTransposeImage(env)

# if self.params["wrappers"]["add_normalized_actions"]:
#     env = WrapperNormalizedActions(env)
################################################



# class WrapperMaskGoal(gym.ObservationWrapper):
#     """
#     Adopted from `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py>`__.
#     """

#     def __init__(self, env, nmask):
#         super(WrapperMaskGoal, self).__init__(env)
#         self.nmask = nmask
#     def observation(self, observation):
#         if self.env._elapsed_steps > 0:
#             observation[-self.nmask:] = 0
#         return observation




# norm_wrappers.append(dict(name="digideep.environment.wrappers.normal.WrapperAddTimestep",
#                           args={},
#                           enabled=False))

# class WrapperAddTimestep(gym.ObservationWrapper):
#     """
#     Adopted from `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py>`__.
#     """
#     def __init__(self, env=None):
#         super(WrapperAddTimestep, self).__init__(env)
#         self.observation_space = spaces.Box(
#             self.observation_space.low[0],
#             self.observation_space.high[0],
#             [self.observation_space.shape[0] + 1],
#             dtype=self.observation_space.dtype)
#     def observation(self, observation):
#         return np.concatenate((observation, [self.env._elapsed_steps]))

