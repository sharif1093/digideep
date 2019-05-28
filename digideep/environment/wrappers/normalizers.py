import numpy as np

import gym
from gym import spaces
from digideep.environment.common.vec_env import VecEnvWrapper
from digideep.environment.common.running_mean_std import RunningMeanStd
from collections import OrderedDict

####################################################
### Normalizer ###
##################
class Normalizer(object):
    """A serializable class to form a running average of a value and to normalize it.
    
    Args:
        shape (tuple): The shape of the variable
        clip (float): Clipping parameter to clip the outputs beyond that value.
        eps (float): A regulizer to avoid division by zero.
    """
    def __init__(self, shape, clip, eps):
        # VecEnvWrapper.__init__(self, venv)
        self.eps = eps
        self.clip = clip
        self.running_mean = RunningMeanStd(shape=shape)
        
    def update(self, val):
        # Should be called only in "training" phase
        self.running_mean.update(val)
    def normalize(self, val, use_mean=True):
        if use_mean:
            mean = self.running_mean.mean
        else:
            mean = 0
        
        normalized = (val - mean) / np.sqrt(self.running_mean.var + self.eps)
        res = np.clip(normalized, -self.clip, self.clip)
        return res.astype(np.float32)
    def state_dict(self):
        return self.running_mean.state_dict()
    def load_state_dict(self, state_dict):
        self.running_mean.load_state_dict(state_dict)

###########################
### VecNormalizeObsDict ###
###########################
class VecNormalizeObsDict(VecEnvWrapper):
    """A serializable vectorized wrapper that normalizes the observations and returns from an environment.

    Inspired by `baselines <https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py>`__
    and `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py>`__.

    This wrapper does not alter the "observation_space".
    Todo: We may change the bounds of the observation space to be in [-clip, +clip].
    """
    def __init__(self, venv, mode, paths, clip=10., epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)

        self.training = True if mode == "train" else False
        self.paths = paths
        # Check all entries
        for path in self.paths:
            # Works only for vector observations, not for images
            if (not len(self.observation_space.spaces[path].shape) == 1) and (isinstance(self.observation_space.spaces[path], spaces.Box)):
                raise TypeError("VecNormalizeObsDict supports 1d (vector) Box observation spaces. Whereas it is {}d for '{}'.".format(
                                    len(self.observation_space.spaces[path].shape), path))
        
        # Create normalizers
        self.normalizers = {}
        for path in self.paths:
            self.normalizers[path] = Normalizer(shape=self.observation_space.spaces[path].shape, clip=clip, eps=epsilon)
            
            self.observation_space.spaces[path].low  = np.full_like(self.observation_space.spaces[path].low,  -clip)
            self.observation_space.spaces[path].high = np.full_like(self.observation_space.spaces[path].high, +clip)
    
    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        # Here obs is a Dict of lists.
        obs = self._obs_filt(obs)
        return obs, rew, dones, infos
    
    def reset(self):
        obs = self.venv.reset()
        return self._obs_filt(obs)
    
    def _obs_filt(self, obs):
        if self.training:
            for path in self.paths:
                self.normalizers[path].update(obs[path])
        
        # res = (type(obs))()
        res = OrderedDict()
        for path in obs:
            if path in self.paths:
                res[path] = self.normalizers[path].normalize(obs[path])
            else:
                res[path] = obs[path]
        return res

    def state_dict(self):
        states = {}
        states["normalizers"] = {}
        for path in self.paths:
            states["normalizers"][path] = self.normalizers[path].state_dict()
        return states
    def load_state_dict(self, state_dict):
        for path in self.paths:
            self.normalizers[path].load_state_dict(state_dict["normalizers"][path])

#######################
### VecNormalizeRew ###
#######################
class VecNormalizeRew(VecEnvWrapper):
    """A serializable vectorized wrapper that normalizes the observations and returns from an environment.

    Inspired by `baselines <https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py>`__
    and `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py>`__.
    """
    def __init__(self, venv, mode, clip=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.training = True if mode == "train" else False

        self.normalizer = Normalizer(shape=(), clip=clip, eps=epsilon)
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma


    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        
        rew = self._rew_filt(rew, dones)
        return obs, rew, dones, infos
    
    def _rew_filt(self, rew, dones):
        if self.training:
            self.ret = self.ret * self.gamma + rew
            self.normalizer.update(rew)
            self.ret[dones] = 0.
        return self.normalizer.normalize(rew, use_mean=False)

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return obs
    
    def state_dict(self):
        states = {}
        states["normalizer"] = self.normalizer.state_dict()
        states["ret"] = self.ret
        return states
    def load_state_dict(self, state_dict):
        self.normalizer.load_state_dict(state_dict["normalizer"])
        self.ret = state_dict["ret"]



###############################
### WrapperNormalizeActDict ###
###############################

## NOTE: This implementation assumes a 1-level (not nested nor path-like style) action space.
##       However, it can be used with a flattened action space equally well.
class WrapperNormalizeActDict(gym.ActionWrapper):
    """
    This is inspired by `RL-Adventure-2 <https://github.com/higgsfield/RL-Adventure-2>`__.

    With this wrapper, the input actions will be in the range of [0,1].

    Todo: Consider nested action spaces.
    """
    def __init__(self, env, mode, paths):
        super(WrapperNormalizeActDict, self).__init__(env)
        self.paths = paths
        
        ## Deforming the action_space
        self.pre_bounds = type(self.action_space.spaces)()
        for key in self.action_space.spaces:
            if key in self.paths:
                # [0,1]
                low = self.action_space.spaces[key].low
                high = self.action_space.spaces[key].high
                self.pre_bounds[key] = (low.copy(), high.copy())
                
                self.action_space.spaces[key].low = np.zeros_like(low)
                self.action_space.spaces[key].high = np.ones_like(high)
    
    def action(self, action):
        # result = type(action)()
        result = OrderedDict()
        for key in self.action_space.spaces:
            if key in self.paths:
                low, high = self.pre_bounds[key]
                
                result[key] = low + (action[key] + 1.0) * 0.5 * (high - low)
                result[key] = np.clip(result[key], low, high)
            else:
                result[key] = action[key]
        return result

    # def reverse_action(self, action):
    #     if reversed:
    #         action_ret = 2 * (action - low) / (high - low) - 1
    #         action_ret = np.clip(action_ret, low, high)
    #     return self._get_action_direct(self.action_space, action, reversed=True)
    