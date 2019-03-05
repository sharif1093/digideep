import gym
import numpy as np

from gym import spaces
from digideep.environment.common.vec_env import VecEnvWrapper
from digideep.environment.common.running_mean_std import RunningMeanStd


# 
class WrapperMaskGoal(gym.ObservationWrapper):
    """
    Adopted from `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py>`__.
    """

    def __init__(self, env, nmask):
        super(WrapperMaskGoal, self).__init__(env)
        self.nmask = nmask
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-self.nmask:] = 0
        return observation

class WrapperAddTimestep(gym.ObservationWrapper):
    """
    Adopted from `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py>`__.
    """
    def __init__(self, env=None):
        super(WrapperAddTimestep, self).__init__(env)
        self.observation_space = spaces.Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)
    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))

class WrapperTransposeImage(gym.ObservationWrapper):
    """
    Adopted from `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py>`__.
    
    If the input has shape ``(W,H,3)``, wrap for PyTorch convolutions.
    """
    def __init__(self, env):
        super(WrapperTransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)
    def observation(self, observation):
        return observation.transpose(2, 0, 1)

# class TransposeImage(TransposeObs):
#     def __init__(self, env=None, op=[2, 0, 1]):
#         """
#         Transpose observation space for images
#         """
#         super(TransposeImage, self).__init__(env)
#         assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
#         self.op = op
#         obs_shape = self.observation_space.shape
#         self.observation_space = Box(
#             self.observation_space.low[0, 0, 0],
#             self.observation_space.high[0, 0, 0],
#             [
#                 obs_shape[self.op[0]],
#                 obs_shape[self.op[1]],
#                 obs_shape[self.op[2]]],
#             dtype=self.observation_space.dtype)

#     def observation(self, ob):
#         return ob.transpose(self.op[0], self.op[1], self.op[2])



class WrapperDummyMultiAgent(gym.ActionWrapper):
    """
    This wrapper is required to make the regular classes compatible with
    multi-agent architecture of Digideep. Use this wrapper for each
    environment that is not multi-agent.
    """
    def __init__(self, env, agent_name="agent"):
        super(WrapperDummyMultiAgent, self).__init__(env)
        self.agent_name = agent_name
        act_space = self.action_space
        self.action_space = spaces.Dict({self.agent_name:act_space})

    def action(self, action):
        assert isinstance(action, dict), "The provided action is not a dictionary."
        assert self.agent_name in action, "The provided agent name ({}) does not exist in the action.".format(self.agent_name)
        return action[self.agent_name]



class VecFrameStackAxis(VecEnvWrapper):
    """A wrapper to stack observations on an arbitrary axis.

    Args:
        venv: The VecEnv environment to be wrapped.
        nstack (int): Number of observations to be stacked in that axis.
        axis (int): The axis of that observations to be used for stacking.

    Inspired by `baselines <https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py>`__.
    """
    def __init__(self, venv, nstack, axis=-1):
        """
        Axis = 0:  For compatibility with PyTorch
        Axis = -1: For OpenAI's default axis.
        """
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        ## This shape0 is in [1,3]
        self.axis = axis
        self.axis_dim = wos.shape[axis]
        low  = np.repeat(wos.low,  self.nstack, axis=self.axis)
        high = np.repeat(wos.high, self.nstack, axis=self.axis)
        # This serves as a queue type history for past observations.
        self.stacked_obs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        # This throwing old history out by rolling.
        self.stacked_obs = np.roll(self.stacked_obs, shift=-self.axis_dim, axis=self.axis)
        # If an episode is reset, remove the history.
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        
        # We are using slice because these are not numpy arrays yet.
        ind=[slice(None)]*(len(self.stacked_obs.shape)); ind[1+self.axis] = slice(-self.axis_dim, None)
        self.stacked_obs[tuple(ind)] = obs

        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs[...] = 0
        
        ind=[slice(None)]*(len(self.stacked_obs.shape)); ind[1+self.axis] = slice(-self.axis_dim, None)
        self.stacked_obs[tuple(ind)] = obs
        
        return self.stacked_obs




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
        return val
    def state_dict(self):
        return self.running_mean.state_dict()
    def load_state_dict(self, state_dict):
        self.running_mean.load_state_dict(state_dict)


class VecNormalize(VecEnvWrapper):
    """A serializable vectorized wrapper that normalizes the observations and returns from an environment.

    Inspired by `baselines <https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py>`__
    and `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py>`__.
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, training=False):
        VecEnvWrapper.__init__(self, venv)
        self.norm_obs = ob
        self.norm_ret = ret
        
        if self.norm_obs:
            self.obs_normalizer = Normalizer(shape=self.observation_space.shape, clip=clipob, eps=epsilon)
        if self.norm_ret:
            self.rew_normalizer = Normalizer(shape=(), clip=cliprew, eps=epsilon)
            self.ret = np.zeros(self.num_envs)
            self.gamma = gamma
        self.training = training

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        
        obs = self._obs_filt(obs)
        rew = self._rew_filt(rew, dones)
        return obs, rew, dones, infos           

    def _obs_filt(self, obs):
        if not self.norm_obs:
            return obs
        
        if self.training:
            self.obs_normalizer.update(obs)
        return self.obs_normalizer.normalize(obs)
    
    def _rew_filt(self, rew, dones):
        if not self.norm_ret:
            return rew
        
        if self.training:
            self.ret = self.ret * self.gamma + rew
            self.rew_normalizer.update(rew)
            self.ret[dones] = 0.
        return self.rew_normalizer.normalize(rew, use_mean=False)

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obs_filt(obs)
    
    def state_dict(self):
        states = {}
        if self.norm_obs:
            states["obs_normalizer"] = self.obs_normalizer.state_dict()
        if self.norm_ret:
            states["rew_normalizer"] = self.rew_normalizer.state_dict()
            states["ret"] = self.ret
        return states
    def load_state_dict(self, state_dict):
        # logger("We are loading VecNormalize with parameters:", state_dict)
        if self.norm_obs:
            self.obs_normalizer.load_state_dict(state_dict["obs_normalizer"])
        if self.norm_ret:
            self.rew_normalizer.load_state_dict(state_dict["rew_normalizer"])
            self.ret = state_dict["ret"]



def get_type_name(cls):
    """Gets the name of a type.

    This function is used to produce a key for each wrapper to store its states in a dictionary of wrappers' states.

    Args:
        cls: The input class.
    Returns:
        str: Name of the class.
    """
    name = "{}:{}".format(cls.__class__.__module__, cls.__class__.__name__)
    # name = str(type(cls))
    return name

class VecSaveState(VecEnvWrapper):
    """
    A vectorized wrapper that saves the state of all wrappers.
    This wrapper must be the last wrapper around a VecEnv so
    the state_dict and load_state_dict functions are exposed.
    We also assume that each wrapper is used once, otherwise
    we will end up saving the state of the first of them.

    Args:
        venv: The VecEnv to be wrapped.
    """
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        
    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()
    
    def state_dict(self):
        states = {}
        venv = self
        while hasattr(venv, "venv"):
            venv = venv.venv
            name = get_type_name(venv)
            if hasattr(venv, "state_dict"):
                if name in states:
                    raise KeyError("The key "+name+" already exists in the wrapper stack!")
                states[name] = venv.state_dict()
        return states
    def load_state_dict(self, state_dict):
        venv = self
        while hasattr(venv, "venv"):
            venv = venv.venv
            name = get_type_name(venv)
            if hasattr(venv, "load_state_dict"):
                venv.load_state_dict(state_dict[name])

