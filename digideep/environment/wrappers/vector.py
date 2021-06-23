import gym
import numpy as np

from gym import spaces
from digideep.environment.common.vec_env import VecEnvWrapper


################################################
# assert len(envs.observation_space.shape) == 3, "VecFrameStack supports 3d observations, i.e. images"
# Why should this class support only images??

# envs = VecFrameStackAxis(envs, **self.params["wrappers_args"]["VecFrameStackAxis"])
################################################

############################################################
class VecFrameStackAxis(VecEnvWrapper):
    """A wrapper to stack observations on an arbitrary axis.

    Args:
        venv: The VecEnv environment to be wrapped.
        nstack (int): Number of observations to be stacked in that axis.
        axis (int): The axis of those observations to be used for stacking.

    Inspired by `baselines <https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py>`__.
    """
    def __init__(self, venv, mode, path, nstack, axis=-1):
        """
        * ``Axis = 0``:  If images are already transposed using :class:`~digideep.environment.wrappers.normal.WrapperTransposeImage`,
            then use ``axis=0`` to stack images on the first dimension which is channels. This will be compatible with PyTorch.
        * ``Axis = -1``: If images are NOT transposed and their shape is like ``(W,H,3)``, then stack images on the last axis, which
            is again the channels. This is compatible with the OpenAI's defaults.
        """
        self.venv = venv
        self.path = path
        self.axis = axis
        self.nstack = nstack
        
        observation_space = venv.observation_space
        wos = observation_space.spaces[self.path]  # wrapped ob space

        
        ## This shape0 is in [1,3].
        self.axis_dim = wos.shape[axis]
        low  = np.repeat(wos.low,  self.nstack, axis=self.axis)
        high = np.repeat(wos.high, self.nstack, axis=self.axis)
        # This serves as a queue type history for past observations.
        self.stacked_obs = np.zeros((venv.num_envs,) + low.shape, dtype=low.dtype)

        mod_observation_space = spaces.Box(low=low, high=high, dtype=observation_space.spaces[self.path].dtype)
        
        observation_space.spaces[self.path] = mod_observation_space
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
        self.stacked_obs[tuple(ind)] = obs[self.path]

        # TODO: Why type is different before/after?
        # obs[self.path] = self.stacked_obs.astype(obs[self.path].dtype)
        obs[self.path] = self.stacked_obs.astype(np.float32)

        # return self.stacked_obs, rews, news, infos
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs[...] = 0
        
        ind=[slice(None)]*(len(self.stacked_obs.shape)); ind[1+self.axis] = slice(-self.axis_dim, None)
        self.stacked_obs[tuple(ind)] = obs[self.path]
        obs[self.path] = self.stacked_obs
        
        # return self.stacked_obs
        return obs

