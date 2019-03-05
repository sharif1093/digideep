"""
This module is mainly inspired by https://github.com/rejuvyesh/gym-dmcontrol & https://github.com/martinseilair/dm_control2gym
"""

import gym
from gym import Env
from gym.utils import seeding
from gym.utils import EzPickle

import numpy as np
import collections
import warnings
import sys
import copy

from .spec2space import spec2space
from .viewer import Viewer

from dm_control.rl.control import FLAT_OBSERVATION_KEY, flatten_observation, _spec_from_observation
from dm_control.rl.control import PhysicsError

class DmControlWrapper(Env, EzPickle):
    """Class to convert dm_control environments into gym environments.

    Note:
        This class supports ``observation_space``, ``action_space``, and ``spec`` for the environment.

    Args:
        dmcenv_creator (func): A callable object that will create the dm_control environment. 
            A callable object can delay the creation of the environment until the time we need it.
        flat_observation (bool): Whether to flatten the observation dict or not.
    """
    def __init__(self, dmcenv_creator, flat_observation=False):
        self.dmcenv = dmcenv_creator()
        self._flat_observation = flat_observation
        # NOTE: We do not use the following to flatten observation to have more control over flattening and extracting "info".
        ## The next line will flatten the observations if we really need it.
        #### self.dmcenv._flat_observation = self._flat_observation

        # We will toggle this boolean value at the first time we do the _delayed_init().
        self._delayed_init_flag = False

        # convert spec to space
        self.action_space = spec2space(self.dmcenv.action_spec())
        # NOTE: We do not use "self.dmcenv.observation_spec()" in order to remove "info" from observation keys and then flatten the dict.
        self.observation_space = spec2space(self._get_observation_spec())
        # self.spec = None
        
        ## Useful for debugging
        # self.dmcenv.action_spec()
        # self.dmcenv.observation_spec()
        # self.action_space
        # self.observation_space
        # self.dmcenv.task.get_observation(self.dmcenv.physics)
        
        self.metadata['render.modes'] = ['rgb_array', 'human']
        # TODO: The following is from mujoco_py. Do we need it?
        # self.metadata['video.frames_per_second'] = int(np.round(1.0 / self.dt))
        # depth_array, human_rgb_array, ...
        self._viewers = {key:None for key in self.metadata['render.modes']}

        # set seed
        self.seed()
        EzPickle.__init__(self)
    
    def _delayed_init(self):
        """This function aims to perform delayed initializations.
        For some parameters we should wait until the object is made by the "gym.make"
        command. Only then we can modify those parameters. All of the stuff in the
        "self.spec" are from that category; we should wait until make is called on
        the environment and then update those at the first time reset is called.
        """
        if self._delayed_init_flag:
            return
        self._delayed_init_flag = True

        # NOTE: Note that our environment will not be wrapped by the TimeLimit wrapper
        if self.dmcenv._step_limit < float('inf'):
            self.spec.max_episode_steps = int(self.dmcenv._step_limit)
        else:
            self.spec.max_episode_steps = None
        

    # @property
    # def _elapsed_steps(self):
    #     # CHEATING: This is steps rather than time.
    #     return self.dmcenv.physics.time()

    def dt(self):
        """
        Returns:
            float: The control timestep of the environment.
        """

        return self.dmcenv.control_timestep()

    def seed(self, seed=None):
        """Seeds the environment.
        """
        self.np_random, seed = seeding.np_random(seed)
        self.dmcenv.task._random = self.np_random
        return [seed]
    
    def _extract_obs_info(self, observation):
        """This function extracts the ``info`` key from the observations.
        """
        if isinstance(observation, collections.OrderedDict) and "info" in observation:
            info = copy.deepcopy(observation["info"])
            del observation["info"]
        else:
            info = {}
        return info

    def _get_observation_spec(self):
        """ This function will extract the ``observation_spec`` of the environment if that is specified explicitly.
        Otherwise, it will first extract the ``info`` key from the observation dict, if that exists, and then forms
        the shape of the observation dict.
        """
        try:
            return self.dmcenv.task.observation_spec(self.dmcenv.physics)
        except NotImplementedError:
            observation = self.dmcenv.task.get_observation(self.dmcenv.physics)
            self._extract_obs_info(observation)
            if self._flat_observation:
                observation = flatten_observation(observation)
                return _spec_from_observation(observation)[FLAT_OBSERVATION_KEY]
            else:
                return _spec_from_observation(observation)    
    
    def _get_observation(self, timestep):
        """ This function will extract the observation from the output of the ``dmcenv.step``'s timestep.

        Returns:
            tuple: ``(observation, info)``
        """
        info = self._extract_obs_info(timestep.observation)
        if self._flat_observation:
            return flatten_observation(timestep.observation)[FLAT_OBSERVATION_KEY], info
        else:
            return timestep.observation, info

    def reset(self):
        """This function resets the environment.
        """
        timestep = self.dmcenv.reset()
        obs, _ = self._get_observation(timestep)
        
        self._delayed_init()
        return obs

    def step(self, action):
        """The step function that will execute actions on the environment and return results.
        """
        try:
            timestep = self.dmcenv.step(action)
            obs, info = self._get_observation(timestep)
            reward = timestep.reward
            done = timestep.last()
        except PhysicsError: # dm_control error
            print(">>>>>>>>>>>> We got PhysicsError! <<<<<<<<<<<<")
            obs = self.reset()
            reward = 0
            done = 0
            info = {}
        return obs, reward, done, info

    def render(self, mode='human', **render_kwargs):
        """Render function which supports two modes: ``rgb_array`` and ``human``.
        If ``mode`` is ``rgb_array``, it will return the image in pixels format.
        """

        # render_kwargs = { 'height', 'width', 'camera_id', 'overlays', 'depth', 'scene_option'}
        if mode == 'rgb_array':
            pixels = self._get_viewer(mode)(**render_kwargs)
            return pixels
        elif mode == 'human':
            self._get_viewer(mode)(**render_kwargs)

    def _get_viewer(self, mode):
        self.viewer = self._viewers[mode]
        if self.viewer is None:
            if mode == "rgb_array":
                self.viewer = self.dmcenv.physics.render
            elif mode == "human":
                self.viewer = Viewer(dmcenv=self.dmcenv, width=640, height=480)
            self._viewers[mode] = self.viewer
        return self.viewer

    def close(self):
        # for k, v in self._viewers.items():
        #     if v is not None:
        #         v.close()
        #         self._viewers[k] = None
        # self._viewers = {}
        pass

