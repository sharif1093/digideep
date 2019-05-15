"""
This module is inspired by `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/>`_.
"""


import numpy as np
import os

from digideep.environment.common.atari_wrappers import make_atari, wrap_deepmind
from digideep.environment.common.monitor import Monitor

from digideep.environment.common.vec_env.subproc_vec_env import SubprocVecEnv
from digideep.environment.common.vec_env.dummy_vec_env import DummyVecEnv
# from digideep.environment.common.vec_env.vec_monitor import VecMonitor
from digideep.environment.wrappers import WrapperAddTimestep
from digideep.environment.wrappers import WrapperTransposeImage
from digideep.environment.wrappers import WrapperDummyMultiAgent
from digideep.environment.wrappers import VecFrameStackAxis
from digideep.environment.wrappers import VecNormalize
from digideep.environment.wrappers import VecSaveState

from gym.wrappers.monitor import Monitor as MonitorVideoRecorder

from digideep.utility.toolbox import get_module

from digideep.utility.logging import logger
from gym import spaces

################################################
###      Importing Environment Packages      ###
################################################
import gym
# Even though we don't need dm_control to be loaded here, it helps in initializing glfw.
# import digideep.environment.dmc2gym

from gym.envs.registration import registry
################################################



#############################
##### Utility Functions #####
#############################
def space2config(S):
    """Function to convert space's characteristics into a config-space dict.
    """
    # S.__class__.__name__: "Discrete" / "Box"
    if isinstance(S, spaces.Discrete):
        typ = S.__class__.__name__
        dim = np.int32(S.n)
        lim = (np.nan, np.nan) # Discrete Spaces do not have high/low
        config = {"typ":typ, "dim":dim, "lim":lim}
    elif isinstance(S, spaces.Box):
        typ = S.__class__.__name__
        dim = S.shape # S.shape[0]: This "[0]" only supports 1d arrays.
        lim = (S.low.tolist(), S.high.tolist())
        config = {"typ":typ, "dim":dim, "lim":lim}
    elif isinstance(S, spaces.Dict):
        config = {}
        for k in S.spaces:
            config[k] = space2config(S.spaces[k])
    else:
        logger.fatal("Unknown type for space:", type(S))
        raise NotImplementedError
    
    return config



#################################
##### MakeEnvironment Class #####
#################################

class MakeEnvironment:
    """This class will make the environment. It will apply the wrappers to the environments as well.

    Tip:
        Except :class:`~digideep.environment.common.monitor.Monitor` environment, no environment will be applied on the environment
        unless explicitly specified.
    """
    
    registered = False

    def __init__(self, session, mode, seed, **params):
        self.mode = mode # train/test/eval
        self.seed = seed
        self.session = session
        self.params  = params
        
        # Won't we have several environment registrations by this?
        if params["from_module"]:
            try:
                get_module(params["from_module"])
            except Exception as ex:
                logger.fatal("While importing user module:", ex)
                exit()
        elif (params["from_params"]) and (not MakeEnvironment.registered):
            try:
                registry.register(**params["register_args"])
                MakeEnvironment.registered = True
            except Exception as ex:
                logger.fatal("While registering from parameters:", ex)
                exit()
        
        # After all of these, check if environment is registered in the gym or not.
        if not params["name"] in registry.env_specs:
            logger.fatal("Environment '" + params["name"] + "' is not registered in the gym registry.")
            exit()
        
    def make_env(self, rank, force_no_monitor=False):
        import sys # For debugging
        def _f():
            env = gym.make(self.params["name"])

            is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
            if is_atari:
                env = make_atari(self.params["name"])

            env.seed(self.seed + rank)

            ## Add the requested wrappers
            # This adds the timestep as an observation to the environment. With this the agent will know the time.
            if self.params["wrappers"]["add_time_step"]:
                assert len(env.observation_space.shape) == 1, "AddTimeStep supports 1d observations."
                assert str(env).find('TimeLimit') > -1, "AddTimeStep need environment to be a TimeLimit."
                env = WrapperAddTimestep(env)

            if not force_no_monitor and self.params["wrappers"]["add_monitor"]:
                log_dir = os.path.join(self.session["path_monitor"], str(rank))
                env = Monitor(env, log_dir, **self.params["wrappers_args"]["Monitor"])
            
            if self.mode == "eval":
                videos_dir = os.path.join(self.session["path_videos"], str(rank))
                env = MonitorVideoRecorder(env, videos_dir, video_callable=lambda id:True)
            # elif self.mode == "train":
            #     env = MonitorVideoRecorder(env, videos_dir)

            if is_atari and len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
            
            if self.params["wrappers"]["add_image_transpose"]:
                obs_shape = env.observation_space.shape
                assert len(obs_shape) == 3
                assert obs_shape[2] in [1, 3]
                env = WrapperTransposeImage(env)
            
            if self.params["wrappers"]["add_dummy_multi_agent"]:
                env = WrapperDummyMultiAgent(env, **self.params["wrappers_args"]["DummyMultiAgent"])

            return env
        return _f
        
    def create_envs(self, num_workers=1, force_no_monitor=False):
        envs = [self.make_env(rank=idx, force_no_monitor=force_no_monitor) for idx in range(num_workers)]
        
        # NOTE: We don't use DummyVecEnvs to avoid performing glfw.init() on the Main process.
        if self.mode == "eval":
            envs = DummyVecEnv(envs)
        else:
            envs = SubprocVecEnv(envs)
        
        # NOTE: We don't use DummyVecEnvs to avoid performing glfw.init() on the Main process.
        # if num_workers > 1:
        #     envs = SubprocVecEnv(envs)
        # else:
        #     envs = DummyVecEnv(envs)
        
        # Monitor seems to have more interesting features than VecMonitor. So we may not use VecMonitor.
        # envs = VecMonitor(envs, 'test.log')

        # TODO: VecNormalize will not work with Dict observation spaces.
        # Works only for vector observations, not for images
        if self.params["wrappers"]["add_vec_normalize"]:
            assert len(envs.observation_space.shape) == 1, "VecNormalize supports 1d (vector) observations"
            training = True if self.mode=="train" else False
            envs = VecNormalize(envs, **self.params["wrappers_args"]["VecNormalize"], training=training)
            ## NOTE: It is OK for gamma to be "None", but that means that we are not interested in normalizing the returns.
            # if gamma is None:
            #     envs = VecNormalize(envs, ret=False)
            # else:
            #     envs = VecNormalize(envs, gamma=gamma)

        # TODO: VecFrameStackAxis will not work with Dict observation spaces.
        if self.params["wrappers"]["add_frame_stack_axis"]:
            # assert len(envs.observation_space.shape) == 3, "VecFrameStack supports 3d observations, i.e. images"
            # envs = VecFrameStack(envs, **self.params["wrappers_args"]["VecFrameStack"])
            envs = VecFrameStackAxis(envs, **self.params["wrappers_args"]["VecFrameStackAxis"])
        
        # We must add VecSaveState to save the state of stateful wrappers.
        envs = VecSaveState(envs)

        return envs
        

    def get_config(self):
        """This function will generate a dict of interesting specifications of the environment.

        """

        # TODO: Assume nested dicts for observation and action
        #       Create nested dicts for the shapes of both.

        _f = self.make_env(rank=0, force_no_monitor=True)
        venv = self.create_envs(num_workers=1, force_no_monitor=True)
        venv.reset()

        config = {
            'action_space' :     space2config(venv.action_space),      # This is type of action space: Discrete, Box, ...
            'observation_space': space2config(venv.observation_space), # Observation space
            
            # 'reward_range':      env.reward_range,
            'max_episode_steps': venv.spec.max_episode_steps,           # Maximum allowable steps
            # 'dt':                env.dt() if hasattr(env, "dt") else None          # The delta t as the timestep of the environment.
        }

        venv.close()
        return config


