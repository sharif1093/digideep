"""
This module is inspired by `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/>`_.
"""

import numpy as np
import os

from digideep.utility.toolbox import get_module, get_class
from digideep.utility.logging import logger

################################################
## Viutal wrappers ##
#####################
from gym.wrappers.monitor import Monitor as MonitorVideoRecorder

from .common.atari_wrappers import make_atari, wrap_deepmind

# from .common.vec_env.vec_monitor import VecMonitor
from .common.monitor import Monitor

from .common.vec_env.subproc_vec_env import SubprocVecEnv
from .common.vec_env.dummy_vec_env import DummyVecEnv

## Our essential wrappers
from .wrappers.save_state import VecSaveState

from .wrappers.adapter import WrapperDummyMultiAgent
from .wrappers.adapter import WrapperDummyDictObs
from .wrappers.adapter import WrapperFlattenObsDict
from .wrappers.adapter import VecObsRewInfoActWrapper
################################################

from gym import spaces

################################################
###      Importing Environment Packages      ###
################################################
import gym
# Even though we don't need dm_control to be loaded here, it helps in initializing glfw when using MuJoCo 1.5.
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
        
    def make_env(self, rank, force_no_monitor=False, extra_env_kwargs={}):
        import sys # For debugging
        def _f():
            # The header of gym.make(.): `def make(id, **kwargs)`
            env = gym.make(self.params["name"], **extra_env_kwargs)
            env.seed(self.seed + rank)
            
            ## Atari environment wrappers
            is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
            if is_atari:
                env = make_atari(self.params["name"])
            if is_atari and len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)


            ## Add monitoring wrappers (not optional).
            if not force_no_monitor:
                log_dir = os.path.join(self.session["path_monitor"], str(rank))
                env = Monitor(env, log_dir, **self.params["main_wrappers"]["Monitor"])


            ## Add a video recorder if mode == "eval".
            if self.mode == "eval":
                videos_dir = os.path.join(self.session["path_videos"], str(rank))
                env = MonitorVideoRecorder(env, videos_dir, video_callable=lambda id:True)

            ## Dummy Dict Action and Observation
            if not isinstance(env.action_space, spaces.Dict):
                env = WrapperDummyMultiAgent(env, **self.params["main_wrappers"]["WrapperDummyMultiAgent"])
            if not isinstance(env.observation_space, spaces.Dict):
                env = WrapperDummyDictObs(env, **self.params["main_wrappers"]["WrapperDummyDictObs"])
            
            ## Flatten the observation_space (which is by now of spaces.Dict type.)
            # spaces.Dict({"obs1":spaces.Box, "obs2": spaces.Dict({"image1":spaces.Box, "sensor2":spaces.Discrete})})
            # Will be:
            # spaces.Dict({"/obs1":spaces.Box, "/obs2/image1":spaces.Box, "/obs2/sensor2":spaces.Discrete})
            env = WrapperFlattenObsDict(env)
            
            ## NOTE: We do not flatten the action_space, since we usually deal with 1-level dicts for actions.
            ##       If nested actions are to be considered we can upgrade action_spaces to flattened spaces.

            ## Adding arbitrary wrapper stack
            env = self.run_wrapper_stack(env, self.params["norm_wrappers"])

            return env
        return _f
        
    def create_envs(self, num_workers=1, force_no_monitor=False, extra_env_kwargs={}):
        envs = [self.make_env(rank=idx, force_no_monitor=force_no_monitor, extra_env_kwargs=extra_env_kwargs) for idx in range(num_workers)]
        
        ## NOTE: We do not use DummyVecEnvs when num_workers==1 to avoid running glfw.init() on the Main process.
        if self.mode == "eval":
            envs = DummyVecEnv(envs)
        else:
            envs = SubprocVecEnv(envs)
        
        ## Converting data structure of obs/rew/infos/actions:
        envs = VecObsRewInfoActWrapper(envs)

        ## Monitor seems to have more interesting features than VecMonitor. So we may not use VecMonitor.
        # envs = VecMonitor(envs, 'test.log')

        ## Adding arbitrary wrapper stack
        envs = self.run_wrapper_stack(envs, self.params["vect_wrappers"])
        
        ## We must add VecSaveState as the last wrapper to save the state of stateful wrappers recursively.
        envs = VecSaveState(envs)

        return envs
    
    def run_wrapper_stack(self, env, stack):
        for index in range(len(stack)):
            if stack[index]["enabled"]:
                wrapper_class = get_class(stack[index]["name"])
                # We pass mode to the wrapper as well, so the wrapper can adjust itself.
                env = wrapper_class(env, mode=self.mode, **stack[index]["args"])
        return env

    def get_config(self):
        """This function will generate a dict of interesting specifications of the environment.

        Note: Observation and action can be nested spaces.Dict.
        """

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


