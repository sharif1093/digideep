"""
This parameter file is designed for continuous action environments.
For discrete action environments minor modifications might be required.

See Also:
    :ref:`ref-parameter-files`
"""

import numpy as np
from copy import deepcopy

from digideep.environment import MakeEnvironment
from collections import OrderedDict

################################################################################
#########                       CONTROL PANEL                          #########
################################################################################
# The control panel brings the most important parameters to the top. It also
# helps to set all parameters that depends on a single value from one specific
# place:
#  - We can print and save this control panel instead of parameter list.
#  - The parameters here can also be taken from a YAML file.
#  - We can have default values now very easily.
#  - It provides semantic grouping of parameters
#  - We may unify the name of parameters which are basically the same in different
#    methods, but have different names.


cpanel = OrderedDict()

#####################
### Runner Parameters
# num_frames = 10e6  # Number of frames to train
cpanel["number_epochs"] = 1000  # epochs
cpanel["epoch_size"]    = 1000  # cycles
cpanel["test_activate"] = True  # Test activated
cpanel["test_interval"] = 100   # Test Interval Every #n Epochs
cpanel["test_win_size"] = 10    # Number of episodes to run test.
cpanel["save_interval"] = 10    # Save Interval Every #n Epochs
## Simulation will end when either time or max iterations exceed the following:
cpanel["max_exec_time"] = 20     # hours
cpanel["max_exec_iter"] = None   # number of epochs
cpanel["seed"] = 0
cpanel["cuda_deterministic"] = False # With TRUE we MIGHT get more deterministic results but at the cost of speed.

#####################
### Memory Parameters
cpanel["keep_old_checkpoints"] = False
cpanel["memory_size_in_chunks"] = int(1e6) # SHOULD be 1 for on-policy methods that do not have a replay buffer.
# SUGGESTIONS: 2^0 (~1e0) | 2^3 (~1e1) | 2^7 (~1e2) | 2^10 (~1e3) | 2^13 (~1e4) | 2^17 (1e5) | 2^20 (~1e6)

##########################
### Environment Parameters
# Acrobot-v1 | CartPole-v1 | MountainCarContinuous-v0
# cpanel["model_name"] = 'DMBenchCheetahRun-v0'        # Classic Control Env
# cpanel["from_module"] = 'digideep.environment.dmc2gym'
# cpanel["observation_key"] = "/agent"

cpanel["model_name"] = 'Pendulum-v0'        # Classic Control Env
cpanel["observation_key"] = "/agent"
cpanel["gamma"] = 0.99     # The gamma parameter used in VecNormalize | Agent.preprocess | Agent.step

# # Wrappers
# cpanel["add_time_step"]          = False # It is suggested for MuJoCo environments. It adds time to the observation vector. CANNOT be used with renders.
# cpanel["add_image_transpose"]    = False # Necessary if training on Gym with renders, e.g. Atari games
# cpanel["add_dummy_multi_agent"]  = True  # Necessary if the environment is not multi-agent (i.e. all dmc and gym environments),
#                                          # to make it compatibl with our multi-agent architecture.
# cpanel["add_vec_normalize"]      = False # NOTE: USE WITH CARE. Might be used with MuJoCo environments. CANNOT be used with rendered observations.
# cpanel["add_frame_stack_axis"]   = False # Necessary for training on renders, e.g. Atari games. The nstack parameter is usually 4
#                                          # This stacks frames at a custom axis. If the ImageTranspose is activated
#                                          # then axis should be set to 0 for compatibility with PyTorch.
# # TODO: Action normalizer and clipper


##################################
### Exploration/Exploitation Balance
### Exploration (~ num_workers * n_steps)
cpanel["num_workers"] = 1     # From Explorer           # Number of exploratory workers working together
cpanel["n_steps"] = 1         # From Explorer           # Number of frames to produce
### Exploitation (~ n_update * batch_size)
cpanel["n_update"] = 1        # From Agents
cpanel["batch_size"] = 64     # From Agents
cpanel["warm_start"] = 0


#####################
### Agents Parameters
cpanel["agent_type"] = "digideep.agent.ddpg.Agent"
cpanel["lr_actor"] = 0.0001 # 0.0001
cpanel["lr_critic"] = 0.001 # 0.001
cpanel["eps"] = 1e-5 # Epsilon parameter used in the optimizer(s) (ADAM/RMSProp/...)

cpanel["polyak_factor"] = 0.001 # 0.01

cpanel["noise_std"] = 0.2


################################################################################
#########                      PARAMETER TREE                          #########
################################################################################
def gen_params(cpanel):
    params = {}
    # Environment
    params["env"] = {}
    params["env"]["name"]   = cpanel["model_name"]
    
    params["env"]["from_module"] = cpanel.get("from_module", '')
    params["env"]["from_params"] = cpanel.get("from_params", False)


    ##############################################
    ### Normal Wrappers ###
    #######################
    norm_wrappers = []

    # Converting observation to 1 level
    norm_wrappers.append(dict(name="digideep.environment.wrappers.normal.WrapperLevelDictObs",
                              args={"path":cpanel["observation_key"],
                              },
                              enabled=False))
    # Normalizing actions (to be in [-1, 1])
    norm_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.WrapperNormalizeActDict",
                              args={"paths":["agent"]},
                              enabled=False))

    ##############################################
    ### Vector Wrappers ###
    #######################
    vect_wrappers = []

    # Normalizing observations
    vect_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.VecNormalizeObsDict",
                              args={"paths":[cpanel["observation_key"]],
                                    "clip":5, # 10
                                    "epsilon":1e-8
                              },
                              enabled=False))
    # Normalizing rewards
    vect_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.VecNormalizeRew",
                              args={"clip":5, # 10
                                    "gamma":cpanel["gamma"],
                                    "epsilon":1e-8
                              },
                              enabled=False))
    ##############################################
    params["env"]["main_wrappers"] = {"Monitor":{"allow_early_resets":True, # We need it to allow early resets in the test environment.
                                                 "reset_keywords":(),
                                                 "info_keywords":()},
                                      "WrapperDummyMultiAgent":{"agent_name":"agent"},
                                      "WrapperDummyDictObs":{"observation_key":"agent"}
                                     }
    params["env"]["norm_wrappers"] = norm_wrappers
    params["env"]["vect_wrappers"] = vect_wrappers


    menv = MakeEnvironment(session=None, mode=None, seed=1, **params["env"])
    params["env"]["config"] = menv.get_config()

    # Some parameters
    # params["env"]["gamma"] = 1-1/params["env"]["config"]["max_steps"] # 0.98



    #####################################
    # Runner: [episode < cycle < epoch] #
    #####################################
    params["runner"] = {}
    params["runner"]["name"] = cpanel.get("runner_name", "digideep.pipeline.Runner")
    params["runner"]["max_time"] = cpanel.get("max_exec_time", None)
    params["runner"]["max_iter"] = cpanel.get("max_exec_iter", None)
    params["runner"]["n_cycles"] = cpanel["epoch_size"]    # Meaning that 100 cycles are 1 epoch.
    params["runner"]["n_epochs"] = cpanel["number_epochs"] # Testing and savings are done after each epoch.
    params["runner"]["randargs"] = {'seed':cpanel["seed"], 'cuda_deterministic':cpanel["cuda_deterministic"]}
    params["runner"]["test_act"] = cpanel["test_activate"] # Test Activate
    params["runner"]["test_int"] = cpanel["test_interval"] # Test Interval
    params["runner"]["save_int"] = cpanel["save_interval"] # Save Interval

    # We "save" after each epoch is done.
    # We "test" after each epoch is done.


    
    params["agents"] = {}
    ##############################################
    ### Agent (#1) ###
    ##################
    params["agents"]["agent"] = {}
    params["agents"]["agent"]["name"] = "agent"
    params["agents"]["agent"]["type"] = cpanel["agent_type"]
    params["agents"]["agent"]["observation_path"] = cpanel["observation_key"]
    params["agents"]["agent"]["methodargs"] = {}
    params["agents"]["agent"]["methodargs"]["n_update"] = cpanel["n_update"]  # Number of times to perform PPO update. Alternative name: PPO_EPOCH
    params["agents"]["agent"]["methodargs"]["gamma"] = cpanel["gamma"]  # Discount factor Gamma
    params["agents"]["agent"]["methodargs"]["clamp_return"] = 1/(1-float(cpanel["gamma"]))
    
    print("Clip Return =", params["agents"]["agent"]["methodargs"]["clamp_return"])
    # params["agents"]["agent"]["methodargs"]["clip_param"] = cpanel["clip_param"]  # PPO clip parameter
    # params["agents"]["agent"]["methodargs"]["entropy_coef"] = cpanel["entropy_coef"]  # Entropy term coefficient
    # params["agents"]["agent"]["methodargs"]["max_grad_norm"] = cpanel["max_grad_norm"]  # Max norm of gradients
    # params["agents"]["agent"]["methodargs"]["use_clipped_value_loss"] = cpanel["use_clipped_value_loss"]

    ################
    
    params["agents"]["agent"]["sampler_list"] = ["digideep.agent.ddpg.sampler.sampler_re"]
    params["agents"]["agent"]["sampler_args"] = {"agent_name":params["agents"]["agent"]["name"],
                                                 "batch_size":cpanel["batch_size"],
                                                 "observation_path":params["agents"]["agent"]["observation_path"]
                                                }

    # # It deletes the last element from the chunk
    # params["agents"]["agent"]["sampler"]["truncate_datalists"] = {"n":1} # MUST be 1 to truncate last item: (T+1 --> T)

    #############
    ### Model ###
    #############
    agent_name = params["agents"]["agent"]["name"]
    observation_path = params["agents"]["agent"]["observation_path"]
    params["agents"]["agent"]["policyname"] = "digideep.agent.ddpg.Policy"
    params["agents"]["agent"]["policyargs"] = {"obs_space": params["env"]["config"]["observation_space"][observation_path],
                                               "act_space": params["env"]["config"]["action_space"][agent_name],
                                               "hidden_size": 256,
                                               "actor_args": {"eps":0.003},
                                               "critic_args": {"eps":0.003},
                                               "average_args": {"mode":"soft", "polyak_factor":cpanel["polyak_factor"]},
                                               # {"mode":"hard", "interval":10000}
                                               }
    
    lim = params["env"]["config"]["action_space"][agent_name]["lim"][1][0]
    # params["agents"]["agent"]["noisename"] = "digideep.agent.noises.EGreedyNoise"
    # params["agents"]["agent"]["noiseargs"] = {"std":cpanel["noise_std"], "e":0.3, "lim": lim}
    
    params["agents"]["agent"]["noisename"] = "digideep.agent.noises.OrnsteinUhlenbeckNoise"
    params["agents"]["agent"]["noiseargs"] = {"mu":0, "theta":0.15, "sigma":cpanel["noise_std"], "lim":lim}
    # params["agents"]["agent"]["noiseargs"] = {"mu":0, "theta":0.15, "sigma":1}

    params["agents"]["agent"]["optimname_actor"] = "torch.optim.Adam"
    params["agents"]["agent"]["optimargs_actor"] = {"lr":cpanel["lr_actor"]}   # , "eps":cpanel["eps"]

    params["agents"]["agent"]["optimname_critic"] = "torch.optim.Adam"
    params["agents"]["agent"]["optimargs_critic"] = {"lr":cpanel["lr_critic"]} # , "eps":cpanel["eps"]

    # RMSprop optimizer alpha
    # params["agents"]["agent"]["optimargs"] = {"lr":1e-2, "alpha":0.99, "eps":1e-5, "weight_decay":0, "momentum":0, "centered":False}
    ##############################################


    ##############################################
    ### Memory ###
    ##############
    params["memory"] = {}

    params["memory"]["train"] = {}
    params["memory"]["train"]["type"] = "digideep.memory.ringbuffer.Memory"
    params["memory"]["train"]["args"] = {"name":"train",
                                         "keep_old_checkpoints":cpanel.get("keep_old_checkpoints", False),
                                         "chunk_sample_len":cpanel["n_steps"],
                                         "buffer_chunk_len":cpanel["memory_size_in_chunks"],
                                         "overrun":1}
    # chunk_sample_len: Number of samples in a chunk
    # buffer_chunk_len: Number of chunks in the buffer
    ##############################################

    
    
    ##############################################
    ### Explorer ###
    ################
    params["explorer"] = {}

    params["explorer"]["train"] = {}
    params["explorer"]["train"]["mode"] = "train"
    params["explorer"]["train"]["env"] = params["env"]
    params["explorer"]["train"]["do_reset"] = False
    params["explorer"]["train"]["final_action"] = False
    params["explorer"]["train"]["warm_start"] = cpanel["warm_start"] # In less than "warm_start" steps the agent will take random actions. 
    params["explorer"]["train"]["num_workers"] = cpanel["num_workers"]
    params["explorer"]["train"]["deterministic"] = False # MUST: Takes random actions
    params["explorer"]["train"]["n_steps"] = cpanel["n_steps"] # Number of steps to take a step in the environment
    params["explorer"]["train"]["n_episodes"] = None # Do not limit # of episodes
    params["explorer"]["train"]["win_size"] = 20 # Number of episodes to episode reward for report
    params["explorer"]["train"]["render"] = False
    params["explorer"]["train"]["render_delay"] = 0
    params["explorer"]["train"]["seed"] = cpanel["seed"] + 90
    params["explorer"]["train"]["extra_env_kwargs"] = {}

    params["explorer"]["test"] = {}
    params["explorer"]["test"]["mode"] = "test"
    params["explorer"]["test"]["env"] = params["env"]
    params["explorer"]["test"]["do_reset"] = True
    params["explorer"]["test"]["final_action"] = False
    params["explorer"]["test"]["warm_start"] = 0
    params["explorer"]["test"]["num_workers"] = cpanel["num_workers"] # We can use the same amount of workers for testing!
    params["explorer"]["test"]["deterministic"] = True   # MUST: Takes the best action
    params["explorer"]["test"]["n_steps"] = None # Do not limit # of steps
    params["explorer"]["test"]["n_episodes"] = cpanel["test_win_size"]
    params["explorer"]["test"]["win_size"] = cpanel["test_win_size"] # Extra episodes won't be counted
    params["explorer"]["test"]["render"] = False
    params["explorer"]["test"]["render_delay"] = 0
    params["explorer"]["test"]["seed"] = cpanel["seed"] + 100 # We want to make the seed of test environments different from training.
    params["explorer"]["test"]["extra_env_kwargs"] = {}

    params["explorer"]["eval"] = {}
    params["explorer"]["eval"]["mode"] = "eval"
    params["explorer"]["eval"]["env"] = params["env"]
    params["explorer"]["eval"]["do_reset"] = False
    params["explorer"]["eval"]["final_action"] = False
    params["explorer"]["eval"]["warm_start"] = 0
    params["explorer"]["eval"]["num_workers"] = 1
    params["explorer"]["eval"]["deterministic"] = True   # MUST: Takes the best action
    params["explorer"]["eval"]["n_steps"] = None # Do not limit # of steps
    params["explorer"]["eval"]["n_episodes"] = 1
    params["explorer"]["eval"]["win_size"] = -1
    params["explorer"]["eval"]["render"] = True
    params["explorer"]["eval"]["render_delay"] = 0
    params["explorer"]["eval"]["seed"] = cpanel["seed"] + 101 # We want to make the seed of eval environment different from test/train.
    params["explorer"]["eval"]["extra_env_kwargs"] = {}
    ##############################################

    return params

