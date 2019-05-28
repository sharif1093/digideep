from dm_control import suite
from gym.envs.registration import register
from .registration import EnvCreatorSuite

# Register all benchmarks already in the suite.
prefix = "DMBench"
for domain_name, task_name in suite.BENCHMARKING:
    gym_id = '{}{}-v0'.format(domain_name.capitalize(), task_name.capitalize())
    gym_id = prefix + gym_id
    register(
        id=gym_id,
        entry_point="digideep.environment.dmc2gym.wrapper:DmControlWrapper",
        kwargs={'dmcenv_creator':EnvCreatorSuite(domain_name, task_name, task_kwargs=None, environment_kwargs=None, visualize_reward=True),
                'flat_observation':True, # Should be True
                'observation_key':"agent"}
    )


## Arguments of OpenAI Gym "register" function:
##   - id, entry_point=None, 
##   - trials=100, 
##   - reward_threshold=None, 
##   - local_only=False, 
##   - kwargs=None, 
##   - nondeterministic=False, 
##   - tags=None, 

##   - max_episode_steps=None, 
##   - max_episode_seconds=None, 
##   - timestep_limit=None
