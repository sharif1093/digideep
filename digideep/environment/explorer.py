import numpy as np
import time
from collections import OrderedDict

from digideep.environment import MakeEnvironment
from .data_helpers import flatten_dict, update_dict_of_lists, complete_dict_of_list, convert_time_to_batch_major, extract_keywise

# from mujoco_py import MujocoException
# from dm_control.rl.control import PhysicsError

from digideep.utility.logging import logger
from digideep.utility.profiling import KeepTime
from digideep.utility.monitoring import monitor

class Explorer:
    """A class which runs environments in parallel and returns the result trajectories in a unified structure.
    It support multi-agents in an environment.

    Note:
        The entrypoint of this class is the :func:`update` function, in which the :func:`step` function will be
        called for ``n_steps`` times. In the :func:`step` function, the :func:`prestep` function is called first to get the
        actions from the agents. Then the ``env.step`` function is called to execute those actions in the environments.
        After the loop is done in the :func:`update`, we do another :func:`prestep` to save the ``observations``/``actions``
        of the last step. This indicates the final action that the agent would take without actually executing that. This
        information will be useful in some algorithms.

    Args:
        session (:obj:`~digideep.pipeline.session.Session`): The running session object.
        agents (dict): A dictionary of the agents and their corresponding agent objects.
        mode (str): The mode of the Explorer, which is any of the three: ``train`` | ``test`` | ``eval``
        env (:obj:`env`): The parameters of the environment.
        do_reset (bool): A flag indicating whether to reset the environment at the update start.
        final_action (bool): A flag indicating whether in the final call of :func:`prestep` the action should also be generated or not.
        num_workers (int): Number of workers to work in parallel.
        deterministic (bool): Whether to choose the optimial action or to mix some noise with the action (i.e. for exploration).
        n_steps (int): Number of steps to take in the :func:`update`.
        render (bool): A flag used to indicate whether environment should be rendered at each step.
        render_delay (float): The amount of seconds to wait after calling ``env.render``. Used when environment is too fast for
            visualization, typically in ``eval`` mode.
        seed (int): The environment seed.

    Attributes:
        steps (int): Number of times the :func:`step` function is called.
        n_episode (int): Number of episodes (a full round of simulation) generated so far.
        timesteps (int): Number of total timesteps of experience generated so far.
        was_reset (bool): A flag indicating whether the Explorer has been just reset or not.
        observations: A tracker of environment observations used to produce the actions for the next step.
        masks: A tracker of environment ``done`` flag indicating the start of a new episode.
        hidden_states: A tracker of hidden_states of the agents for producing the next step action in recurrent policies.

    Caution:
        Use ``do_reset`` with caution; only when you know what the consequences are.
        Generally there are few oportunities when this flag needs to be true.
    
    Tip:
        This class is partially serializable. It only saves the state of environment wrappers and not the environment per se.
    
    See Also:
        :ref:`ref-data-structure`
    """
    def __init__(self, session, agents=None, **params):
        self.agents = agents
        self.params = params
        self.session = session

        # Create models
        extra_env_kwargs = self.params.get("extra_env_kwargs", {})
        menv = MakeEnvironment(session, mode=self.params["mode"], seed=self.params["seed"], **self.params["env"])
        self.envs = menv.create_envs(num_workers=self.params["num_workers"], extra_env_kwargs=extra_env_kwargs)
        
        # self.params["env"]["env_type"]
        
        self.state = {}
        self.state["steps"] = 0
        self.state["n_episode"] = 0
        self.state["timesteps"] = 0
        self.state["was_reset"] = False

        self.local = {}
        self.local["steps"] = 0
        self.local["n_episode"] = 0

        self.monitor_n_episode()
        self.monitor_timesteps()

        # We only reset once. Later environments will be reset automatically.
        self.reset()
        # Will the results be reported when using ``do_reset``?`

    def monitor_n_episode(self):
        if self.params["mode"] == "train":
            monitor.set_meta_key("episode", self.state["n_episode"])
    def monitor_timesteps(self):
        if self.params["mode"] == "train":
            monitor.set_meta_key("frame", self.state["timesteps"])

    def state_dict(self):
        # TODO" Should we make a deepcopy?
        return {"state":self.state, "envs":self.envs.state_dict()}
    def load_state_dict(self, state_dict):
        self.state.update(state_dict["state"])
        self.envs.load_state_dict(state_dict["envs"])

        self.monitor_n_episode()
        self.monitor_timesteps()
        
        # if self.params["mode"] in ["test", "eval"]:
        #     # We reset the explorer in case of test/eval to clear the history of observations/masks/hidden_state.
        #     # Because this part does not make sense to be transferred.
        #     self.reset()

    def report_rewards(self, infos):
        """This function will extract episode information from infos and will send them to
        :class:`~digideep.utility.monitoring.Monitor` class.
        """
        # This episode keyword only exists if we use a Monitor wrapper.
        # This keyword will only appear at the "reset" times.
        # TODO: If this is a true multi-agent system, then the rewards
        #       must be separated as well!
        if '/episode/r' in infos.keys():
            rewards = infos['/episode/r']
            for rew in rewards:
                if (rew is not None) and (not np.isnan(rew)):
                    self.local["n_episode"] += 1
                    self.state["n_episode"] += 1
                    
                    self.monitor_n_episode()

                    monitor("/reward/"+self.params["mode"]+"/episodic", rew, window=self.params["win_size"])
                    self.session.writer.add_scalar('reward/'+self.params["mode"], rew, self.state["n_episode"])

    def close(self):
        """It closes all environments.
        """
        self.envs.close()

    def reset(self):
        """Will reset the Explorer and all of its states. Will set ``was_reset`` to ``True`` to prevent immediate resets.
        """
        self.state["observations"] = self.envs.reset()
        self.state["masks"] = np.array([[0]]*self.params["num_workers"], dtype=np.float32)

        # The initial hidden_state is not saved in the memory. The only use for it is
        # getting passed to the action_generator.
        # So if there is a size mismatch between this and the next hidden_states, no
        # conflicts/errors would happen.
        self.state["hidden_state"] = {}
        for agent_name in self.agents:
            self.state["hidden_state"][agent_name] = self.agents[agent_name].reset_hidden_state(self.params["num_workers"])
        
        self.state["was_reset"] = True

    def prestep(self, final_step=False):
        """
        Function to produce actions for all of the agents. This function does not execute the actions in the environment.
        
        Args:
            final_step (bool): A flag indicating whether this is the last call of this function.
        
        Returns:
            dict: The pre-transition dictionary containing observations, masks, and agents informations. The format is like:
            ``{"observations":..., "masks":..., "agents":...}``
        """

        with KeepTime("to_numpy"):
            # TODO: Is it necessary for conversion of obs?
            # NOTE: The np conversion will not work if observation is a dictionary.
            # observations = np.array(self.state["observations"], dtype=np.float32)
            observations = self.state["observations"]
            masks = self.state["masks"]
            hidden_state = self.state["hidden_state"]

        with KeepTime("gen_action"):
            publish_agents = True 
            agents = {}
            # TODO: We are assuming a one-level action space.
            if (not final_step) or (self.params["final_action"]):
                if self.state["steps"] < self.params["warm_start"]:
                    # Take RANDOM actions if warm-starting
                    for agent_name in self.agents:
                        agents[agent_name] = self.agents[agent_name].random_action_generator(self.envs, self.params["num_workers"])
                else:
                    # Take REAL actions if not warm-starting
                    for agent_name in self.agents:
                        action_generator = self.agents[agent_name].action_generator
                        agents[agent_name] = action_generator(observations, hidden_state[agent_name], masks, deterministic=self.params["deterministic"])
            else:
                publish_agents = False
            # We are saving the "new" hidden_state now.

            # for agent_name in self.agents:
            #     if (not final_step) or (self.params["final_action"]):
            #         action_generator = self.agents[agent_name].action_generator
            #         agents[agent_name] = action_generator(observations, hidden_state[agent_name], masks, deterministic=self.params["deterministic"])
            #     else:
            #         publish_agents = False
                
            
        with KeepTime("form_dictionary"):
            if publish_agents:
                pre_transition = dict(observations=observations,
                                      masks=masks,
                                      agents=agents)
            else:
                pre_transition = dict(observations=observations,
                                      masks=masks)
        return pre_transition


    def step(self):
        """Function that runs the ``prestep`` and the actual ``env.step`` functions.
        It will also manipulate the transition data to be in appropriate format.

        Returns:
            dict: The full transition information, including the pre-transition (actions, last observations, etc) and the
            results of executing actions on the environments, i.e. rewards and infos. The format is like:
            ``{"observations":..., "masks":..., "rewards":..., "infos":..., "agents":...}``
        
        See Also:
            :ref:`ref-data-structure`
        """

        # We are saving old versions of observations, hidden_state, and masks.
        with KeepTime("prestep"):
            pre_transition = self.prestep()
            
        # TODO: For true multi-agent systems, rewards must be a dictionary as well,
        #       i.e. one reward for each agent. However, if the agents are pursuing
        #       a single goal, the reward can still be a single scalar!

        # Updating observations and masks: These two are one step old in the trajectory.
        # hidden_state is the newest.
        
        with KeepTime("envstep"):
            # Prepare actions
            actions = extract_keywise(pre_transition["agents"], "actions")

            # Step
            self.state["observations"], rewards, dones, infos = self.envs.step(actions)
            # Post-step
            self.state["hidden_state"] = extract_keywise(pre_transition["agents"], "hidden_state")
            self.state["masks"] = np.array([0.0 if done_ else 1.0 for done_ in dones], dtype=np.float32).reshape((-1,1))

            # NOTE: Uncomment if you find useful information in the continuous rewards ...
            # monitor("/reward/"+self.params["mode"]+"/continuous", np.mean(rewards))

        with KeepTime("render"):
            if self.params["render"]:
                self.envs.render()
                if self.params["render_delay"] > 0:
                    time.sleep(self.params["render_delay"])
        # except MujocoException as e:
        #     logger.error("We got a MuJoCo exception!")
        #     raise
        #     ## Retry??
        #     # return self.run()
        
        with KeepTime("poststep"):
            # TODO: Sometimes the type of observations is "dict" which shouldn't be. Investigate the reason.
            if isinstance(self.state["observations"], OrderedDict) or isinstance(self.state["observations"], dict):
                for key in self.state["observations"]:
                    if np.isnan(self.state["observations"][key]).any():
                        logger.warn('NaN caught in observations during rollout generation.', 'step =', self.state["steps"])
                        raise ValueError
            else:
                if np.isnan(self.state["observations"]).any():
                    logger.warn('NaN caught in observations during rollout generation.', 'step =', self.state["steps"])
                    raise ValueError
                ## Retry??
                # return self.run()

            self.state["steps"] += 1
            self.state["timesteps"] += self.params["num_workers"]
            self.monitor_timesteps()
            # TODO: Adapt with the new dict_of_lists data structure.
            with KeepTime("report_reward"):
                self.report_rewards(infos)

            transition = dict(**pre_transition,
                              rewards=rewards,
                              infos=infos)
        return transition
        

    def update(self):
        """Runs :func:`step` for ``n_steps`` times.

        Returns:
            dict: A dictionary of unix-stype file system keys including all information generated by the simulation.
        
        See Also:
            :ref:`ref-data-structure`
        """

        # trajectory is a dictionary of lists
        trajectory = {}

        if not self.state["was_reset"] and self.params["do_reset"]:
            self.reset()

        self.state["was_reset"] = False

        # Run T (n-step) steps.
        self.local["steps"] = 0
        self.local["n_episode"] = 0

        while (self.params["n_steps"]    and self.local["steps"]     < self.params["n_steps"]) or \
              (self.params["n_episodes"] and self.local["n_episode"] < self.params["n_episodes"]):
            with KeepTime("step"):
                # print("one exploration step ...")
                transition = self.step()

            with KeepTime("append"):
                # Data is flattened in the explorer per se.
                transition = flatten_dict(transition)
                # Update the trajectory with the current list of data.
                # Put nones if the key is absent.
                update_dict_of_lists(trajectory, transition, index=self.local["steps"])

            self.local["steps"] += 1

        with KeepTime("poststep"):
            # Take one prestep so we have the next observation/hidden_state/masks/action/value/ ...
            transition = self.prestep(final_step=True)
            transition = flatten_dict(transition)
            update_dict_of_lists(trajectory, transition, index=self.local["steps"])

            # Complete the trajectory if one key was in a transition, but did not occur in later
            # transitions. "length=n_steps+1" is because of counting final out-of-loop prestep.
            
            # complete_dict_of_list(trajectory, length=self.params["n_steps"]+1)
            complete_dict_of_list(trajectory, length=self.local["steps"]+1)
            result = convert_time_to_batch_major(trajectory)
        
        # We discard the rest of monitored episodes for the test mode to prevent them from affecting next test.
        monitor.discard_key("/reward/test/episodic")
        return result


### Data Structure:
# Pre-step:
#   observations
#   masks:
#
# Agent (policies):
#   actions
#   hidden_state
#   artifacts:
#       action_log_p
#       value
#
# Step:
#   rewards
#   infos

######################
##### Statistics #####
######################
# Stats: Wall-time
