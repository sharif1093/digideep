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

        # Create models
        menv = MakeEnvironment(session, mode=self.params["mode"], seed=self.params["seed"], **self.params["env"])
        self.envs = menv.create_envs(num_workers=self.params["num_workers"])
        
        self.state = {}
        self.state["steps"] = 0
        self.state["n_episode"] = 0
        self.state["timesteps"] = 0
        self.state["was_reset"] = False

        # We only reset once. Later environments will be reset automatically.
        self.reset()
        # Will the results be reported when using ``do_reset``?`

    def state_dict(self):
        # TODO" Should we make a deepcopy?
        return {"state":self.state, "envs":self.envs.state_dict()}
    def load_state_dict(self, state_dict):
        self.state.update(state_dict["state"])
        self.envs.load_state_dict(state_dict["envs"])
        if not self.params["mode"]=="train":
            # We reset the explorer in case of test/eval to clear the history of observations/masks/hidden_state.
            # Because this part does not make sense to be transferred.
            self.reset()

    def report_rewards(self, infos):
        """This function will extract episode information from infos and will send them to
        :class:`~digideep.utility.monitoring.Monitor` class.
        """
        # print(infos)
        if '/episode/r' in infos.keys():
            rewards = infos['/episode/r']
            for rew in rewards:
                if not np.isnan(rew):
                    self.state["n_episode"] += 1
                    monitor("/explore/reward/"+self.params["mode"], rew)
        #     print("Everything is here:", episode)
        #     # for e in 
        #     # self.state["n_episode"] += 1
        #     # r = info['episode']['r']
        #     # monitor("/explore/reward/"+self.params["mode"], r)
        
        # for info in infos:
            # This episode keyword only exists if we use a Monitor wrapper.
            # This keyword will only appear at the "reset" times.
            # TODO: If this is a true multi-agent system, then the rewards
            #       must be separated as well!
            

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

        with KeepTime("/explore/step/prestep/to_numpy"):
            # TODO: Is it necessary for conversion of obs?
            # NOTE: The np conversion will not work if observation is a dictionary.
            # observations = np.array(self.state["observations"], dtype=np.float32)
            observations = self.state["observations"]
            masks = self.state["masks"]
            hidden_state = self.state["hidden_state"]

        with KeepTime("/explore/step/prestep/gen_action"):
            publish_agents = True 
            agents = {}
            for agent_name in self.agents:
                if (not final_step) or (self.params["final_action"]):
                    action_generator = self.agents[agent_name].action_generator
                    agents[agent_name] = action_generator(observations, hidden_state[agent_name], masks, deterministic=self.params["deterministic"])
                else:
                    publish_agents = False
                # We are saving the "new" hidden_state now.
            
        with KeepTime("/explore/step/prestep/form_dictionary"):
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
        with KeepTime("/explore/step/prestep"):
            pre_transition = self.prestep()
            
        # TODO: For true multi-agent systems, rewards must be a dictionary as well,
        #       i.e. one reward for each agent. However, if the agents are pursuing
        #       a single goal, the reward can still be a single scalar!

        # Updating observations and masks: These two are one step old in the trajectory.
        # hidden_state is the newest.
        
        with KeepTime("/explore/step/envstep"):
            # Prepare actions
            actions = extract_keywise(pre_transition["agents"], "actions")

            # Step
            self.state["observations"], rewards, dones, infos = self.envs.step(actions)
            # Post-step
            self.state["hidden_state"] = extract_keywise(pre_transition["agents"], "hidden_state")
            self.state["masks"] = np.array([0.0 if done_ else 1.0 for done_ in dones], dtype=np.float32).reshape((-1,1))

        # TODO: Adapt with the new dict_of_lists data structure.
        with KeepTime("/explore/step/report_reward"):
            self.report_rewards(infos)

        with KeepTime("/explore/step/render"):
            if self.params["render"]:
                self.envs.render()
                if self.params["render_delay"] > 0:
                    time.sleep(self.params["render_delay"])
        # except MujocoException as e:
        #     logger.error("We got a MuJoCo exception!")
        #     raise
        #     ## Retry??
        #     # return self.run()
        
        with KeepTime("/explore/step/poststep"):
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

        # counter_exit = 0

        # Run T (n-step) steps.
        for t in range(self.params["n_steps"]):
            with KeepTime("/explore/step"):
                # print("one exploration step ...")
                transition = self.step()
                
                # if counter_exit<10:
                #     print("c=", counter_exit, " --> ", transition)
                #     counter_exit += 1
                # else:
                #     exit()

            with KeepTime("/explore/append"):
                # Data is flattened in the explorer per se.
                transition = flatten_dict(transition)
                # Update the trajectory with the current list of data.
                # Put nones if the key is absent.
                update_dict_of_lists(trajectory, transition, index=t)

        with KeepTime("/explore/poststep"):
            # Take one prestep so we have the next observation/hidden_state/masks/action/value/ ...
            transition = self.prestep(final_step=True)
            transition = flatten_dict(transition)
            update_dict_of_lists(trajectory, transition, index=t)

            # Complete the trajectory if one key was in a transition, but did not occur in later
            # transitions. "length=n_steps+1" is because of counting final out-of-loop prestep.
            complete_dict_of_list(trajectory, length=self.params["n_steps"]+1)
            result = convert_time_to_batch_major(trajectory)

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
