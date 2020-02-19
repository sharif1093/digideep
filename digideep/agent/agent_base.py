from abc import abstractmethod
import numpy as np

class AgentBase:
    """
    This is the base class for agents. Also see the already implemented agents for more details:
    :mod:`~digideep.agent.ddpg` and :mod:`~digideep.agent.ppo`.
    
    Note:
        To make your agent serializable, implement the :func:`state_dict` and :func:`load_state_dict` functions.
        Otherwise you will not be able to save and play policies.
    
    Tip:
        :func:`step` method typically performs a single step of update on the agent's policy.
    
    Note:
        The :func:`update` method is called by :class:`~digideep.pipeline.runner.Runner` and is responsible for updating
        the agent's polic. Typically, :func:`update`  is a function that calls :func:`step` multiple times. It can do
        pre-processing and post-processing as well. For example, in DDPG, :func:`~digideep.agent.ddpg.DDPG.update` also
        updates the target networks.
    """

    def __init__(self, session, memory, **params):
        self.session = session
        self.memory = memory
        self.params = params
        self.state = {}
    
    @abstractmethod
    def state_dict(self):
        """
        Returns:
            dict: The state of the current agent and all components that have states.
        """
        pass
    @abstractmethod
    def load_state_dict(self, state_dict):
        """Loads the states of the object.
        """
        pass

    def reset_hidden_state(self, num_workers):
        """
        Returns:
            :obj:`np.ndarray`: A Numpy zero array with the true size of the hidden state of the agent's policy.
        """
        hidden_size = 1
        return np.zeros((num_workers, hidden_size), dtype=np.float32)
    
    def random_action_generator(self, envs, num_workers):
        actions = np.array([envs.action_space.spaces[self.params["name"]].sample() for i in range(num_workers)], dtype=np.float32)
        hidden_state = self.reset_hidden_state(num_workers)
        return dict(actions=actions, hidden_state=hidden_state)
    
    @abstractmethod
    def action_generator(self, observations, hidden_state, masks, deterministic=False):
        """
        All agents should override :func:`action_generator` with the same signature.
        :class:`~digideep.environment.explorer.Explorer` calls this function to generate actions.

        Returns:
            dict: The output of this function must follow this data-structure:
            ``{'actions':actions, 'hidden_state':hidden_state, 'artifacts':{...}}``

        """
        pass
    
    @abstractmethod
    def step(self):
        """The function called by :func:`update` to update the agent's policy parameters for *one step*.
        """
        pass
    
    @abstractmethod
    def update(self):
        """The function called by :class:`~digideep.pipeline.runner.Runner` to update the agent's policy parameters (train).
        """
        pass
