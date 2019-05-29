"""This module provides helper functions for registering ``dm_control`` environments. It can be
used both for user-defined and dm_control suite environments.

Note:
    To import all dm_control suite environments, run ``import digideep.environment.dmc2gym``.
"""

from dm_control import suite
from .wrapper import DmControlWrapper
from gym.utils import EzPickle


# When registration is done using this function, the parameters are different from
# the above. We no more use "dmcenv", "domain_name", or "task_name".
class EnvCreator(EzPickle):
    """Class for registering user-defined dm_control environments.

    Args:
        task: The dm_control task.
        task_kwargs (dict): The arguments used for the task.
        environment_kwargs (dict): The keywords that will be passed to the environment maker function.
        visualize_reward (bool): Whether to visualize rewards in the viewer or not.
    """
    def __init__(self, task, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
        self.task = task
        self.task_kwargs = task_kwargs
        self.environment_kwargs = environment_kwargs
        self.visualize_reward = visualize_reward
        EzPickle.__init__(self, task, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs, visualize_reward=visualize_reward)
    
    def __call__(self, **extra_env_kwargs):
        """
        Returns:
            :obj:`dm_control.rl.control.Environment`: The ``dm_control`` environment.
        """
        task_kwargs_subs = self.task_kwargs or {}

        if extra_env_kwargs:
            task_kwargs_subs.update(extra_env_kwargs)

        if (self.environment_kwargs is not None) or not (self.environment_kwargs == {}):
            task_kwargs_subs = task_kwargs_subs.copy()
            task_kwargs_subs['environment_kwargs'] = self.environment_kwargs
        dmcenv = self.task(**task_kwargs_subs)
        dmcenv.task.visualize_reward = self.visualize_reward
        return dmcenv


####################
## For Benchmarks ##
####################
class EnvCreatorSuite(EzPickle):
    """
    Class for registering dm_control suite environments.

    Args:
        task: The dm_control task.
        task_kwargs (dict): The arguments used for the task.
        environment_kwargs (dict): The keywords that will pass to the environment maker function.
        visualize_reward (bool): Whether to visualize rewards in the viewer or not.
    """
    def __init__(self, domain_name, task_name, task_kwargs={}, environment_kwargs=None, visualize_reward=False):
    # def __init__(self, domain_name, task_name, task_kwargs={}, environment_kwargs={}, visualize_reward=False):
        self.domain_name = domain_name
        self.task_name = task_name
        self.task_kwargs = task_kwargs
        self.environment_kwargs = environment_kwargs
        self.visualize_reward = visualize_reward
        EzPickle.__init__(self, domain_name, task_name, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs, visualize_reward=visualize_reward)

    def __call__(self, **extra_env_kwargs):
        """
        Returns:
            :obj:`dm_control.rl.control.Environment`: The ``dm_control`` environment.
        """
        if extra_env_kwargs:
            self.task_kwargs.update(extra_env_kwargs)

        dmcenv = suite.load(domain_name=self.domain_name,
                            task_name=self.task_name,
                            task_kwargs=self.task_kwargs,
                            environment_kwargs=self.environment_kwargs,
                            visualize_reward=self.visualize_reward)
        return dmcenv

