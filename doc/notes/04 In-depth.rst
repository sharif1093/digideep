=====================================
Developer Guide: In-Depth Information
=====================================

In this section, we cover several topics which are essential to understanding how Digideep works.


.. _ref-parameter-files:

Understanding the parameters file
---------------------------------

There are two sections in a parameter file. The main section is the ``def gen_params(cpanel)`` function, which gets the ``cpanel``
dictionary as its input, and gives the ``params`` dictionary as the output. The ``params`` dictionary is the parameter tree
of all classes in the project, all in one place. This helps to see the whole structure of the code in one place and have control
over them from a centralized location. Moreover, it allows for scripting the parameter relationships, in a more transparent way.
Then, there is the ``cpanel`` dictionary for modifying important parameters from a "control panel". The ``cpanel`` dictionary may
be modified through command-line access:

.. code-block:: bash

    python -m digideep.main ... --cpanel '{"cparam1":"value1", "cparam2":"value2"}'


.. note::

    It was possible to implement the parameter file using ``json`` or ``yaml`` files. But then it was less intuitive to script the
    relationships between coupled parameters.



.. _ref-data-structure:

Understanding the data structure of trajectories
------------------------------------------------

The output of the :class:`~digideep.environment.explorer.Explorer`, trajectories, are organized in the form of a dictionary with
the following structure:

.. code-block:: python

    {'/observations':(batch_size, n_steps, ...),
    '/masks':(batch_size,n_steps,1),
    '/rewards':(batch_size,n_steps,1),
    '/infos/<info_key_1>':(batch_size,n_steps,...),
    '/infos/<info_key_2>':(batch_size,n_steps,...),
    ...,
    '/agents/<agent_1_name>/actions':(batch_size,n_steps,...),
    '/agents/<agent_1_name>/hidden_state':(batch_size,n_steps,...),
    '/agents/<agent_1_name>/artifacts/<artifact_1_name>':(batch_size,n_steps,...),
    '/agents/<agent_1_name>/artifacts/<artifact_2_name>':(batch_size,n_steps,...),
    ..., 
    '/agents/<agent_2_name>/actions':(batch_size,n_steps,...),
    '/agents/<agent_2_name>/hidden_state':(batch_size,n_steps,...),
    '/agents/<agent_2_name>/artifacts/<artifact_1_name>':(batch_size,n_steps,...),
    '/agents/<agent_2_name>/artifacts/<artifact_2_name>':(batch_size,n_steps,...),
    ...
    }

Here, ``batch_size`` is the number of concurrent workers in the :class:`~digideep.environment.explorer.Explorer` class, and 
``n_steps`` is the length of each trajectory, i.e. number of timesteps the environment is run.

.. note::

    The names in angle brackets are arbitrary, depending on the agent and environment.

Here's what each entry in the output mean:

* ``/observations``: Observations from the environment.
* ``/masks``: The ``done`` flags of the environment. A ``mask`` value of ``0`` indicates "finished" episode.
* ``/rewards``: The rewards obtained from the environment.
* ``/infos/*``: Optional information produced by the environment.
* ``/agents/<agent_name>/actions``: Actions took by ``<agent_name>``.
* ``/agents/<agent_name>/hidden_state``: Hidden_states of ``<agent_name>``.
* ``/agents/<agent_name>/artifacts/*``: Optional outputs from the agents which includes additional information required for training.

:class:`~digideep.memory.generic.Memory` will preserve the format of this data structure and store it as it is.
:class:`~digideep.memory.generic.Memory` is basically a queue; new data will replace old data when queue is full.



Understanding the structure of agents
-------------------------------------

Digideep supports multiple agents in an environment. Agents are responsible to generate exploratory actions
and update their parameters. Agents should inherit :class:`~digideep.agent.base.AgentBase`. There are two important
components in a typical component: sampler and policy.

.. note::

    The interface of the agent class with the :class:`~digideep.environment.explorer.Explorer` is the 
    :func:`~digideep.agent.base.AgentBase.action_generator`. This function is called to generate actions
    in the environment. The interface of the agent class with the :class:`~digideep.pipeline.runner.Runner`
    class is the :func:`~digideep.agent.base.AgentBase.update` class. This function is meant to update
    the parameters of the agent policy based on collected information from the environment.

As an example of agents, refer to :mod:`~digideep.agent.ppo.PPO` or :mod:`~digideep.agent.ddpg.DDPG`.

Sampler
^^^^^^^

A sampler samples transitions from the memory to train the policy on. Samplers for different methods share similar
parts, thus suggesting to decompose a sampler into smaller units. This obviates developers from some boilerplate coding.
See :mod:`digideep.memory.sampler` for some examples.

Policies
^^^^^^^^

Policy is the function inside an agent that generates actions. A policy should inherit from :class:`~digideep.policy.base.PolicyBase`.
Policies support multi-GPU architectures for inference and architecture. We use :class:`torch.nn.DataParallel` to activate multi-GPU
functionalities. Note that using multi-GPUs sometimes does not lead to faster computations, due to larger overheads with respect to
gains. It is really problem-dependant.

Every policy should implement the :func:`~digideep.policy.base.PolicyBase.generate_actions` function. This function is to be called in
the agent's :func:`~digideep.agent.base.AgentBase.action_generator`.

For examples on policies, refer to two available policies in Digideep:

* A stochastic :class:`~digideep.policy.stochastic.policy.Policy` for :mod:`~digideep.agent.ppo.PPO` agent.
* A deterministic :class:`~digideep.policy.deterministic.policy.Policy`for :mod:`~digideep.agent.ddpg.DDPG` agent.


Understanding serialization
---------------------------

Digideep is written with serialization in mind from the beginning. The main burden of serialization is on the
:class:`~digideep.pipeline.runner.Runner` class. It saves both the parameters and states of its sub-components:
explorer, memory, and agents. Each of these sub-components are responsible for saving their sub-components states,
i.e. in a recursive manner.

.. caution::

    By now, checkpoints only save object states that are necessary for playing the policy, not to resume training.

At each instance of saving two pickle objects are saved, one saving the :class:`~digideep.pipeline.runner.Runner`,
the other saving the states. "Saving", at its core, is done by using ``pickle.dump`` for the
:class:`~digideep.pipeline.runner.Runner` and ``torch.save`` for the states in the session class.
"Loading", uses counterpart functions ``pickle.load`` and ``torch.load`` for the :class:`~digideep.pipeline.runner.Runner`
and states, respectively.

.. note::

    If you are implementing a new method, you should implement your own ``state_dict`` and ``load_state_dict`` methods for saving the
    state of "stateful" objects. Make sure those are called properly during saving and loading.


Debugging tools
---------------

There are some tools commonly used while implementing a reinforcement learning method. We have provided the following assistive tools
to help developers debug their codes:

* :class:`digideep.utility.profiling.Profiler`: A lightweight profiling tool. This will help find parts of code that irregularly take more
  time to complete.
* :class:`digideep.utility.monitoring.Monitor`: A lightweight monitoring tool to keep track of values of variables in training.
* Debugging tools in :mod:`digideep.memory.sampler`: There a few sampler units that can be injected into the sampler to inspect shapes, NaN
  values, and means and standard deviations of a chunk of memory.
* Monitoring CPU/GPU utilization of cores and memory. See :mod:`~digideep.utility.stats` and
  :func:`~digideep.pipeline.session.Session.runMonitor`.


Documentation
-------------

We use Sphinx for documentation. If you are not familiar with the syntax, follow the links below:

* Cheat sheet for Google/Numpy style: http://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
* Basics of reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
* Example Google Style: https://www.sphinx-doc.org/en/1.7/ext/example_google.html

