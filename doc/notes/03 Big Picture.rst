============================
Developer Guide: Big Picture
============================

The session and runner
----------------------

The entrypoint of the program is the ``main.py`` module. This module, first creates a :class:`~digideep.pipeline.session.Session`.

A :class:`~digideep.pipeline.session.Session` is responsible for command-line arguments, creating a directory for saving the
all results related to that session (logs, checkpoints, ...), and initiating the assitive tools, e.g. loggers, monitoring tools,
visdom server, etc.

After the :class:`~digideep.pipeline.session.Session` object is created, a :class:`~digideep.pipeline.runner.Runner` object is
built, either from an existing checkpoint or from the parameters file specified at the command-line. The runner class will run
the main loop.


.. _ref-how-runner-works:

How does runner work
--------------------

The :class:`~digideep.pipeline.runner.Runner` depends on three main classes: :class:`~digideep.environment.explorer.Explorer`,
:class:`~digideep.memory.generic.Memory`, and :class:`~digideep.agent.base.AgentBase`. The connection between these classes
is really simple (and is intentionally written to be so), as depicted in the following general graph about reinforcement learning:


.. code-block:: text

    +-------------+               +--------+
    |   Explorer  | ------------> | Memory |
    +-------------+               +--------+
           ^                           |
           | (ACTIONS)                 | (TRAJECTORIES)
           |                           | 
    +------------------------------------------+
    |      |                           |       |
    |      |                      +---------+  |
    |      |                      | SAMPLER |  |
    |      |                      +---------+  |
    |      |                           |       |
    |      |     (SAMPLED TRANSITIONS) |       |
    |      |         ----------        |       |
    |      | <------ | POLICY | <----- |       |
    |                ----------                |
    +------------------------------------------+
                      AGENT
          

The corresponding (pseudo-)code for the above graph is:

.. code-block:: python

    do in loop:
        chunk = self.explorer["train"].update()
        self.memory.store(chunk)
        for agent_name in self.agents:
            self.agents[agent_name].update()

* :class:`~digideep.environment.explorer.Explorer`: Explorer is responsible for multi-worker environment simulations.
  It delivers the outputs to the memory in the format of a flattened dictionary (with depth 1). The explorer is tried to
  be written in its most general manner so it needs least possible modifications for adaptation to new methods.
* :class:`~digideep.memory.generic.Memory`: It stores all of the information from the explorer in a dictionary of numpy
  arrays. The memory is also written in a very general way, so it is usable with most of the methods without modifications.
* :mod:`~digideep.agent`: The agent uses :mod:`~digideep.memory.sampler` and :mod:`~digideep.policy`, and is responsible
  for training the policy and generating actions for simulations in the environment.
