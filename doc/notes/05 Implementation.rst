=========================================
Developer Guide: Implementation Guideline
=========================================

To implement a new method you need to get a pipeline working as soon as possible.
Digideep helps in that manner with developer-friendly source codes, i.e. extensive
comments and documentation besides self-descriptive code. The pipeline does not need
to train any policies at the beginning. 

Digideep is very modular, so that you can use your own implementation for any part instead.
However, you are encouraged to fork the source on work on your own copy of the source code
for deeper modifications.


Implementation steps
--------------------

1. Create a parameter file for your method. You may leave parts that you have not implemented yet blank.
   Take a look at :mod:`digideep.params` for some examples of parameters file or see the descriptions in
   :ref:`ref-parameter-files`.
2. Create a class for your agent. Inherit from the :class:`~digideep.agent.base.AgentBase`.
3. Override :func:`~digideep.agent.base.AgentBase.action_generator` function in your agent's class.
   Explorer will call this function to generate actions. Follow the expected interface described at
   :func:`~digideep.agent.base.AgentBase.action_generator`. You can generate random actions but in
   the correct output shape to get the pipeline done faster.

.. tip::

    Complete your parameters file as you move forward. Run the program early.
    Try to debug the interface issues as soon as possible.

4. In your agent's class, override reset_hidden_state if you are planning to use recursive policies.
5. Now, the explorer should work fine, and the trajectories may be stored in the memory. Now, it is time
   to start implementation of your policy.

.. note::

    You should first make sure of correct flow of information through
    components of the runner, i.e. explorer, memory, and agent, then try
    to implement the real algorithms. The :class:`~digideep.environment.explorer.Explorer`
    and :class:`~digideep.memory.generic.Memory` classes are general classes which can be
    used with different algorithms.
    
6. To implement your policy, you can inherit from :class:`~digideep.policy.base.PolicyBase`.
7. When implementation of policy is done, modify :func:`~digideep.agent.base.AgentBase.action_generator`
   in your agent to generate actions based on the policy.
8. When policy is done, it's time to implement the sampler for your method. The sampler is typically
   used at the beginning of the :func:`~digideep.agent.base.AgentBase.step` function of the agent.
   At the same time, :func:`~digideep.agent.base.AgentBase.update` function can be implemented. It is
   usually just a loop of calls on the :func:`~digideep.agent.base.AgentBase.step` function.
9. At this point, you have successfully finished implementation of your agent. Now it's time to debug.
   You may use the :class:`~digideep.utility.profiling.Profiler` and :class:`~digideep.utility.monitoring.Monitor`
   tools to inspect the values inside your code and watch the timings.

