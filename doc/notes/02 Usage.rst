=====
Usage
=====

Training/Replaying
------------------

.. code-block:: text
    :caption: Command-line arguments

    $ python -m digideep.main --help
    usage: main.py [-h] [--load-checkpoint <path>] [--play]
                    [--session-path <path>] [--save-modules <path> [<path> ...]]
                    [--log-level <n>] [--visdom] [--visdom-port <n>]
                    [--monitor-cpu] [--monitor-gpu] [--params <name>]
                    [--cpanel <json dictionary>]

    optional arguments:
        -h, --help            show this help message and exit
        --load-checkpoint <path>
                              Load a checkpoint to resume training from that point.
        --play                Will play the stored policy.
        --session-path <path>
                              The path to store the sessions. Default is in /tmp
        --save-modules <path> [<path> ...]
                              The modules to be stored in the session.
        --log-level <n>       The logging level: 0 (debug and above), 1 (info and
                              above), 2 (warn and above), 3 (error and above), 4
                              (fatal and above)
        --visdom              Whether to use visdom or not!
        --visdom-port <n>     The port of visdom server, it's on 8097 by default.
        --monitor-cpu         Use to monitor CPU resource statistics on Visdom.
        --monitor-gpu         Use to monitor GPU resource statistics on Visdom.
        --params <name>       Choose the parameter set.
        --cpanel <json dictionary>
                              Set the parameters of the cpanel by a json dictionary.


.. code-block:: bash
    :caption: Example Usage

    # Start a training session for a MuJoCo environment using DDPG 
    # Default environment is "Pendulum-v0"
    python -m digideep.main --params digideep.params.classic_ddpg
    
    # Start a training session for an Atari environment using PPO
    # Default environment is "PongNoFrameskip-v4"
    python -m digideep.main --params digideep.params.atari_ppo
    
    # Start a training session for a MuJoCo environment using PPO
    # Default environment is "Ant-v2"
    python -m digideep.main --params digideep.params.mujoco_ppo

    # Change the parameters in command-line
    python -m digideep.main --params digideep.params.mujoco_ppo \
        --cpanel '{"model_name":"DMBenchCheetahRun-v0", "from_module":"digideep.environment.dmc2gym"}'

    python -m digideep.main --params digideep.params.mujoco_ppo \
        --cpanel '{"model_name":"DMBenchCheetahRun-v0", "from_module":"digideep.environment.dmc2gym", "recurrent":True}'
    

.. code-block:: bash
    :caption: Loading a checkpoint to play

    # Typical loading
    python -m digideep.main --play --load-checkpoint "<path-to-checkpoint>"
    
    # Loading a checkpoint using its saved modules (through --save-modules option)
    PYTHONPATH="<path-to-session>/modules" python -m digideep.main --play --load-checkpoint "<path-to-checkpoint>"


.. _ref-play-debug:

Playing for Debugging
---------------------

.. code-block:: text
    :caption: Command-line arguments

    $ python -m digideep.environment.play --help
    usage: play.py [-h] [--list-include [<pattern>]] [--list-exclude [<pattern>]]
                  [--module <module_name>] [--model <model_name>] [--runs <n>]
                  [--n-step <n>] [--delay <ms>] [--no-action]

    optional arguments:
      -h, --help            show this help message and exit
      --list-include [<pattern>]
                            List by a pattern
      --list-exclude [<pattern>]
                            List by a pattern
      --module <module_name>
                            The name of the module which will register the model
                            in use.
      --model <model_name>  The name of the model to play with random actions.
      --runs <n>            The number of times to run the simulation.
      --n-step <n>          The number of timesteps to run each episode.
      --delay <ms>          The time in milliseconds to delay in each timestep to
                            make simulation slower.
      --no-action           The number of timesteps to run each episode.



.. code-block:: bash
    :caption: Running a model with random actions

    python -m digideep.environment.play --model "Pendulum-v0"

.. code-block:: bash
    :caption: Running a model with no actions

    python -m digideep.environment.play --model "Pendulum-v0" --no-action

.. code-block:: bash
    :caption: Running a model from another module (your custom designed environment).

    python -m digideep.environment.play --model "<model-name>" --module "<module-name>"

.. code-block:: bash
    :caption: List registered modules

    python -m digideep.environment.play --list-include ".*"
    python -m digideep.environment.play --list-include ".*Humanoid.*"
    python -m digideep.environment.play --list-include ".*Humanoid.*" --list-exclude "DM*"
