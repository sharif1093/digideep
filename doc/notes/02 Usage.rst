=====
Usage
=====

Training/Replaying
------------------

.. command-output:: python -m digideep.main --help

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
        --cpanel '{"model_name":"DMBenchCheetahRun-v0"}'

    python -m digideep.main --params digideep.params.mujoco_ppo \
        --cpanel '{"model_name":"DMBenchCheetahRun-v0", "recurrent":True}'
    

.. code-block:: bash
    :caption: Loading a checkpoint to play

    python -m digideep.main --play --load-checkpoint \
           "/tmp/digideep_sessions/session_YYYYYMMDDHHMMSS/checkpoints/checkpoint-XXX"


.. _ref-play-debug:

Playing for Debugging
---------------------

.. command-output:: python -m digideep.environment.play --help



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
