============
Installation
============

Requirements
------------

* Python 3
* `PyTorch <https://pytorch.org/>`_ 
* [OPTIONAL] `Tensorboard <https://www.tensorflow.org/tensorboard>`_.
* `MuJoCo <https://www.roboti.us/index.html>`_ ``v200``.
* `mujoco_py <https://github.com/openai/mujoco-py>`_ and `Gym <https://github.com/openai/gym>`_.
* `dm_control <https://github.com/deepmind/dm_control>`_.

.. note::
    If you are a student, you can get a free student license for MuJoCo.

Installation
------------

Simply download the package using the following command and add it to your ``PYTHONPATH``:

.. code-block:: bash
    cd
    git clone https://github.com/sharif1093/digideep.git
    cd digideep
    pip install -e .


Set your environment
--------------------

Add the following to your ``.bashrc`` or ``.zshrc``:

.. code-block:: bash

    # Assuming you have installed mujoco in '$HOME/.mujoco'
    export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200_linux/bin:$LD_LIBRARY_PATH
    export MUJOCO_GL=glfw


.. _FixGLFW:

Patch ``dm_control`` initialization issue
-----------------------------------------

If you hit an error regarding GLFW initialization, try the following patch: 

Go to the ``digideep`` installation path and run:

.. code-block:: python

    cd <digideep_path>
    cp patch/glfw_renderer.py `pip show dm_control | grep -Po 'Location: (\K.*)'`/dm_control/_render
