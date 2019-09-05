============
Installation
============

Prerequisites
-------------

* Install `PyTorch <https://pytorch.org/>`_ and `Visdom <https://github.com/facebookresearch/visdom>`_.
* Install `MuJoCo <https://www.roboti.us/index.html>`_ ``v150`` and ``v200``.
* Install `mujoco_py <https://github.com/openai/mujoco-py>`_ and `Gym <https://github.com/openai/gym>`_.
* Install `dm_control <https://github.com/deepmind/dm_control>`_.

.. note::
    If you are a student, you can use the free Student License for MuJoCo.

Installation
------------

Simply download the package using the following command and add it to your ``PYTHONPATH``:


.. code-block:: bash

    git clone https://github.com/sharif1093/digideep.git
    cd digideep
    pip install -e .


Set your environment
--------------------

Add the following to your ``.bashrc`` or ``.zshrc``:

.. code-block:: bash

    # Assuming that you have installed mujoco in '$HOME/.mujoco'
    export LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
    export MUJOCO_GL=glfw  

