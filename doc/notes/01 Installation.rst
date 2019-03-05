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

.. warning::
    Because of an issue in ``dm_control``'s ``glfw`` initialization, the regular version
    of ``dm_control`` will not work with ``Digideep``. To fix the issue check :ref:`FixGLFW`.


Installation
------------

Simply download the package using the following command and add it to your ``PYTHONPATH``:


.. code-block:: bash

    git clone https://github.com/sharif1093/digideep.git
    cd digideep
    pip install -e .


.. _FixGLFW:

Patch ``dm_control`` initialization issue
-----------------------------------------

Go to the ``digideep`` installation path. Run the following:

.. code-block:: python

    cd <digideep_path>
    cp patch/glfw_renderer.py `pip show dm_control | grep -Po 'Location: (\K.*)'`/dm_control/_render

Or, alternatively go to the ``dm_control`` installation path, and find the ``dm_control/_render/glfw_renderer.py`` file.
In that file, move the following block of code to the beginning of the ``def _platform_init(self, max_width, max_height)`` function:

.. code-block:: python

    try:
      glfw.init()
    except glfw.GLFWError as exc:
      _, exc, tb = sys.exc_info()
      six.reraise(ImportError, ImportError(str(exc)), tb)

.. note::

    This change is required to use ``dm_control`` in multi-worker environments. The master thread should never initialize the ``glfw``
    like this if it is going to be forked to parallel threads.


Set your environment
--------------------

Add the following to your ``.bashrc`` or ``.zshrc``:

.. code-block:: bash

    # Assuming that you have installed mujoco in '$HOME/.mujoco'
    export LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
    export MUJOCO_GL=glfw  

