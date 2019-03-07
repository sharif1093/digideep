"""
This module is inspired by `gym-dmcontrol <https://github.com/rejuvyesh/gym-dmcontrol>`_
and `dm_control2gym <https://github.com/martinseilair/dm_control2gym>`_.
"""

from gym import spaces
# spaces.Dict | spaces.Tuple | spaces.Box | spaces.Discrete | spaces.MultiDiscrete | spaces.MultiBinary
from dm_control.rl import specs
import numpy as np
import collections
# import warnings

def spec2space_single(spec):
    """
    Args:
        spec: A single dm_control ``spec``.

    Returns:
        :obj:`gym.spaces`: The ``gym`` equivalent ``spaces``.
    """

    if (type(spec) is specs.BoundedArraySpec) and (spec.dtype == np.int):
    # Discrete
        # warnings.warn("The DMC environment uses a discrete action space!")
        if spec.minimum == 0:
            return spaces.Discrete(spec.maximum)
        else:
            raise ValueError("The environment's minimum values must be zero in the Discrete case!")
    # Box
    elif type(spec) is specs.BoundedArraySpec:
        _min = np.broadcast_to(spec.minimum, shape=spec.shape)
        _max = np.broadcast_to(spec.maximum, shape=spec.shape)
        # if clip_inf:
        #     _min = np.clip(_min, -sys.float_info.max, sys.float_info.max)
        #     _max = np.clip(_max, -sys.float_info.max, sys.float_info.max)
        return spaces.Box(_min, _max, dtype=np.float32)        
    elif type(spec) is specs.ArraySpec:
        return spaces.Box(-np.inf, np.inf, shape=spec.shape, dtype=np.float32)
    else:
        raise ValueError('Unknown spec in spec2space_single!')

def spec2space(spec):
    """This function converts the ``spec`` of a ``dm_control`` to its ``gym.spaces`` equivalent.

    Caution:
        Currently it supports ``spaces.Discrete``, ``spaces.Box``, and ``spaces.Dict`` as outputs.
    """
    if isinstance(spec, specs.ArraySpec) or isinstance(spec, specs.BoundedArraySpec):
        return spec2space_single(spec)
    elif isinstance(spec, collections.OrderedDict):
        space = collections.OrderedDict()
        for k,s in spec.items():
            space[k] = spec2space(s)
        return spaces.Dict(space)
    else:
        raise ValueError("Unknown spec in spec2space!")

