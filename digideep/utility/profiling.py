from collections import OrderedDict as odict
import time
import numpy as np

SHOULD_PROFILE = True

class Profiler:
    """This class provides a very simple yet light implementation of function profiling.
    It is very easy to use:

        >>> profiler.reset()
        >>> profiler.start("loop")
        >>> for i in range(100000):
        ...   print(i)
        ... 
        >>> profiler.lapse("loop")
        >>> print(profiler)
        >> loop [1x, 27.1s]
    
    Alternatively, you may use ``profiler`` with :class:`KeepTime`:

        >>> with KeepTime("loop2"):
        ...   for i in range(100000):
        ...     print(i)
        ...
        >>> print(profiler)
        >> loop2 [1x, 0.0s]
    
    Note:
        The number of callings to :func:`start` and :func:`lapse` should be the same.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.starts = odict()
        self.totals = odict()
        self.occurs = odict()
    
    def start(self, name):
        if name in self.starts:
            assert self.starts[name] == None, "The previous start should be lapsed first for [{:s}]".format(name)
        self.starts[name] = time.time()

    def lapse(self, name):
        assert name in self.starts and self.starts[name] is not None, "You should first start for [{:s}]".format(name)
        elapsed = time.time()-self.starts[name]

        if name in self.totals:
            self.totals[name] += elapsed
            self.occurs[name] += 1
        else:
            self.totals[name] = elapsed
            self.occurs[name] = 1

        self.starts[name] = None

    def get_time_average(self, name):
        assert name in self.totals
        return self.totals[name]/self.occurs[name]

    def get_time_overall(self, name):
        assert name in self.totals
        return self.totals[name]

    def get_occurence(self, name):
        assert name in self.totals
        return self.occurs[name]

    def get_keys(self):
        return list(self.starts.keys())
    
    def __repr__(self):
        res = ""
        for k in self.get_keys():
            res += (">> {:s} [{:d}x, {:.1f}s]\n".format(k, self.get_occurence(k), self.get_time_overall(k)))
        return res

# Global profiler object (use KeepTime to interact with this object):
profiler = Profiler()

class KeepTime(object):
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        if SHOULD_PROFILE or self.name=="/":
            profiler.start(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if SHOULD_PROFILE or self.name=="/":
            profiler.lapse(self.name)
