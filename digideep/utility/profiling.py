from collections import OrderedDict as odict
import time
import numpy as np
from .json_encoder import JsonEncoder

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
        self.filename = None
        self.reset()

    def reset(self):
        self.data = odict()
        
    def start(self, name):
        if name in self.data:
            assert self.data[name]["starts"] == None, "The previous start should be lapsed first for [{:s}]".format(name)
        else:
            self.data[name] = {"starts":None, "occurs":None, "totals":None}
        self.data[name]["starts"] = time.time()

    def lapse(self, name):
        assert name in self.data and self.data[name]["starts"] is not None, "You should first start for [{:s}]".format(name)
        elapsed = time.time() - self.data[name]["starts"]

        if self.data[name]["totals"] is None:
            self.data[name]["totals"] = elapsed
            self.data[name]["occurs"] = 1
        else:
            self.data[name]["totals"] += elapsed
            self.data[name]["occurs"] += 1

        self.data[name]["starts"] = None

    def get_time_average(self, name):
        assert name in self.data
        return self.data[name]["totals"]/self.data[name]["occurs"]

    def get_time_overall(self, name):
        assert name in self.data
        return self.data[name]["totals"]

    def get_occurence(self, name):
        assert name in self.data
        return self.data[name]["occurs"]

    def get_keys(self):
        return list(self.data.keys())
    
    def __repr__(self):
        res = ""
        for k in self.get_keys():
            res += (">> {:s} [{:d}x, {:.1f}s]\n".format(k, self.get_occurence(k), self.get_time_overall(k)))
        return res
    
    def set_output_file(self, path):
        self.filename = path
    def dump(self, meta = {}):
        if self.filename:
            f = open(self.filename, 'a')
            out = {"meta":meta,"data":self.data}
            jsonstring = JsonEncoder(out)
            print(jsonstring, flush=True, file=f)
            f.close()

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
