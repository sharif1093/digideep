from collections import OrderedDict as odict
import time
import numpy as np
from .json_encoder import JsonEncoder

SHOULD_MONITOR = True

class Monitor(object):
    """
    A very simple and lightweight implementation for a global monitoring tool.
    This class keeps track of a variable's mean, standard deviation, minimum, maximum, and sum in a recursive manner.
    
        >>> monitor.reset()
        >>> for i in range(1000):
        ...   monitor('loop index', i)
        ...
        >>> print(monitor)
        >> loop index [1000x] = 499.5 (±577.639 %95) in range{0 < 999}
    
    Todo:
        Provide batched monitoring of variables.
    
    Note:
        This class does not implement moving average. For a moving average implementation refer to :class:`~digideep.utility.filter.MovingAverage`.
    """

    def __init__(self):
        self.filename = None
        self.reset()

    def reset(self):
        # This is the count of data
        self.data = odict()
        # Each key of the dicts is a numpy array.
        # num: This is the count of data
        # std, min, max, sum
    
    def __call__(self, *args, **kwargs):
        """
        Args:
            name (str): The name of the variable.
            value (list, :obj:`np.array`, float): The value of the variable in ``list``, :obj:`np.array`, or ``float`` format.
        """
        if SHOULD_MONITOR:
            self.append(*args, **kwargs)

    def append(self, name, value):
        try:
            if not name in self.data:
                arr = np.array(value)
                self.data[name] = {}
                self.data[name]["std"] = np.zeros_like(arr)
                self.data[name]["num"] = 1
                self.data[name]["min"] = arr
                self.data[name]["max"] = arr
                self.data[name]["sum"] = arr
                
            else:
                # "std" whould be updated first
                # https://math.stackexchange.com/a/2105509
                self.data[name]["std"] = self._update_std(name, value)
                self.data[name]["num"] = self.data[name]["num"] + 1
                self.data[name]["min"] = np.minimum(self.data[name]["min"], value)
                self.data[name]["max"] = np.maximum(self.data[name]["max"], value)
                self.data[name]["sum"] = np.add(self.data[name]["sum"], value)
                
                # self.data[name] += [value]
        except Exception as ex:
            raise RuntimeError("Error occured at name = " + name + ": " + str(ex))
            
    def _update_std(self, name, value):
        # Assumption: num > 1
        sum = self.data[name]["sum"]
        num = self.data[name]["num"]
        var = np.power(self.data[name]["std"], 2) * (num-1)
        var = var + np.power(np.multiply(value, num)-sum,2) / num / (num+1)
        return np.sqrt(var / num)
    
    def get_num(self, name):
        if name in self.data:
            return self.data[name]["num"]
        else:
            return 0
    
    def get_sum(self, name):
        assert name in self.data
        return self.data[name]["sum"]
    
    def get_min(self, name):
        assert name in self.data
        return self.data[name]["min"]
    
    def get_max(self, name):
        assert name in self.data
        return self.data[name]["max"]
        
    def get_std(self, name):
        assert name in self.data
        return self.data[name]["std"]

    def get_avg(self, name):
        assert name in self.data
        return self.data[name]["sum"] / self.data[name]["num"]
        
    #########    
    def get_keys(self):
        return list(self.data.keys())
    
    def __repr__(self):
        # TODO: We can show this as a tree to be more readable.
        # Example:
        # /
        #   explore
        #     step
        #     render
        #   update
        #     ...
        res = ""
        for k in self.get_keys():
            res += ">> {:s} [{:d}x] = {:s} (\xB1{:s} %95) in range{{{:s} < {:s}}}\n".format(
                       k,
                       self.get_num(k),
                       np.array2string(self.get_avg(k), precision=3),
                       np.array2string(2*self.get_std(k), precision=3),
                       np.array2string(self.get_min(k), precision=3),
                       np.array2string(self.get_max(k), precision=3)
                       )
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

# Global monitor object:
monitor = Monitor()
