from collections import OrderedDict as odict
import time
import numpy as np

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
        self.reset()

    def reset(self):
        # This is the count of data
        self.num = odict()
        # Each key of the dicts is a numpy array.
        self.std = odict()
        self.min = odict()
        self.max = odict()
        self.sum = odict()
        
        # self.data = odict()
    
    def __call__(self, *args, **kwargs):
        """
        Args:
            name (str): The name of the variable.
            value (list, :obj:`np.array`, float): The value of the variable in ``list``, :obj:`np.array`, or ``float`` format.
        """
        if SHOULD_MONITOR:
            self.append(*args, **kwargs)

    def append(self, name, value):
        if not name in self.num:
            arr = np.array(value)
            
            self.std[name] = np.zeros_like(arr)
            self.num[name] = 1
            self.min[name] = arr
            self.max[name] = arr
            self.sum[name] = arr
            
            # self.data[name] = [value]
        else:
            # "std" whould be updated first
            # https://math.stackexchange.com/a/2105509
            self.std[name] = self._update_std(name, value)
            self.num[name] = self.num[name] + 1
            self.min[name] = np.minimum(self.min[name], value)
            self.max[name] = np.maximum(self.max[name], value)
            self.sum[name] = np.add(self.sum[name], value)
            
            # self.data[name] += [value]
            
    def _update_std(self, name, value):
        # Assumption: num > 1
        sum = self.sum[name]
        num = self.num[name]
        var = np.power(self.std[name], 2) * (num-1)
        var = var + np.power(np.multiply(value, num)-sum,2) / num / (num+1)
        return np.sqrt(var / num)
    
    def get_num(self, name):
        assert name in self.num
        return self.num[name]
    
    def get_sum(self, name):
        assert name in self.num
        return self.sum[name]
    
    def get_min(self, name):
        assert name in self.num
        return self.min[name]
    
    def get_max(self, name):
        assert name in self.num
        return self.max[name]
        
    def get_std(self, name):
        assert name in self.num
        return self.std[name]

    def get_avg(self, name):
        assert name in self.num
        return self.sum[name] / self.num[name]
        
    #########    
    def get_keys(self):
        return list(self.num.keys())
    
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

# Global monitor object:
monitor = Monitor()
