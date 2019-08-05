from collections import OrderedDict as odict
from copy import deepcopy
import numpy as np
from digideep.utility.json_encoder import JsonEncoder
import time

class WindowValue:
    def __init__(self, value, window):
        # If value is an n-d array, it will be processed element-wise.
        arr = np.array(value)
        self.data = {}
        self.data["std"] = np.zeros_like(arr)
        self.data["num"] = 1
        self.data["min"] = arr
        self.data["max"] = arr
        self.data["sum"] = arr
        
        # TODO: window should be an integer >= -1
        assert (window >= -1) and isinstance(window, int), "'window' should be an integer greater than or equal to '-1'."
        self.window = window
    def append(self, value):
        # "std" whould be updated first
        # https://math.stackexchange.com/a/2105509
        self.data["std"] = self._update_std(value)
        self.data["num"] = self.data["num"] + 1
        self.data["min"] = np.minimum(self.data["min"], value)
        self.data["max"] = np.maximum(self.data["max"], value)
        self.data["sum"] = np.add(self.data["sum"], value)
    
    def _update_std(self, value):
        # Assumption: num > 1
        sum = self.data["sum"]
        num = self.data["num"]
        var = np.power(self.data["std"], 2) * (num-1)
        var = var + np.power(np.multiply(value, num)-sum, 2) / num / (num+1)
        return np.sqrt(var / num)
    
    def get_win(self):
        return self.window
    
    def get_num(self):
        return self.data["num"]
    def get_sum(self):
        return self.data["sum"]
    def get_min(self):
        return self.data["min"]
    def get_max(self):
        return self.data["max"]
    def get_std(self):
        return self.data["std"]

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
        self.meta = {}
        self.data = odict()
        self.pack = [] # A list of dicts: {"meta":{}, "data":{}}
                
    def reset(self):
        self.pack = []
    
    def set_meta_key(self, key, value):
        self.meta[key] = value
    def get_meta_key(self, key):
        assert key in self.meta
        return self.meta[key]
    
    def __call__(self, name, value, window=-1):
        """
        Args:
            name (str): The name of the variable.
            value (list, :obj:`np.array`, float): The value of the variable in ``list``, :obj:`np.array`, or ``float`` format.
            window (int): Minimum number of occurrences for the variable to be packed.
            
        """
        try:
            if not name in self.data:
                self.data[name] = WindowValue(value, window)
            else:
                assert self.data[name].get_win() == window, "Window size cannot change in the midst of monitoring!"
                self.data[name].append(value)
            
            # We need to pack those with win>-1 instantly.
            if self.data[name].get_win() >= 0:
                self.pack_keys([name])
                
        except Exception as ex:
            raise RuntimeError("Error occured at name = " + name + ": " + str(ex))
    
    def pack_keys(self, keys): # i.e. take a snapshop of the specified keys.
        # Pack those keys that have fulfilled their window sizes.
        data = {}
        for key in keys:
            if self.data[key].get_num() >= self.data[key].get_win():
                data[key] = self.data[key].data
                del self.data[key]
        if data:
            meta = deepcopy(self.meta)
            meta["clock"] = time.time()
            pack = dict(meta=meta, data=data)
            self.pack += [pack]
    def pack_data(self):
        # Pack all keys available in the self.data
        self.pack_keys(list(self.data.keys()))
    #########
    def __repr__(self):
        self.pack_data()
        res = ""
        for pack in self.pack:
            meta = pack["meta"]
            data = pack["data"]
            
            for key in sorted(list(data.keys())):
                val = data[key]
                
                res += ">> {name:s} [{num:d}x] = {mean:s} (\xB1{std:s} %95) in range{{{r_min:s} <= {r_max:s}}}\n".format(
                           name=key,
                           num=val["num"],
                           mean=np.array2string(val["sum"]/val["num"], precision=3),
                           std=np.array2string(2*val["std"], precision=3),
                           r_min=np.array2string(val["min"], precision=3),
                           r_max=np.array2string(val["max"], precision=3)
                        )
        return res
    
    def set_output_file(self, path):
        self.filename = path
    def dump(self):
        self.pack_data()
        if self.filename:
            f = open(self.filename, 'a')
            for pack in self.pack:
                jsonstring = JsonEncoder(pack)
                print(jsonstring, flush=True, file=f)
            f.close()

# Global monitor object:
monitor = Monitor()
