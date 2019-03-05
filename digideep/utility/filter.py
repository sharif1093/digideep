from collections import deque
import numpy as np

class MovingAverage(object):
    """An implementation of moving average. It has an internal queue of the values.

    Args:
        size (int): The length of the value vector.
        window_size (int): The window size for calculation of the moving average.
    """
    def __init__(self, size=1, window_size=10):
        self.size = size
        self.window_size = window_size

        self.state = {}
        self.state["current_size"] = 0
        self.state["memory"] = np.empty((window_size, size))
        
    def append(self, item):
        # Because of 'maxlen' the first item will be dropped from the deque after reaching 'maxlen'.
        assert len(item)==self.size, "Item should have the same size as was initialized = (" + str(self.size) + ")."
        self.state["current_size"] += 1
        if self.state["current_size"] > self.window_size:
            self.state["memory"] = np.roll(self.state["memory"], shift=-1, axis=0)
            self.state["current_size"] -= 1
        self.state["memory"][self.state["current_size"]-1,:] = item
    
    @property
    def data(self):
        return self.state["memory"][0:self.state["current_size"],:]

    @property
    def mean(self):
        return np.mean(self.state["memory"][0:self.state["current_size"],:], axis=0)
    
    @property
    def std(self):
        return np.std(self.state["memory"][0:self.state["current_size"],:], axis=0)
    
    @property
    def median(self):
        return np.median(self.state["memory"][0:self.state["current_size"],:], axis=0)
    
    @property
    def min(self):
        return np.min(self.state["memory"][0:self.state["current_size"],:], axis=0)
    
    @property
    def max(self):
        return np.max(self.state["memory"][0:self.state["current_size"],:], axis=0)
