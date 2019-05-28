"""
This module provides helping functions and tools to create a sampler from the memory.
The samplers take advantage of a highly modular pattern in order to create new samplers
or change the behavior of the current ones much easier.
One can build modular samplers by cascading functions using :class:`Compose` class.
All function must have the following signature:

.. code-block:: python

  def func(data, info)

* ``data`` is a dictionary where all data is stored. It can be the whole memory at the first sampler
  block, and then narrowing down to a small sampled chunk of data at the end.
* ``info`` is a dict containing information that is passed through to the last sampler. It basically
  contains information that one sampler may need, e.g. ``batch_size``, ``memory_size``, etc.

Parts of this module is inspired by `pytorch-a2c-ppo-acktr <https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/>`_.
"""

import numpy as np
import os, inspect
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from digideep.utility.logging import logger

import warnings

class Compose:
    """A class to create a composed function from a list.
    
    Args:
        functions (list): A list of functions with the following prototype which will be called
          in cascade after the class is called. The first function in the list will be called
          first and results will be passed to the second one and so on.
    
    .. code-block:: python

      f_composed = Compose([f1, f2])
    """
    def __init__(self, functions):
        self.functions = functions
    def __call__(self, data, info):
        for f in self.functions:
            data = f(data, info)
        return data


##################################
##           SAMPLER            ##
##################################
def get_memory_params(memory, info):
    """A sampler function to get memory parameters and store them in the ``info`` for the future samplers.

    Args:
        memory: The main memory object.
        info (dict): A dictionary that can be used to transfer hyper-information among sampler functions.
    
    Returns:
        dict: A reference to the internal buffer of the memory.

    Todo:
        We may have a ``include_keys``/``exclude_keys`` argument to filter keys in the memory.
        It can help control the downstream flow of keys in the memory. Only those keys would be
        passed through that satisfy both ``include_keys`` and ``exclude_keys``. Example:

        .. code-block:: python

          include_keys:["/actions/*","/observations"]
          exclude_keys:["/info*"]
    """

    info["num_steps"]   = memory.get_chunk_sample_num()
    info["num_workers"] = memory.get_num_batches()
    info["num_records"] = memory.get_last_trans_index()

    buffer = memory.get_buffer()
    return buffer
    

####################
###### HELPER ######
####################
def flatten_first_two(batch):
    """This is a helper function that is used in other sampler functions.
    It flattens the first two dimensions of each key entry
    in the batch, thus making the data flattened.
    """
    # The data must be intact up to preprocess.
    # After that we are free.
    for key in batch:
        batch[key] = batch[key].reshape(-1, *batch[key].shape[2:])
    return batch

def truncate_datalists(chunk, info):
    """This sampler function truncates the last data entry in the chunk and returns the rest.
    For an example key in the chunk, if the input shape is ``(batch_size, T+1, ...)``, it will become
    ``(batch_size, T, ...)``. This is basically to remove the final "half-step" taken in the environment.
    """
    params = info["truncate_datalists"]
    n = params["n"]

    for key in chunk:
        chunk[key] = chunk[key][:,:-n]
    return chunk
    

#########################
###### CHECK CHUNK ######
#########################
# if torch.isnan(torch.tensor( ... )).any():
def check_nan(chunk, info):
    """This sampler function has debugging purposes and will publish a warning message if there are NaN values in the chunk.
    """
    if chunk:
        for key in chunk:
            if np.isnan(chunk[key]).any():
                logger.warn("%s:%s[%d]: Found NaN '%s'." % 
                            (os.path.basename(inspect.stack()[2].filename),
                            inspect.stack()[2].function,
                            inspect.stack()[2].lineno,
                            key))
    return chunk

def check_shape(chunk, info):
    """This sampler function has debugging purposes and reports the shapes of every key in the data chunk.
    """
    logger.warn("%s:%s[%d]: Checking shapes:" % 
                (os.path.basename(inspect.stack()[2].filename),
                 inspect.stack()[2].function,
                 inspect.stack()[2].lineno))
    if chunk:
        for key in chunk:
            logger.warn("%s %s" % ( key, str(chunk[key].shape)))
    return chunk

def check_stats(chunk, info):
    """This sampler function has debugging purposes and will report the mean and standard deviation of every key in the data chunk.
    """
    logger.warn("%s:%s[%d]: Checking stats:" % 
                (os.path.basename(inspect.stack()[2].filename),
                 inspect.stack()[2].function,
                 inspect.stack()[2].lineno))
    if chunk:
        for key in chunk:
            logger.warn("{} = {:.2f} (\xB1{:.2f} 95%)".format(key, np.nanmean(chunk[key]), 2*np.nanstd(chunk[key])))
    return chunk

def print_line(chunk, info):
    logger.warn("=========================================")
    return chunk
