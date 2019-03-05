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
    

############################
##          DDPG          ##
############################
def get_sample_memory(buffer, info):
    """Sampler function for DDPG-like algorithms where we want to sample data from an experience replay buffer.

    This function does not sample from final steps where mask equals ``0`` (as they don't have subsequent observations.)
    This function adds the following key to the memory:
    
    * ``/observations_2``

    Returns:
        dict: One sampled batch to be used in the DDPG algorithm for one step of training. The shape of each
        key in the output batch will be: ``(batch_size, *key_shape[2:])``

    """
    batch_size = info["batch_size"]
    
    num_workers = info["num_workers"]
    N = info["num_records"] - 1 # We don't want to consider the last "incomplete" record, hence "-1"

    masks_arr = buffer["/masks"][:,:N]
    masks_arr = masks_arr.reshape(-1)
    total_arr = np.arange(0,num_workers*N)
    valid_arr = total_arr[masks_arr.astype(bool)]
    
    if batch_size >= len(valid_arr):
        # We don't have enough data in the memory yet.
        warnings.warn("batch_size ({}) should be smaller than total number of records (~ {}={}x{}).".format(batch_size, num_workers*N, num_workers, N))
        return None

    sample_indices = np.random.choice(valid_arr, batch_size, replace=False)

    sample_tabular   = [[sample_indices // N], [sample_indices % N]]
    sample_tabular_2 = [[sample_indices // N], [sample_indices % N + 1]]

    # Extracting the indices
    batch = {}
    for key in buffer:
        batch[key] = buffer[key][sample_tabular[0],sample_tabular[1]]
    # Adding predictive keys
    batch["/observations_2"] = buffer["/observations"][sample_tabular_2[0],sample_tabular_2[1]]
    
    batch = flatten_first_two(batch)
    return batch


############################
##           PPO          ##
############################
def get_last_chunk(buffer, info):
    """A sampler function to extract only the last chunk of the memory.
    """
    i_trans = info["num_records"]

    # Get the last chunk, reshaped (mix the first two dimensions)
    chunk = {}
    for key in buffer:
        chunk[key] = buffer[key][:,i_trans-info["num_steps"]-1:i_trans]
    return chunk

def wrap_ff(chunk, info):
    """ Wrapper for feed-forward policy in :class:`~digideep.agent.ppo.PPO`.
    
    This function shuffles data and reshapes them. This sampler does not preserve the sequence of data,
    so it is suitable for feed-forward algorithms (other than recurrent ones).

    Returns:
        generator: An iterator which returns output batches upon iteration.
    """
    info["batch_size"]  = info["num_workers"] * info["num_steps"]
    # assert batch_size >= num_mini_batches, "Please increase batch_size"
    info["mini_batch_size"] = info["batch_size"] // info["num_mini_batches"]

    chunk = flatten_first_two(chunk)
    batch = {}
    sampler = BatchSampler(sampler=SubsetRandomSampler(range(info["batch_size"])), batch_size=info["mini_batch_size"], drop_last=False)
    # OR
    # batch_indices = np.arange(info["batch_size"])
    # np.random.shuffle(batch_indices)
    # sampler = np.array_split(batch_indices, num_mini_batches)
    for indices in sampler:
        for key in chunk:
            batch[key] = chunk[key][indices]
        yield batch

############################
def wrap_rn(chunk, info):
    """A sampler that preserves the sequence of data and hence is suitable for recurrent policies.
    """
    perm = np.random.permutation(info["num_workers"])
    indices = np.array_split(perm, info["num_mini_batches"])

    assert info["num_workers"] >= info["num_mini_batches"], (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(info["num_workers"], info["num_mini_batches"]))

    hidden_state_adr = "/agents/"+info["agent_name"]+"/hidden_state"

    for batch_indices in indices:
        batch = {}
        for key in chunk:
            # Consider "/hidden_state" separately.
            if key == hidden_state_adr:
                # NOTE: Why are we storing only the first element in the hidden states?
                #       Note that we only need this key for the case when policy is recurrent.
                #       In that case, we only need the h0, and then the rest will be generated
                #       in the evaluate_actions function of the policy.
                batch[key] = chunk[key][batch_indices, 0:1] # This only gets the very first hidden_state.
            else:
                batch[key] = chunk[key][batch_indices, :]
        
        for key in batch:
            batch[key] = batch[key].swapaxes(0, 1)

        # Flatten here
        batch = flatten_first_two(batch)
        yield batch

#################################
##        PREPROCESSORS        ##
#################################
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
    

def compute_advantages(chunk, info):
    """This sampler function computes the advantages for the data chunk (see `link <https://arxiv.org/abs/1506.02438>`_).
    
    It will add the following two keys to the memory:

    * ``/agents/<agent_name>/artifacts/advantages``
    * ``/agents/<agent_name>/artifacts/returns``

    """
    params = info["compute_advantages"]
    gamma = params["gamma"]
    use_gae = params["use_gae"]
    tau = params["tau"]

    #######################################################
    # This piece of code is duplicated from "Memory.store":
    sizes = [chunk[key].shape[0:2] for key in chunk.keys()]
    assert np.all(np.array(sizes) == sizes[0]), "All keys should have the same size (batch, samples, *)."
    size = sizes[0]

    num_workers = info["num_workers"]
    assert num_workers == size[0]
    
    num_steps = info["num_steps"]
    assert num_steps == size[1]-1
    

    # print("CHUNK", chunk)
    # exit()

    #######################################################
    rewards = chunk["/rewards"]
    masks   = chunk["/masks"]
    values  = chunk["/agents/"+info["agent_name"]+"/artifacts/values"]

    advantages_adr = "/agents/"+info["agent_name"]+"/artifacts/advantages"
    returns_adr    = "/agents/"+info["agent_name"]+"/artifacts/returns"

    returns = np.empty(shape=(num_workers, num_steps+1, 1), dtype=np.float32)
    if use_gae:
        logger.debug("Using GAE ...")
        gae = 0
        for t in reversed(range(num_steps)):
            delta = rewards[:,t] + gamma * values[:,t+1] * masks[:,t+1] - values[:,t]
            gae = delta + gamma * tau * gae * masks[:,t+1]
            returns[:,t] = gae + values[:,t]
    else:
        logger.debug("Not using GAE ...")
        returns[:,-1] = values[:,-1]
        for t in reversed(range(num_steps)):
            returns[:,t] = returns[:,t+1] * gamma * masks[:,t+1] + rewards[:,t]
    
    chunk[returns_adr] = returns
    ## The final index will not produce a real advantage function. So we prefer to
    ## not consider that in the mean and std of normalization.
    advantages = chunk[returns_adr] - values
    advantages = (advantages - advantages[:,:-1].mean()) / (advantages[:,:-1].std() + 1e-5)
    chunk[advantages_adr] = advantages
    return chunk








#########################
###### CHECK CHUNK ######
#########################
# if torch.isnan(torch.tensor( ... )).any():
def check_nan(chunk, info):
    """This sampler function has debugging purposes and will publish a warning message if there are NaN values in the chunk.
    """
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

    for key in chunk:
        logger.warn("{} = {:.2f} (\xB1{:.2f} 95%)".format(key, np.nanmean(chunk[key]), 2*np.nanstd(chunk[key])))
    return chunk

def print_line(chunk, info):
    logger.warn("=========================================")
    return chunk

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


##################################################################################
####                               METHOD: PPO                                ####
##################################################################################
# NOTE: The order is IMPORTANT!
preprocess = Compose([get_memory_params,  # Must be present: It gets the memory parameters and passes them to the rest of functions through "info".
                      get_last_chunk,     # Must be present: It gets the last chunk in the memory, which is useful for PPO, A2C, etc.
                      compute_advantages, # Must be present: It computes the advantage function based on the other values.
                      truncate_datalists, # Must be present: It truncates the last entry, i.e. time t+1 which was used to produce other values
                      # check_nan,        # It complains about existing NaNs in the chunk.
                      # check_shape,      # It prints the shapes of the existing keys in the chunk.
                      # check_stats,
                      # print_line,       # This only prints a line for more beautiful debugging.
                     ])

sampler_ff = Compose([preprocess,
                      wrap_ff             # Must be present at end: Wraps everything for use by feed-forward policies
                     ])

sampler_rn = Compose([preprocess,
                      wrap_rn             # Must be present at end: Wraps everything for use by recurrent policies
                     ])
##################################################################################
####                              METHOD: DDPG                                ####
##################################################################################
# Sampler with replay buffer
sampler_re = Compose([get_memory_params,  # Must be present: It gets the memory parameters and passes them to the rest of functions through "info".
                      get_sample_memory,  # Sample
                      # check_nan,        # It complains about existing NaNs in the chunk.
                      # check_shape,      # It prints the shapes of the existing keys in the chunk.
                      # check_stats,
                      # print_line,       # This only prints a line for more beautiful debugging.
                     ])



