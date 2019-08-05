import numpy as np
import warnings

from .common import Compose, flatten_memory_to_train_key, get_memory_params, check_nan, check_shape, check_stats, print_line
from .common import flatten_first_two
from digideep.utility.logging import logger

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
    observation_path = info["observation_path"]

    N = info["num_records"] - 1 # We don't want to consider the last "incomplete" record, hence "-1"

    masks_arr = buffer["/masks"][:,:N]
    masks_arr = masks_arr.reshape(-1)
    total_arr = np.arange(0,num_workers*N)
    ## This is if we want to mask final states (mask equals 0 for final state, 1 otherwise.)
    # valid_arr = total_arr[masks_arr.astype(bool)]
    valid_arr = total_arr
    
    if batch_size >= len(valid_arr):
        # We don't have enough data in the memory yet.
        logger.debug("batch_size ({}) should be smaller than total number of records (~ {}={}x{}).".format(batch_size, num_workers*N, num_workers, N))
        return None

    # Sampling with replacement:
    # sample_indices = np.random.choice(valid_arr, batch_size, replace=True)
    sample_indices = np.random.choice(valid_arr, batch_size, replace=False)

    sample_tabular   = [[sample_indices // N], [sample_indices % N]]
    sample_tabular_2 = [[sample_indices // N], [sample_indices % N + 1]]

    # Extracting the indices
    batch = {}
    for key in buffer:
        batch[key] = buffer[key][sample_tabular[0],sample_tabular[1]]
    
    observation_path = "/observations" + observation_path
    batch["/obs_with_key"] = batch[observation_path]
    # Adding predictive keys
    batch["/obs_with_key_2"] = buffer[observation_path][sample_tabular_2[0],sample_tabular_2[1]]
    
    batch = flatten_first_two(batch)
    return batch


#############################
### Composing the sampler ###
#############################

# Sampler with replay buffer
sampler_re = Compose([flatten_memory_to_train_key, # Must be present: It flattens the memory dict to the "train" key.
                      get_memory_params,  # Must be present: It gets the memory parameters and passes them to the rest of functions through "info".
                      get_sample_memory,  # Sample
                      # check_shape,      # It prints the shapes of the existing keys in the chunk.
                      # check_nan,        # It complains about existing NaNs in the chunk.
                      # check_stats,
                      # print_line,       # This only prints a line for more beautiful debugging.
                     ])
