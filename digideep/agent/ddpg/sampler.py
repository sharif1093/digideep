import numpy as np
import warnings

from digideep.agent.sampler_common import Compose, flatten_memory_to_train_key, get_memory_params, check_nan, check_shape, check_stats, print_line
from digideep.agent.sampler_common import flatten_first_two

from digideep.utility.logging import logger
from digideep.utility.profiling import KeepTime

def get_sample_memory(memory, info):
    """Sampler function for DDPG-like algorithms where we want to sample data from an experience replay buffer.

    This function adds the following key to the buffer:
    
    * ``/observations_2``

    Returns:
        dict: One sampled batch to be used in the DDPG algorithm for one step of training. The shape of each
        key in the output batch will be: ``(batch_size, *key_shape[2:])``

    """
    # Get information from info
    batch_size = info["batch_size"]
    observation_path = info["observation_path"]
    # Whether to use CER or not:
    use_cer = info.get("use_cer", False)
    
    # Get the main data from the memory
    buffer = memory.get_buffer()

    # Get some constants from the memory
    num_workers = memory.get_num_batches()
    N = memory.get_last_trans_index() - 1 # We don't want to consider the last "incomplete" record, hence "-1"

    record_arr = memory.get_index_valid_elements()
    worker_arr = np.arange(0, num_workers)

    num_records = len(record_arr) * num_workers

    # with KeepTime("mask_array"):
    #     masks_arr = buffer["/masks"][:,record_arr]
    #     masks_arr = masks_arr.reshape(-1)
    
    if batch_size >= num_records:
        # We don't have enough data in the memory yet.
        logger.debug("batch_size ({}) should be smaller than total number of records (~ {}={}x{}).".format(batch_size, num_records, num_workers, len(record_arr)))
        return None

    with KeepTime("sampling_by_choice"):
        if use_cer:
            last_chunk_indices = memory.get_index_valid_last_chunk()
            available_batch_size = len(last_chunk_indices) * num_workers
            if available_batch_size <= batch_size:
                # We have selected a few transitions from previous step.
                # Now, we should sample the rest from the replay buffer.
                sample_record_recent = np.repeat(last_chunk_indices, num_workers)   # 10 10 10 10 11 11 11 11 ...
                sample_worker_recent = np.tile(worker_arr, len(last_chunk_indices)) #  0  1  2  3  0  1  2  3 ...

                batch_size_prime = batch_size - available_batch_size

                # Select the rest ...
                sample_record_prime = np.random.choice(record_arr, batch_size_prime, replace=True)
                sample_worker_prime = np.random.choice(worker_arr, batch_size_prime, replace=True)

                # Combine
                sample_record = np.concatenate([sample_record_recent, sample_record_prime])
                sample_worker = np.concatenate([sample_worker_recent, sample_worker_prime])

            else:
                # OK, we have enough data, so no sampling!
                logger.warn("CER: Latest transitions greater than batch size. Sample from last transitions.")
                
                sample_record = np.random.choice(last_chunk_indices, batch_size, replace=True)
                sample_worker = np.random.choice(worker_arr,         batch_size, replace=True)

        else:    
            # NOTE: NEVER ever use sampling WITHOUT replacement: Its time scales up with th array size.
            # Sampling with replacement:
            sample_record = np.random.choice(record_arr, batch_size, replace=True)
            sample_worker = np.random.choice(worker_arr, batch_size, replace=True)

        # Move the next step samples
        sample_record_2 = memory.get_index_move_n_steps(sample_record, 1)
        # Make a table of indices to extract transitions
        sample_tabular   = [[sample_worker], [sample_record]]
        sample_tabular_2 = [[sample_worker], [sample_record_2]]

    with KeepTime("tabular_index_extraction"):
        # Extracting the indices
        batch = {}
        for key in buffer:
            batch[key] = buffer[key][sample_tabular[0],sample_tabular[1]]
    with KeepTime("post_key_generation"):
        observation_path = "/observations" + observation_path
        # Adding predictive keys
        batch[observation_path+"_2"] = buffer[observation_path][sample_tabular_2[0],sample_tabular_2[1]]
    
    with KeepTime("flatten_first_two"):
        batch = flatten_first_two(batch)
    return batch


#############################
### Composing the sampler ###
#############################

# Sampler with replay buffer
sampler_re = Compose([flatten_memory_to_train_key, # Must be present: It flattens the memory dict to the "train" key.
                      # get_memory_params,  # Must be present: It gets the memory parameters and passes them to the rest of functions through "info".
                      get_sample_memory,  # Sample
                      # check_shape,      # It prints the shapes of the existing keys in the chunk.
                      # check_nan,        # It complains about existing NaNs in the chunk.
                      # check_stats,
                      # print_line,       # This only prints a line for more beautiful debugging.
                     ])
