import numpy as np
import os, inspect, warnings
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from digideep.utility.logging import logger


from digideep.agent.sampler_common import Compose, flatten_memory_to_train_key, get_memory_params, check_nan, check_shape, check_stats, print_line
from digideep.agent.sampler_common import truncate_datalists, flatten_first_two


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
    """ Wrapper for feed-forward policy in :class:`~digideep.agent.ppo.agent.Agent`.
    
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



#############################
### Composing the sampler ###
#############################
# NOTE: The order is IMPORTANT!
preprocess = Compose([flatten_memory_to_train_key, # Must be present: It flattens the memory dict to the "train" key.
                      get_memory_params,  # Must be present: It gets the memory parameters and passes them to the rest of functions through "info".
                      get_last_chunk,     # Must be present: It gets the last chunk in the memory, which is useful for PPO, A2C, etc.
                      compute_advantages, # Must be present: It computes the advantage function based on the other values.
                      truncate_datalists, # Must be present: It truncates the last entry, i.e. time t+1 which was used to produce other values
                      # check_nan,        # It complains about existing NaNs in the chunk.
                      # check_shape,      # It prints the shapes of the existing keys in the chunk.
                      # check_stats,
                      # print_line,       # This only prints a line for more beautiful debugging.
                     ])

# Sampler for non-recurrent policies
sampler_ff = Compose([preprocess,
                      wrap_ff             # Must be present at end: Wraps everything for use by feed-forward policies
                     ])

# Sampler for recurrent policies
sampler_rn = Compose([preprocess,
                      wrap_rn             # Must be present at end: Wraps everything for use by recurrent policies
                     ])
