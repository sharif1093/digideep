import threading
import numpy as np
from sys import getsizeof
from digideep.utility.logging import logger


class Memory:
    """Implements a generic memory to store trajectories from the :class:`~digideep.environment.explorer.Explorer`.

    Note:
        We call the output of :class:`~digideep.environment.explorer.Explorer` class a *chunk*.
        A *chunk* is a bath of *trajectories*. Each *trajectory* is a sequence of *transitions*.
        Each *transition* is a single step in one environment by the explorer.

    Args:
        session: The reference to the current :class:`~digideep.pipeline.session.Session`.
        chunk_sample_len (int): the size of the chunks, measured in sample
        buffer_chunk_len (int): the size of the buffer, measured in chunks
    
    Attributes:
        buffer (dict): Where all of transitions are stored. The overall size of the buffer will be
            ``chunk_sample_len x buffer_chunk_len``
        state (dict): Includes the state variables of the memory.

            * ``n_batch`` is the batch size of data in the memory. Here it is related to the ``num_workers``
              parameter in the :class:`~digideep.environment.explorer.Explorer`.
            * ``i_index`` is the index of the last transition stored in the memory.
            * ``i_chunk`` is the index of last chunk stored in the memory.
    
    Warning:
        This class is not completely thread-safe yet. Use with caution if using with multiple threads.

    """
    def __init__(self, session, **params):
        self.session = session
        self.params = params

        # Buffer size in #transitions
        self.buffer_size = self.params["chunk_sample_len"] * self.params["buffer_chunk_len"] + 1
        # This +1 is very important indeed and has special meaning.
        
        # The data will be stored in the buffer variable.
        self.buffer = {}

        # Memory states
        self.state = {}
        self.state['n_batch'] = None
        self.state['i_index'] = 1 # Shows the index of last time-index in the buffer. Should be one as a result of presteping.
        self.state['i_chunk'] = 0 # Shows the index of last chunk in the buffer.
        

        # Be careful not to call another function which needs the lock within a lock,
        # or it will hang forever.
        self.lock = threading.Lock()
    
    def state_dict(self):
        """The function to store the state of this class. ``buffer`` and ``state`` must be returned.
        
        Todo:
            Implement for resuming training.
        """
        return None
    def load_state_dict(self, state_dict):
        """The function to laod the state of the class from a dictionary.

        Todo:
            Implement for resuming training.
        """
        pass

    def store(self, chunk):
        """ Store a chunk of data in the memory.

        Args:
            chunk (dict): The chunk of information in dictionary format: ``{"keys": array(batch_size x num_steps x *(key_shape))}``
              This function does not assume anything about the key names in the ``chunk``. If the key is new, it create a new entry
              for that in the memory. If it already exists, it will be appended under the existing key.
        
        This function appends the new ``chunk`` to the ``buffer`` in a key-wise manner. If the memory is already full,
        the new data will replace the oldest data, i.e. a queue.

        Tip:
            A chunk from the :class:`~digideep.environment.explorer.Explorer` consists of ``batch_size`` trajectories. Each
            trajectory includes ``n-steps + 1`` transitions.
            The last transition from the last step is always overriden, since that is a "half-step" and all information in that
            half-step would recur in the new step, i.e. observations or actions. So the size of the buffer (in terms of transitions)
            is always ``k x n-steps + 1``, where ``k`` is the number of chunks stored so far.

        """
        sizes = [chunk[key].shape[0:2] for key in chunk.keys()]
        assert np.all(np.array(sizes) == sizes[0]), "All keys should have the same size (batch, samples, *)."
        assert sizes[0][1] == self.params["chunk_sample_len"]+1, "Chunk should have " + str(self.params["chunk_sample_len"]+1) + " samples."
        # size (batch, samples)
        
        size = sizes[0]
        batch_size = size[0]   # Here it indicates the number of workers
        trans_size = size[1]-1 # Note this -1 (related to overriding the last half-step)

        if self.full:
            # Roll memory if it's full. Do it for all existing keys.
            # Missing keys will have no problem.
            for key in self.buffer:
                self.buffer[key] = np.roll(self.buffer[key], -trans_size, axis=1)
            self.state["i_index"] -= trans_size
            self.state["i_chunk"] -= 1

        with self.lock:
            for key in chunk:
                if not key in self.buffer:
                    ## np.empty is much faster than np.full
                    # Check if the batch_size is the same with old entries.
                    if self.state["n_batch"]:
                        assert self.state["n_batch"] == batch_size, "Number of batches in "+key+" is not consistent with the buffer ("+self.state["n_batch"]+")"
                    else:
                        self.state["n_batch"] = batch_size
                    # self.buffer[key] = np.empty(shape=(batch_size, self.buffer_size, *chunk[key].shape[2:]), dtype=np.float32)

                    data_type = chunk[key].dtype
                    
                    self.buffer[key] = np.empty(shape=(batch_size, self.buffer_size, *chunk[key].shape[2:]), dtype=chunk[key].dtype)
                    if np.issubdtype(data_type, np.floating):
                        self.buffer[key][:,0:self.state["i_index"]] = np.nan
                    elif np.issubdtype(data_type, np.integer):
                        self.buffer[key][:,0:self.state["i_index"]] = np.iinfo(data_type).min

                    size = getsizeof(self.buffer[key]) / 1024. / 1024.
                    logger.warn("Dictionary entry [{}] added (type: {:s}, size: {:9.1f} MB)".format(key,  str(chunk[key].dtype), size))
                    
                    
                # Update memory
                self.buffer[key][:,self.state["i_index"]-1:self.state["i_index"]+trans_size] = chunk[key]

            for key in self.buffer:
                if not key in chunk:
                    self.buffer[key][:,self.state["i_index"]-1:self.state["i_index"]+trans_size] = np.nan

        self.state["i_index"] += trans_size
        self.state["i_chunk"] += 1
        logger.debug("Memory i_chunk: {} | i_index: {}".format(self.state["i_chunk"], self.state["i_index"]))

    @property
    def full(self):
        """Indicates whether the memory is full or not.
        
        Returns:
            bool: ``True`` if memory is full.
        """
        return self.state["i_chunk"] == self.params["buffer_chunk_len"]

    def get_key_shape(self, key):
        """Returns the shape of a key in the memory.
        """
        if key in self.buffer:
            return self.buffer[key].shape[2:]
        else:
            raise KeyError("The key[" + key + "] was not found in the buffer.")


    def get_last_trans_index(self):
        """Returns the index of the last stored entry in the memory.
        """
        with self.lock:
            return self.state["i_index"]
    def get_last_chunk_index(self):
        """Returns the index of last chunk stored in the memory.
        """
        with self.lock:
            return self.state["i_chunk"]
    def get_buffer_keys(self):
        """Returns a ``list`` of keys stored in the memory.
        """
        with self.lock:
            return list(self.buffer.keys())
    
    ## These two functions get T and N
    def get_chunk_sample_num(self):
        """Returns the size of the chunks, measured in sample. Also known as ``n-steps``.
        """
        with self.lock:
            return self.params["chunk_sample_len"]
    def get_num_batches(self):
        """Returns the batch size of the memory data. The batch size should be consistent within all chunks stored in the memory.
        """
        with self.lock:
            return self.state['n_batch']
    
    ## Buffer
    def get_buffer(self):
        """Returns the buffer variable, i.e. the whole memory.
        """
        return self.buffer

    def clear_buffer(self):
        """Clears all state variables, ``i_index``, ``i_chunk``, and ``n_batch``, to zero. The new data will override the old data in new store commands.
        """
        with self.lock:
            self.state["i_index"] = self.state["i_chunk"] = 0
            self.state['n_batch'] = 0
