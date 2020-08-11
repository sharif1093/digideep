import os, shutil
import time
import numpy as np
from sys import getsizeof
from digideep.utility.logging import logger

from PIL import Image

class Memory:
    def __init__(self, session, **params):
        self.session = session
        self.params = params

        # Assertions
        assert self.params["buffer_chunk_len"] > 1

        # assert self.params["chunk_sample_len"] > self.params["overrun"]
        # Buffer size in #transitions
        self.buffer_size = self.params["chunk_sample_len"] * self.params["buffer_chunk_len"]
        
        # When self.params["buffer_chunk_len"] == 1:
        ## self.buffer_size = self.params["chunk_sample_len"] * self.params["buffer_chunk_len"] + self.params["overrun"]
        
        # The data will be stored in the buffer variable.
        self.buffer = {}

        # Memory states
        self.state = {}
        self.state['start'] = 0 # Start index
        self.state['end']   = self.params["overrun"] - 1 # End index
        self.state['batch_size'] = None
        self.state['frame'] = 0

        self.counter = 0
    

    def save_snapshot(self, index):
        t = time.time()
        logger.warn("Taking memory ({}) snapshot started ...".format(self.params["name"]))
        # NOTE: Save self.buffer under memsnapshot/checkpoint-X/<memory-name>.npz
        #                     or under memsnapshot/checkpoint-X/<name>/ tree of ".npy" files
        dirname = os.path.join(self.session.state['path_memsnapshot'], "checkpoint-" + str(index))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, self.params["name"])
        logger.warn("Saving memory to:", filename+".npz")
        np.savez(filename, **self.buffer)
        logger.warn("Taking memory snapshot finished in {:.2f} seconds...".format(time.time() - t))
        # NOTE: Now that we have successfully saved current checkpoint, we can remove old checkpoints.
        if not self.params["keep_old_checkpoints"]:
            # Go and find the memory checkpoint that we started from.
            if self.session.state["checkpoint_name"] and self.session.is_loading:
                dirname = os.path.join(self.session.state["path_memsnapshot"], self.session.state["checkpoint_name"])
                if os.path.exists(dirname):
                    shutil.rmtree(dirname)

    def load_snapshot(self):
        t = time.time()
        logger.warn("Loading memory ({}) from snapshot started ...".format(self.params["name"]))
        dirname = os.path.join(self.session.state['path_memsnapshot'], self.session.state['checkpoint_name'])
        filename = os.path.join(dirname, self.params["name"] + ".npz")
        logger.warn("Loading memory from:", filename)
        loading = np.load(filename)
        for key in loading.files:
            self.buffer[key] = loading[key]
            logger.info("[Memory] Loading from disk: '{key}' (shape:{shape}), (dtype:{dtype}).".format(
                        key=key,
                        dtype=self.buffer[key].dtype,
                        shape=self.buffer[key].shape))
        
        logger.warn("Loading memory from snapshot finished in {:.2f} seconds...".format(time.time() - t))

    # def load_datasets(self):
    #     # NOTE: Very important, memory will not throw any exceptions if a key is added to it after loading!
    #     #       In other words, loading memory can fail silently!
    #     filelist = []
    #     keyslist = []
    #     for root, dirs, files in os.walk(self.memroot):
    #         for file in files:
    #             filelist += [os.path.join(root, file)]
    #             key = os.path.splitext("/"+os.path.relpath(os.path.join(root, file), self.memroot))[0]
    #             keyslist += [key]
    #
    #     for key, filename in zip(keyslist, filelist):
    #         self.buffer[key] = np.memmap(filename, mode='r+', shape=self.state['keys'][key]['shape'], dtype=self.state['keys'][key]['dtype'])
    #         logger.warn("[Memory] Loading from disk: '{key}' (shape:{shape}), (dtype:{dtype}).".format(
    #                     key=key,
    #                     dtype=self.state['keys'][key]['dtype'],
    #                     shape=self.state['keys'][key]['shape']))

    def state_dict(self):
        # TODO: In order to save self.buffer, implement the save_snapshot/load_snapshot interface.
        # return {"state":self.state, "buffer":self.buffer}
        return {"state":self.state}
    def load_state_dict(self, state_dict):
        self.state.update(state_dict["state"])
        # self.buffer = state_dict["buffer"]
        
    
    
    def get_valid_index(self, index):
        return index % self.buffer_size
    
    def get_length(self, start, end):
        if start < end:
            return end - start + 1
        elif start > end:
            not_included_length = start - end - 1
            return self.buffer_size - not_included_length
    
    @property
    def length(self):
        return self.get_length(self.state['start'], self.state['end'])
    
    @property
    def full(self):
        if self.length == self.buffer_size:
            return True
        else:
            # It is less then ...
            return False

    ## Generating useful indices
    def get_index_within_limits(self, start, end):
        if start <= end:
            index_list = np.arange(start, end+1)
        elif start > end:
            index_list = np.concatenate([np.arange(start, self.buffer_size), np.arange(0, end+1)])
        return index_list
    def get_index_move_n_steps(self, elements, steps):
        return self.get_valid_index(elements + steps)
    
    def get_index_up_to_end(self):
        return self.get_index_within_limits(self.state['start'], self.state['end'])
    
    # Remove the overrun transitions from list
    def get_index_valid_elements(self):
        end_prime = self.get_valid_index(self.state['end'] - self.params["overrun"])
        return self.get_index_within_limits(self.state['start'], end_prime)
    
    ## Chunk-index operations
    def get_index_last_chunk(self):
        end = self.state['end']
        chunk_size = self.params["chunk_sample_len"] + self.params["overrun"]
        start_prime = self.get_valid_index(end - chunk_size + 1)
        return self.get_index_within_limits(start_prime, end)
    
    ## Last chunk without the overrun transitions
    def get_index_valid_last_chunk(self):
        index_last_chunk = self.get_index_last_chunk()
        overrun = self.params["overrun"]
        return index_last_chunk[:-overrun]
    
    def get_index_new_chunk(self):
        chunk_size = self.params["chunk_sample_len"]
        last_chunk_index = self.get_index_last_chunk()
        return self.get_index_move_n_steps(last_chunk_index, chunk_size)
        
    def move_list_one_chunk_forward(self):
        chunk_size = self.params["chunk_sample_len"]
        
        end_prime = (self.state['end'] + chunk_size)
        self.state['end'] = self.get_valid_index(end_prime)
        
        if end_prime >= self.buffer_size:
            # overflow
            self.state['start'] = self.get_valid_index(self.state['end'] + 1)
        else:
            # no overflow
            if self.state['start'] > 0:
                self.state['start'] = self.get_valid_index(self.state['end'] + 1)
    
    def get_appropriate_nan(self, dtype):
        if np.issubdtype(dtype, np.floating):
            return np.nan
        elif np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).min
    
    # Store a chunk
    def store(self, chunk):
        """ Store a chunk of data in the memory.

        Args:
            chunk (dict): The chunk of information in dictionary format:
              ``{"keys": array(batch_size x num_steps x *(key_shape))}``
              This function does not assume anything about the key names in the ``chunk``.
              If the key is new, it create a new entry for that in the memory. If it already
              exists, it will be appended under the existing key.
        
        This function appends the new ``chunk`` to the ``buffer`` in a key-wise manner.
        If the memory is already full, the new data will replace the oldest data, i.e. a queue.

        Tip:
            A chunk from the :class:`~digideep.environment.explorer.Explorer` consists of
            ``batch_size`` trajectories. Each trajectory includes ``n-steps + 1`` transitions.
            The last transition from the last step is always overriden, since that is a
            "half-step" and all information in that half-step would recur in the new step, i.e.
            observations or actions. So the size of the buffer (in terms of transitions)
            is always ``k x n-steps + 1``, where ``k`` is the number of chunks stored so far.

        """



        #########################################
        ### CODE FOR DEBUGGING THE INPUT DATA ###
        #########################################
        ## Tensorboard functions:
        # add_scalar(tag, scalar_value, global_step=None, walltime=None)
        # add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
        # add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)
        # add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
        # add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
        # add_figure(tag, figure, global_step=None, close=True, walltime=None)
        # add_video(tag, vid_tensor, global_step=None, fps=4, walltime=None)
        # add_audio(tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None)
        # add_text(tag, text_string, global_step=None, walltime=None)
        # add_graph(model, input_to_model=None, verbose=False)
        # add_embedding(mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)
        # add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None)
        # add_custom_scalars(layout)
        # add_mesh(tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None)
        # add_hparams(hparam_dict=None, metric_dict=None)

        # print(chunk.keys())
        # # print("-"*40)

        # print(chunk["/observations/camera"].shape)
        # Camera: 1, 2, 4, 180, 240)
        
        # self.session.writer.add_histogram('memory:observations/agent', chunk["/observations/agent"][0,0], self.state['frame'])
        # self.session.writer.add_scalar('memory:rewards', chunk["/rewards"][0,0], self.state['frame'])
        # add_hparams
        
        
        
        # dataformats=NCHW, NHWC, CHW, HWC, HW, WH

        # shape = chunk["/observations/camera"][0,0].shape
        # cam_batch = chunk["/observations/camera"][0,0]

        ## Sequence of images as stacking
        # self.session.writer.add_images(tag=self.params["name"]+"_images", 
        #                                img_tensor=chunk["/observations/camera"][0,0].reshape(shape[0],1,shape[1],shape[2]),
        #                                global_step=self.state['frame'],
        #                                dataformats='NCHW')
        
        ## Sequence of images as channels
        # self.session.writer.add_image(tag=self.params["name"]+"_images", 
        #                               img_tensor=chunk["/observations/camera"][0,0],
        #                               global_step=self.state['frame'],
        #                               dataformats='CHW')

        ## Histograms
        # self.session.writer.add_histogram('distribution centers', x + i, i)
        
        # cam_stacked = np.concatenate(cam_batch, axis=1)
        # new_img = Image.fromarray(cam_stacked, 'L')
        # new_img.save("/master/reports/{:04d}_gray_{}.jpg".format(self.state['frame'], self.params["name"]))
        ######################################################
        
        
        self.state['frame'] += 1
        
        

        
        ## Assertions
        sizes = [chunk[key].shape[0:2] for key in chunk.keys()]
        assert np.all(np.array(sizes) == sizes[0]), "All keys should have the same size (batch, samples, *)."
        
        # size (batch, samples)
        size = sizes[0]
        batch_size = size[0]   # Here it indicates the number of workers
        chunk_size_plus_overrun = size[1]
        
        assert chunk_size_plus_overrun == self.params["chunk_sample_len"]+self.params['overrun'], "Chunk should have " + str(self.params['chunk_sample_len']+self.params['overrun']) + " samples."
        if self.state['batch_size']:
            assert batch_size == self.state['batch_size']
        else:
            self.state['batch_size'] = batch_size
        

        
        
        # self.counter += 1
        # # if self.params["mode"]=="demo":
        # pic = chunk["/observations/camera"][0,0,-1]
        # img = Image.fromarray(pic)
        # img = img.convert("L")
        # img.save("/home/sharif/frames/{}_{:04d}.jpg".format(self.params["mode"], self.counter))
        # exit()



        ## Assignments (memory)
        all_index_list = self.get_index_up_to_end()
        new_index_list = self.get_index_new_chunk()
        for key in chunk:
            if not key in self.buffer:
                ## np.empty is much faster than np.full
                dtype = chunk[key].dtype
                self.buffer[key] = np.empty(shape=(batch_size, self.buffer_size, *chunk[key].shape[2:]), dtype=dtype)
                self.buffer[key][:,all_index_list] = self.get_appropriate_nan(dtype)

                size = getsizeof(self.buffer[key]) / 1024. / 1024.
                logger.warn("Dictionary entry [{}] added (type: {:s}, size: {:9.1f} MB)".format(key,  str(dtype), size))
            
            # TODO: Check the shape of new data and complain if not consistent.
            self.buffer[key][:,new_index_list] = chunk[key]
            # # TODO: Fix it for integer types!
            # if np.issubdtype(dtype, np.floating):
            #     self.buffer[key][:,new_index_list] = chunk[key]
            # elif np.issubdtype(dtype, np.integer):
            #     self.buffer[key][:,new_index_list] = chunk[key]
            # else:
            #     logger.warn("The [{}] type in memory in neither integer nor floating, it is {}.".format(key, self.buffer[key].dtype))
        
        for key in self.buffer:
            if not key in chunk:
                self.buffer[key][:,new_index_list] = self.get_appropriate_nan(self.buffer[key].dtype)
        
        self.move_list_one_chunk_forward()    

    def get_key_shape(self, key):
        """Returns the shape of a key in the memory.
        """
        if key in self.buffer:
            return self.buffer[key].shape[2:]
        else:
            raise KeyError("The key[" + key + "] was not found in the buffer.")

    def get_buffer_keys(self):
        """Returns a ``list`` of keys stored in the memory.
        """
        return list(self.buffer.keys())
    
    ## These two functions get T and N
    def get_chunk_sample_num(self):
        """Returns the size of the chunks, measured in sample. Also known as ``n-steps``.
        """
        return self.params["chunk_sample_len"]
    def get_num_batches(self):
        """Returns the batch size of the memory data. The batch size should be consistent within all chunks stored in the memory.
        """
        return self.state['batch_size']
    def get_last_trans_index(self):
        return self.state['end']
    
    ## Buffer
    def get_buffer(self):
        """Returns the buffer variable, i.e. the whole memory.
        """
        return self.buffer

    def clear_buffer(self):
        """Clears all state variables, ``start``, ``end``, and ``batch_size``, to initial values.
        The new data will override the old data in new store commands.
        """
        self.state['start'] = 0 # Start index
        self.state['end']   = self.params["overrun"] - 1 # End index
        self.state['batch_size'] = None
