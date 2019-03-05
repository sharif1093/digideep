import os, datetime, argparse
from shutil import copytree, copyfile, ignore_patterns

from digideep.utility.logging import logger
from digideep.utility.toolbox import dump_dict_as_json, dump_dict_as_yaml, get_module
from digideep.utility.json_encoder import JsonDecoder

import pickle, torch
from copy import deepcopy

def generateTimestamp():
    now = datetime.datetime.now()
    timestamp = '{:%Y%m%d%H%M%S}'.format(now)
    return timestamp

class Session(object):
    """
    This class provides the utilities for storing results of a session.
    It provides a unique path based on a timestamp and creates all sub-
    folders that are required there. A session directory will have the
    following contents:

    * :file:`session_YYYYMMDDHHMMSS/`:
    
        * :file:`checkpoints/`: The directory of all stored checkpoints.
        * :file:`digideep/`: A copy of digideep module.
        * :file:`monitor/`: Summary results of each worker environment.
        * :file:`cpanel.json`: A json file including control panel (``cpanel``) parameters in ``params`` file.
        * :file:`params.yaml`: The parameter tree of the session, i.e. the params variable in ``params`` file.
        * :file:`report.log`: A log file for Logger class.
        * :file:`visdom.log`: A log file for visdom logs.
        * :file:`__init__.py`: Python ``__init__`` file to convert the session to a module.
    
    .. comment out this part
    .. * :file:`loader.py`: A helping module for loading saved checkpoints more intuitively.
        
        

    Arguments:
        root_path (str): The path to the ``digideep`` module.

    Note:
        This class also initializes helping tools (e.g. Visdom, Logger, Monitor,
        etc.) and has helper functions for saving/loading checkpoints.
    
    Tip:
        The default directory for storing sessions is :file:`/tmp/digideep_sessions`.
        To change the default directory use the program with cli argument ``--session-path <path>``
    
    Todo:
      Complete the session-as-a-module (SaaM) implementation. Then, :file:`session_YYYYMMDDHHMMSS`
      should work like an importable module for testing and inference.
    
    Todo:
      If restoring a session, ``visdom.log`` should be copied from there and replayed.
    
    
    """
    def __init__(self, root_path):
        self.parse_arguments()
        # TODO: If loading from a checkpoint, we must copy the visdom log
        #       from the previous path to the current one.
        # TODO: If loading from a checkpoint, we must check the existence
        #       of that path and whether that's a valid digideep session.
        self.is_loading = True if self.args["load_checkpoint"] else False
        self.is_playing = True if self.args["play"] else False
        if self.is_playing:
            assert self.is_loading, "For playing the checkpoint path should be specified using `--load-checkpoint`."

        if self.args["monitor_cpu"] or self.args["monitor_gpu"]:
            # Force visdom ON if "--monitor-cpu" or "--monitor-gpu" are provided.
            self.args["visdom"] = True

            

        self.state = {}
        # Root: Indicates where we are right now
        self.state['path_root'] = os.path.split(root_path)[0]
        
        # Session: Indicates where we want our codes to be stored
        if not self.is_playing:
            self.state['path_root_session']  = self.args["session_path"]
            self.state['path_base_sessions'] = os.path.join(self.state['path_root_session'], 'digideep_sessions')
        else:
            # If we are playing a recorded checkpoint, we must save the results into the `evaluations` path
            # of that session.
            checkpoint_path = os.path.split(self.args["load_checkpoint"])[0]
            self.state['path_base_sessions'] = os.path.join(os.path.split(checkpoint_path)[0], "evaluations")

        self.state['path_session']     = os.path.join(self.state['path_base_sessions'], 'session_' + generateTimestamp())
        self.state['path_checkpoints'] = os.path.join(self.state['path_session'], 'checkpoints')
        self.state['path_monitor']     = os.path.join(self.state['path_session'], 'monitor')
        # Hyper-parameters basically is a snapshot of intial parameter engine's state.
        self.state['file_cpanel'] = os.path.join(self.state['path_session'], 'cpanel.json')
        self.state['file_params'] = os.path.join(self.state['path_session'], 'params.yaml')
        self.state['file_report'] = os.path.join(self.state['path_session'], 'report.log')
        self.state['file_visdom'] = os.path.join(self.state['path_session'], 'visdom.log')
        
        
        if not os.path.exists(self.state['path_base_sessions']):
            os.makedirs(self.state['path_base_sessions'])
            # Create an empty __init__.py in it!
            with open(os.path.join(self.state['path_base_sessions'], '__init__.py'), 'w') as f:
                print("", file=f)

        if not os.path.exists(self.state['path_session']):
            os.makedirs(self.state['path_session'])
            if not self.is_playing:
                os.makedirs(self.state['path_checkpoints'])
            os.makedirs(self.state['path_monitor'])
        else:
            raise BaseException("Session path already exists, cannot proceed.")
        
        
        self.set_device()
        self.initLogger()
        if self.args["visdom"]:
            self.initVisdom()
        if not self.is_playing:
            self.createSaaM()
        #################
        self.runMonitor() # Monitor CPU/GPU/RAM

        # Check valid params file:
        if not self.is_loading:
            try:
                get_module(self.args["params"])
            except:
                logger.fatal("Neither a checkpoint nor a valid params file are specified!")
                exit()

        print(':: The session will be stored in ' + self.state['path_session'])

    def initLogger(self):
        """
        This function sets the logger level and file.
        """
        logger.set_logfile(self.state['file_report'])
        logger.set_log_level(self.args["log_level"])
    
    def initVisdom(self):
        """
        This function initializes the connection to the Visdom server. The Visdom server must be running.

        .. code-block:: bash
            :caption: Running visdom server

            visdom -port 8097 &
        """
        from digideep.utility.visdom_engine.Instance  import VisdomInstance
        VisdomInstance(port=self.args["visdom_port"], log_to_filename=self.state["file_visdom"], replay=True)

    def createSaaM(self):
        """ SaaM = Session-as-a-Module
        This function will make the session act like a python module.
        The user can then simply import the module for inference.
        """
        # Copy the module
        module_source = os.path.join(self.state['path_root'], 'digideep')
        module_target = os.path.join(self.state['path_session'], 'digideep')
        copytree(module_source, module_target, ignore=ignore_patterns('*.pyc', '__pycache__'))
        # Copy loader.py to the session root path
        # copyfile(os.path.join(module_source, 'loader.py'), os.path.join(self.state['path_session'], 'loader.py'))
        # Create __init__.py at the session root path
        with open(os.path.join(self.state['path_session'], '__init__.py'), 'w') as f:
            print("", file=f)
            # print("from .loader import loader", file=f)
            # print("from .loader import ModelCarousel", file=f)

    def runMonitor(self):
        """
        This function will load the monitoring tool for CPU and GPU utilization and memory consumption.
        This tool will use Visdom.
        """
        if self.args["monitor_cpu"] or self.args["monitor_gpu"]:
            from digideep.utility.stats import StatVizdom
            sv = StatVizdom(monitor_cpu=self.args["monitor_cpu"], monitor_gpu=self.args["monitor_gpu"])
            sv.start()

    def dump_cpanel(self, cpanel):
        dump_dict_as_json(self.state['file_cpanel'], cpanel)
    def dump_params(self, params):
        dump_dict_as_yaml(self.state['file_params'], params)
    
    def set_device(self):
        ## CPU
        # Sets the number of OpenMP threads used for parallelizing CPU operations
        torch.set_num_threads(1)
        
        ## GPU
        cuda_available = torch.cuda.is_available()
        if cuda_available: # and use_gpu:
            logger("GPU available. Using 1 GPU.")
            self.device = torch.device("cuda")
            # self.dtype = torch.cuda.FloatTensor
            # self.dtypelong = torch.cuda.LongTensor
        else:
            logger("Using CPUs.")
            self.device = torch.device("cpu")
            # self.dtype = torch.FloatTensor
            # self.dtypelong = torch.LongTensor
        
        # TODO: For debugging
        # self.device = torch.device("cpu")
    def get_device(self):
        return self.device
    
    #################################
    # Apparatus for model save/load #
    #################################
    # TODO: Copying the visdom.log file to the current session for replaying.
    def load_states(self):
        filename = os.path.join(self.args["load_checkpoint"], "states.pt")
        states = torch.load(filename, map_location=self.device)
        return states
    def load_runner(self):
        try:
            filename = os.path.join(self.args["load_checkpoint"], "runner.pt")
            runner = pickle.load(open(filename,"rb"))
        except:
            logger.fatal("Checkpoint could not be loaded successfully!")
            exit()
        return runner

    def save_states(self, states, index):
        import torch
        dirname = os.path.join(self.state['path_checkpoints'], "checkpoint-"+str(index))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, "states.pt")
        torch.save(states, filename)
    def save_runner(self, runner, index):
        dirname = os.path.join(self.state['path_checkpoints'], "checkpoint-"+str(index))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, "runner.pt")
        pickle.dump(runner, open(filename, "wb"), pickle.HIGHEST_PROTOCOL)
        logger.warn('>>> Network runner saved at {}\n'.format(dirname))
    #################################

    def __getitem__(self, key):
        return self.state[key]
    def __setitem__(self, key, value):
        self.state.update({key: value})
    def __delitem__(self, key):
        del self.state[key]

    def parse_arguments(self):
        # A bunch of arguments can come here!
        # These arguments are not saved!
        parser = argparse.ArgumentParser()
        ## Save/Load
        parser.add_argument('--load-checkpoint', metavar=('<path>'), default='', type=str, help="Load a checkpoint to resume training from that point.")
        parser.add_argument('--play', action="store_true", help="Will play the stored policy.")
        ## Session
        parser.add_argument('--session-path', metavar=('<path>'), default='/tmp', type=str, help="The path to store the sessions. Default is in /tmp")
        parser.add_argument('--log-level', metavar=('<n>'), default=1, type=int, help="The logging level: 0 (debug and above), 1 (info and above), 2 (warn and above), 3 (error and above), 4 (fatal and above)")
        ## Visdom Server
        parser.add_argument('--visdom', action='store_true', help="Whether to use visdom or not!")
        parser.add_argument('--visdom-port', metavar=('<n>'), default=8097, type=int, help="The port of visdom server, it's on 8097 by default.")
        ## Monitor Thread
        parser.add_argument('--monitor-cpu', action="store_true", help="Use to monitor CPU resource statistics on Visdom.")
        parser.add_argument('--monitor-gpu', action="store_true", help="Use to monitor GPU resource statistics on Visdom.")
        ## Parameters
        parser.add_argument('--params', metavar=('<name>'), default='', type=str, help="Choose the parameter set.")
        parser.add_argument('--cpanel', metavar=('<json dictionary>'), default=r'{}', type=JsonDecoder, help="Set the parameters of the cpanel by a json dictionary.")

        # NOTE: No default value for params. MUST be specified explicitly. "digideep.params.mujoco"
        ##
        # parser.add_argument('--override', action='store_true', help="Provide this option to explicitly override saved options with new options.")
        # Override option should explicitly be set if you want to use "input-params".
        args = parser.parse_args()

        self.args = vars(args)
