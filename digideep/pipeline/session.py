import os, sys, datetime, argparse
from shutil import copytree, copyfile, ignore_patterns
import pipes
import subprocess
import threading
import time

from digideep.utility.logging import logger
from digideep.utility.toolbox import dump_dict_as_json, dump_dict_as_yaml, get_module
from digideep.utility.json_encoder import JsonDecoder
from digideep.utility.monitoring import monitor
from digideep.utility.profiling import profiler, KeepTime
from digideep.utility.name_generator import get_random_name

import pickle, torch
from copy import deepcopy

def generateTimestamp():
    # Always uses UTC as timezone
    now = datetime.datetime.now()
    timestamp = '{:%Y%m%d%H%M%S}'.format(now)
    return timestamp

def make_unique_path_session(path_base_session, prefix="session_"):
    session_name = prefix + generateTimestamp() + "_" + get_random_name()
    path_session = os.path.join(path_base_session, session_name)
    try:
        os.makedirs(path_session)
        return path_session
    except FileExistsError as e:
        return make_unique_path_session(path_base_session=path_base_session, prefix=prefix)


class ParCommand(threading.Thread):
    def __init__(self, command, logger):
        self.stdout = None
        self.stderr = None
        self.command = command
        self.logger = logger
        threading.Thread.__init__(self)
    def run(self):
        t = time.time()
        p = subprocess.Popen(self.command,
                             shell=False,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        self.stdout, self.stderr = p.communicate()
        p.wait()
        # self.logger.warn(" Thread '", " ".join(self.command), "' is over with exit code = ", p.returncode, " in ",time.time()-t," seconds.", sep="")
        self.logger.warn("Command: '{}' is over with exit code: {} in {:6.2f} seconds".format(" ".join(self.command),
                                                                                             p.returncode,
                                                                                             time.time()-t))
                        
        # t.wait()
        # t.poll()


writers = []

class Session(object):
    """
    This class provides the utilities for storing results of a session.
    It provides a unique path based on a timestamp and creates all sub-
    folders that are required there. A session directory will have the
    following contents:

    * :file:`session_YYYYMMDDHHMMSS/`:
    
        * :file:`checkpoints/`: The directory of all stored checkpoints.
        * :file:`modules/`: A copy of all modules that should be saved with the results. This helps to load
          checkpoints in evolving codes with breaking changes. Use extra modules with ``--save-modules``
          command-line option.
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
    

                                         play    resume    loading    dry-run    implemented
    ----------------------------------------------------------------------------------------
    Train                                 0         0         0          0            1
    Train from a checkpoint               0         1         1          0            0
    Play (policy initialized)             1         0         0         0/1           1
    Play (policy loaded from checkpoint)  1         0         1         0/1           1

    """
    def __init__(self, root_path):
        self.parse_arguments()

        # If '--dry-run' is specified no reports should be generated. It is not relevant to whether
        # we are loading from a checkpoint or running from scratch. If dry-run is there no reports
        # should be generated.
        self.dry_run = True if self.args["dry_run"] else False

        # TODO: If loading from a checkpoint, we must copy the visdom log
        #       from the previous path to the current one.
        self.is_loading = True if self.args["load_checkpoint"] else False
        self.is_playing = True if self.args["play"] else False
        self.is_resumed = True if self.args["resume"] else False
                
        # TODO: Change the path for loading the packages?
        # sys.path.insert(0, '/path/to/whatever')

        # if self.args["monitor_cpu"] or self.args["monitor_gpu"]:
        #     # Force visdom ON if "--monitor-cpu" or "--monitor-gpu" are provided.
        #     self.args["visdom"] = True

        self.state = {}
        # Root: Indicates where we are right now
        self.state['path_root'] = os.path.split(root_path)[0]
        
        # Session: Indicates where we want our codes to be stored
        if self.is_loading and self.is_playing:
            # If we are playing a recorded checkpoint, we must save the results into the `evaluations` path
            # of that session.
            checkpoint_path = os.path.split(self.args["load_checkpoint"])[0]
            self.state['path_base_sessions'] = os.path.join(os.path.split(checkpoint_path)[0], "evaluations")
        elif self.is_loading and self.is_resumed:
            if self.args['session_name']:
                print("Warning: --session-name is ignored.")

            directory = os.path.dirname(os.path.dirname(self.args["load_checkpoint"]))
            self.state['path_base_sessions'] = os.path.split(directory)[0]
            self.args['session_name'] = os.path.split(directory)[1]
        elif self.is_loading:
            raise Exception("--load-checkpoint should be used either with --play or --resume.")
        else:
            # OK, we are loading from a checkpoint, just create session from scratch.
            # self.state['path_root_session']  = self.args["session_path"]
            # self.state['path_base_sessions'] = os.path.join(self.state['path_root_session'], 'digideep_sessions')
            self.state['path_base_sessions'] = self.args["session_path"]
            

        # 1. Creating 'path_base_sessions', i.e. '/tmp/digideep_sessions':
        try: # TODO: and not self.dry_run:
            os.makedirs(self.state['path_base_sessions'])
            # Create an empty __init__.py in it!
            with open(os.path.join(self.state['path_base_sessions'], '__init__.py'), 'w') as f:
                print("", file=f)
        except Exception as ex:
            print(ex)

        # 2. Create a unique 'path_session':
        if not self.dry_run:
            if self.args['session_name']:
                self.state['path_session'] = os.path.join(self.state['path_base_sessions'], self.args["session_name"])
            else:
                self.state['path_session'] = make_unique_path_session(self.state['path_base_sessions'], prefix="session_")
        else:
            self.state['path_session'] = os.path.join(self.state['path_base_sessions'], "no_session")
        
        
        self.state['session_name'] = os.path.split(self.state['path_session'])[-1]

        self.state['path_checkpoints'] = os.path.join(self.state['path_session'], 'checkpoints')
        self.state['path_memsnapshot'] = os.path.join(self.state['path_session'], 'memsnapshot')
        self.state['path_monitor']     = os.path.join(self.state['path_session'], 'monitor')
        self.state['path_videos']      = os.path.join(self.state['path_session'], 'videos')
        self.state['path_tensorboard'] = os.path.join(self.state['path_session'], 'tensorboard')
        # Hyper-parameters basically is a snapshot of intial parameter engine's state.
        self.state['file_cpanel'] = os.path.join(self.state['path_session'], 'cpanel.json')
        self.state['file_params'] = os.path.join(self.state['path_session'], 'params.yaml')
        self.state['file_report'] = os.path.join(self.state['path_session'], 'report.log')
        # self.state['file_visdom'] = os.path.join(self.state['path_session'], 'visdom.log')
        self.state['file_varlog'] = os.path.join(self.state['path_session'], 'varlog.json')
        self.state['file_prolog'] = os.path.join(self.state['path_session'], 'prolog.json')

        # 3. Creating the rest of paths:
        if not self.is_playing and not self.is_resumed and not self.dry_run:
            os.makedirs(self.state['path_checkpoints'])
            os.makedirs(self.state['path_memsnapshot'])
        if not self.is_resumed and not self.dry_run:
            os.makedirs(self.state['path_monitor'])
        
        
        self.initLogger()
        self.initVarlog()
        self.initProlog()
        self.initTensorboard()
        # self.initVisdom()
        # TODO: We don't need the "SaaM" when are loading from a checkpoint.
        # if not self.is_playing:
        if not self.is_loading:
            self.createSaaM()
        #################
        self.runMonitor() # Monitor CPU/GPU/RAM
        self.set_device()

        # Check valid params file:
        if not self.is_loading:
            try:
                get_module(self.args["params"])
            except Exception as ex:
                logger.fatal("While importing user-specified params:", ex)
                exit()
        if self.is_loading:
            logger.warn("Loading from:", self.args["load_checkpoint"])
        
        if not self.dry_run:
            print(':: The session will be stored in ' + self.state['path_session'])
        else:
            print(':: This session has no footprints. Use without `--dry-run` to store results.')

    def finalize(self):
         pass
    def initLogger(self):
        """
        This function sets the logger level and file.
        """
        if not self.dry_run:
            logger.set_logfile(self.state['file_report'])
        logger.set_log_level(self.args["log_level"])
    
    def initVarlog(self):
        if not self.dry_run:
            monitor.set_output_file(self.state['file_varlog'])
    
    def initProlog(self):
        if not self.dry_run:
            profiler.set_output_file(self.state['file_prolog'])
        KeepTime.set_level(self.args["profiler_level"])
    
    def initTensorboard(self):
        """
        Will initialize the SummaryWriter for tensorboard logging.
        
        Link: https://pytorch.org/docs/stable/tensorboard.html
        """
        # TODO: Is it required?
        # if self.dry_run:
        #     logger.warn("Tensorboard initialization was ignored due to --dry-run argument.")
        #     return

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.state['path_tensorboard'])
        
        # Put it here for global access to tensorboard!
        writers.append(self.writer)

        if self.args["tensorboard"]:
            # Run a dedicated Tensorboard server:
            from tensorboard import program
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--bind_all', '--logdir', self.state['path_tensorboard']])
            url = tb.launch()
            logger.warn("Access Tensorboard through: " + str(url))
        else:
            # Nullify the attributes so time would not be wasted logging.
            for attr in dir(self.writer):
                if attr.startswith("add_") or (attr=="flush") or (attr=="close"):
                    setattr(self.writer, attr, lambda *args, **kw: None)    

    # def initVisdom(self):
    #     """
    #     This function initializes the connection to the Visdom server. The Visdom server must be running.
    #
    #     .. code-block:: bash
    #         :caption: Running visdom server
    #
    #         visdom -port 8097 &
    #     """
    #     if self.args["visdom"]:
    #         from digideep.utility.visdom_engine.Instance  import VisdomInstance
    #         if not self.dry_run:
    #             VisdomInstance(port=self.args["visdom_port"], log_to_filename=self.state["file_visdom"], replay=True)
    #         else:
    #             VisdomInstance(port=self.args["visdom_port"])

    def createSaaM(self):
        """ SaaM = Session-as-a-Module
        This function will make the session act like a python module.
        The user can then simply import the module for inference.
        """
        if self.dry_run:
            return
        # Copy the all modules
        modules = set(self.args["save_modules"])
        # Add digideep per se to the saved modules.
        modules.add("digideep")
        modules_path = os.path.join(self.state['path_session'], 'modules')
        for mod in modules:
            real_mod = get_module(mod)
            module_source = real_mod.__path__[0]
            module_target = os.path.join(modules_path, mod)
            copytree(module_source, module_target, ignore=ignore_patterns('*.pyc', '__pycache__'))
            if mod == "digideep":
                digideep_path = module_source

        # Copy saam.py to the session root path
        copyfile(os.path.join(digideep_path, 'saam.py'), os.path.join(self.state['path_session'], 'saam.py'))
        # Create __init__.py at the session root path
        with open(os.path.join(self.state['path_session'], '__init__.py'), 'w') as f:
            print("from .saam import loader", file=f)
            # print("from .loader import ModelCarousel", file=f)

    def runMonitor(self):
        """
        This function will load the monitoring tool for CPU and GPU utilization and memory consumption.
        """
        # TODO: StatVisdom is deprecated. Update the following code.
        # if self.args["monitor_cpu"] or self.args["monitor_gpu"]:
        #     from digideep.utility.stats import StatVizdom
        #     sv = StatVizdom(monitor_cpu=self.args["monitor_cpu"], monitor_gpu=self.args["monitor_gpu"])
        #     sv.start()
        pass

    
    def update_params(self, params):
        params['session_name'] = self.state['session_name']
        params['session_msg'] = self.args['msg']
        params['session_cmd'] = 'python ' + ' '.join(pipes.quote(x) for x in sys.argv)
        return params

    def dump_cpanel(self, cpanel):
        if self.dry_run:
            return
        dump_dict_as_json(self.state['file_cpanel'], cpanel)
    def dump_params(self, params):
        if self.dry_run:
            return
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

    def mark_as_done(self):
        with open(os.path.join(self.state['path_session'], 'done.lock'), 'w') as f:
            print("", file=f)
    
    def rsync(self, source, target):
        t = ParCommand(['rsync', '-azP', '--delete', '--perms', '--chmod=ugo+rwx', source, target], logger)
        t.start()
        return t
    def take_memory_snapshop(self, memroot, name):
        target = os.path.join(self.state['path_memsnapshot'], name) + "/"
        t = self.rsync(source=memroot+"/", target=target)
        # Join, because memory will keep changing during "rsync".
        t.join()
    def load_memory_snapshot(self, memroot, name):
        source = os.path.join(self.state['path_memsnapshot'], name) + "/"
        t = self.rsync(source=source, target=memroot+"/")
        # Join, because we cannot proceed without this task already completed.
        t.join()

    #################################
    # Apparatus for model save/load #
    #################################
    # TODO: Copying the visdom.log file to the current session for replaying.
    def load_states(self):
        filename = os.path.join(self.args["load_checkpoint"], "states.pt")
        states = torch.load(filename, map_location=self.device)
        return states
    def load_runner(self):
        # If loading from a checkpoint, we must check the existence
        # of that path and whether that's a valid digideep session.
        # Existence is checked but validity is not. How is that?
        try:
            filename = os.path.join(self.args["load_checkpoint"], "runner.pt")
            runner = pickle.load(open(filename,"rb"))
        except Exception as ex:
            logger.fatal("Error loading from checkpoint:", ex)
            exit()
        return runner

    def save_states(self, states, index):
        if self.dry_run:
            return
        import torch
        dirname = os.path.join(self.state['path_checkpoints'], "checkpoint-"+str(index))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, "states.pt")
        torch.save(states, filename)
    def save_runner(self, runner, index):
        if self.dry_run:
            return
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
        ## Save/Load/Dry-run
        parser.add_argument('--load-checkpoint', metavar=('<path>'), default='', type=str, help="Load a checkpoint to resume training from that point.")
        parser.add_argument('--play', action="store_true", help="Will play the stored policy.")
        parser.add_argument('--resume', action="store_true", help="Will resume training the stored policy.")
        parser.add_argument('--dry-run', action="store_true", help="If used no footprints will be stored on disc whatsoever.")
        ## Session
        parser.add_argument('--session-path', metavar=('<path>'), default='/tmp/digideep_sessions', type=str, help="The path to store the sessions. Default is in /tmp")
        parser.add_argument('--session-name', metavar=('<name>'), default='', type=str, help="A default name for the session. Random name if not provided.")
        parser.add_argument('--save-modules', metavar=('<path>'), default=[], nargs='+', type=str, help="The modules to be stored in the session.")
        parser.add_argument('--log-level', metavar=('<n>'), default=1, type=int, help="The logging level: 0 (debug and above), 1 (info and above), 2 (warn and above), 3 (error and above), 4 (fatal and above)")
        parser.add_argument('--profiler-level', metavar=('<n>'), default=-1, type=int, help="Profiler level. '-1' profiles all level. Default: '-1'")
        parser.add_argument('--msg',  metavar=('<msg>'), default='', type=str, help="A message describing the current simulation and its significance.")
        ## Visdom Server
        parser.add_argument('--visdom', action='store_true', help="Whether to use visdom or not!")
        parser.add_argument('--visdom-port', metavar=('<n>'), default=8097, type=int, help="The port of visdom server, it's on 8097 by default.")
        ## Tensorboard
        parser.add_argument('--tensorboard', action='store_true', help="Whether to use Tensorboard or not!")
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

