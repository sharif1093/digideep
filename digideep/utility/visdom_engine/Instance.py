import visdom
import os.path

class VisdomInstance(object):
    """
    This class is a singleton for getting an instance of Visdom client.
    It also replays all the logs at the loading time.
    
    :class:`~digideep.pipeline.session.Session` is responsible for initializing the log_file
    and replaying the old log.

    Args:
        port (int): The port number of the running Visdom server.
        log_to_filename (str): The log file for the Visdom server.
        replay (bool, False): Whether to replay from existing Visdom log files in the path.
            Use with care if the log file is very big.

    """
    # This is a static member
    __viz_instance = None

    @staticmethod
    def getVisdomInstance(**kwargs):
        """ Static access method. """
        if VisdomInstance.__viz_instance == None:
            VisdomInstance(**kwargs)
        
        return VisdomInstance.__viz_instance

    def __init__(self, port=8097, log_to_filename=None, replay=True):
        if VisdomInstance.__viz_instance != None:
            raise Exception("This class is a singleton!")
        else:
            VisdomInstance.__viz_instance = visdom.Visdom(server="http://localhost", port=port, log_to_filename=log_to_filename)
            assert VisdomInstance.__viz_instance.check_connection(), 'No connection could be formed quickly'
            
            # We replay the log-file if it already exists and 'replay' option is set to True:
            if replay and os.path.isfile(log_to_filename):
                VisdomInstance.__viz_instance.replay_log(log_to_filename)

# Guidelines
# * Copy old visdom log to the location of the current log
# * Use win for every plot, text, property, etc.

# TODO: If replay is off, we should delete the logfile if it already exists (?)
