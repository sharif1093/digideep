# SaaM: Session-as-a-Module

import sys, os
import numpy as np

session_path = os.path.dirname(os.path.realpath(__file__))
modules_path = os.path.join(session_path, "modules")
print("Adding the following path to PYTHONPATH:", modules_path)

# We need to insert the new path at the beginning of the list to
# override existing same-name modules.
sys.path.insert(0, modules_path)

import digideep
from digideep.utility.json_encoder import JsonDecoder
from digideep.utility.toolbox import load_yaml_as_dict, load_json_as_dict

class Loader:
    def __init__(self, root_path):
        self.root_path = root_path
        self.params = self._loadParams()
        self.cpanel = self._loadCpanel()
    
    @property
    def getRootPath(self):
        return self.root_path
    
    @property
    def getCheckpointsPath(self):
        return os.path.join(self.root_path, "checkpoints/*")
    
    # @property
    # def getPlotsPath(self):
    #     path = os.path.join(self.root_path, "plots/*")
    #     # TODO: Create the path if it does not exist.
    #     return path
    
    @property
    def _getParamsPath(self):
        return os.path.join(self.root_path, "params.yaml")
    def _loadParams(self):
        return load_yaml_as_dict(self._getParamsPath)
    
    @property
    def _getCpanelPath(self):
        return os.path.join(self.root_path, "cpanel.json")
    def _loadCpanel(self):
        return load_json_as_dict(self._getCpanelPath)

    @property
    def _getVarlogPath(self):
        return os.path.join(self.root_path, "varlog.json")
    def getVarlogLoader(self):
        return LogLoader(self._getVarlogPath)
    
    @property
    def _getPrologPath(self):
        return os.path.join(self.root_path, "prolog.json")
    def getPrologLoader(self):
        return LogLoader(self._getPrologPath)
    
    @property
    def _getStalogPath(self):
        return os.path.join(self.root_path, "stalog.json")
    def getStalogLoader(self):
        return LogLoader(self._getStalogPath)

    # def getEnv(self):
    #    To get an environment created with MakeEnvironment.
    #    All wrappers should be already applied.
    # def getRunner(self):
    #    To get the runner so that one can start playing the runner.
    # def getAgent(self): # getAgent(self, name)
    #    To get the agent that is trained and saved.

loader = Loader(session_path)

class LogLoader:
    def __init__(self, path_to_log_file):
        self.path_to_log_file = path_to_log_file
        self.data = {}
        # Actually load the log file.
        with open(self.path_to_log_file, 'r') as f:
            # Read line-by-line. Parse using JsonDecoder.
            for line in f:
                # Update a dictionary with all the values.
                self._update_by_entry(JsonDecoder(line))
        self._convert_to_numpy()
    
    def keys(self):
        return self.data.keys()
    def __setitem__(self, key, value):
        self.data.update({key: value})
    def __getitem__(self, key):
        return self.data[key]
    def __delitem__(self, key):
        del self.data[key]

    def _update_by_entry(self, entry):
        for key in entry["data"]:
            if key in self.data:
                for m in entry["meta"]:
                    self.data[key][m] += [entry["meta"][m]]
                for d in entry["data"][key]:
                    self.data[key][d] += [entry["data"][key][d]]
            else:
                self.data[key] = {}
                for m in entry["meta"]:
                    self.data[key][m] = [entry["meta"][m]]
                for d in entry["data"][key]:
                    self.data[key][d] = [entry["data"][key][d]]
    
    def _convert_to_numpy(self):
        for key in self.data:
            for subkey in self.data[key]:
                self.data[key][subkey] = np.asarray(self.data[key][subkey])
        

  