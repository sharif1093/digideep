from digideep.environment.common.vec_env import VecEnvWrapper

def get_type_name(cls):
    """Gets the name of a type.

    This function is used to produce a key for each wrapper to store its states in a dictionary of wrappers' states.

    Args:
        cls: The input class.
    Returns:
        str: Name of the class.
    """
    name = "{}:{}".format(cls.__class__.__module__, cls.__class__.__name__)
    # name = str(type(cls))
    return name

class VecSaveState(VecEnvWrapper):
    """
    A vectorized wrapper that saves the state of all wrappers.
    This wrapper must be the last wrapper around a VecEnv so
    the state_dict and load_state_dict functions are exposed.
    We also assume that each wrapper is used once, otherwise
    we will end up saving the state of the first of them.

    Args:
        venv: The VecEnv to be wrapped.
    """
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        
    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()
    
    def state_dict(self):
        states = {}
        venv = self
        while hasattr(venv, "venv"):
            venv = venv.venv
            name = get_type_name(venv)
            if hasattr(venv, "state_dict"):
                if name in states:
                    raise KeyError("The key "+name+" already exists in the wrapper stack!")
                states[name] = venv.state_dict()
        return states
    def load_state_dict(self, state_dict):
        venv = self
        while hasattr(venv, "venv"):
            venv = venv.venv
            name = get_type_name(venv)
            if hasattr(venv, "load_state_dict"):
                venv.load_state_dict(state_dict[name])
