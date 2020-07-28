from digideep.environment.common.vec_env import VecEnvWrapper

class VecRandomState(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        
    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()
    
    def state_dict(self):        
        # print(">>>> state_dict of VecRandomState is called. <<<<")
        states = self.venv.unwrapped.get_rng_state()
        return states
    def load_state_dict(self, state_dict):
        # print(">>>> load_state_dict of VecRandomState is called. <<<<")
        self.venv.unwrapped.set_rng_state(state_dict)
