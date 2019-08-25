class Averager:
    """A helper class to update the target models in the SAC method. It supports a ``hard`` mode and a ``soft`` mode.

    Args:
        model: The reference model.
        model_target: The target model (which will be a lagging version of the main model for stability).
        
        mode (str): The mode of updating which can be ``soft`` | ``hard``.
        polyak_factor (float): A number in :math:`[0,1]` which indicates the forgetting factor in the ``soft`` mode.
            The smaller this value the less it will forget the previous models and so is less sensitive to noise.
        interval (int): The interval of updating the target factor in the ``hard`` mode.
    """
    def __init__(self, model, model_target, **params):
        self.model = model
        self.model_target = model_target
        self.params = params

        self.state = {}
        # This will hold the counter for the hard mode updates
        self.state['update_counter'] = 0

    def update_target(self):
        """Upon calling this function the target model will be updated based on its ``mode``.
        """
        self.state['update_counter'] += 1
        interval = self.params.get("interval", 1)
        if self.state['update_counter'] % interval != 0:
            return

        mode = self.params["mode"]
        if mode == 'soft':
            polyak_factor = self.params["polyak_factor"]
            # y = TAU*x + (1 - TAU)*y
            for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - polyak_factor) + param.data * polyak_factor)
            self.state['update_counter'] = 0
        elif mode == 'hard':
            self.model_target.load_state_dict(self.model.state_dict())
            self.state['update_counter'] = 0
    def state_dict(self):
        return {"state":self.state}
    def load_state_dict(self, state_dict):
        self.state.update(state_dict["state"])

