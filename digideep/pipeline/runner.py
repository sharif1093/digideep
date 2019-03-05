import gc
import time

from digideep.environment import Explorer
from digideep.memory.generic import Memory
from digideep.utility.logging import logger
from digideep.utility.toolbox import seed_all, get_class, get_module
from digideep.utility.profiling import profiler, KeepTime
from digideep.utility.monitoring import monitor

# Runner should be irrelevent of torch, gym, dm_control, etc.

class Runner:
    """
    This class controls the main flow of the program. The main components of the class are:

    * explorer: A dictionary containing :class:`~digideep.environment.explorer.Explorer` for the three modes of ``train``, ``test``, and ``eval``.
      An :class:`~digideep.environment.explorer.Explorer` is a class which handles running simulations concurrently in several environments.
    * memory: The component responsible for storing the trajectories generated by the explorer.
    * agents: A dictionary containing all agents in the environment.

    This class also prints the :class:`~digideep.utility.profiling.Profiler` and :class:`~digideep.utility.monitoring.Monitor` information.
    Also the main serialization burden is on this class. The rest of classes only need to implement the ``state_dict`` and ``load_state_dict``
    functions for serialization.

    Caution:
        The lines of code for testing while training are commented out.
      
    """

    def __init__(self, params):
        self.params = params
        self.state = {}
        self.state["i_frame"] = 0
        self.state["i_cycle"] = 0
        self.state["i_epoch"] = 0
        self.state["loading"] = False

        profiler.reset()
        monitor.reset()

    def start(self, session):
        """A function to initialize the objects and load their states (if loading from a checkpoint).
        This function must be called before using the :func:`train` and :func:`enjoy` functions.

        If we are starting from scrarch, we will:

        * Instantiate all internal components using parameters.

        If we are loading from a saved checkpoint, we will:

        * Instantiate all internal components using old parameters.
        * Load all state dicts.
        * (OPTIONAL) Override parameters.
        """
        self.session = session
        seed_all(**self.params["runner"]["randargs"])

        # The order is as it is:
        self.instantiate()
        self.load()
        self.override()
    

    def instantiate(self):
        """
        This function will instantiate the memory, the explorers, and the agents with their specific parameters.
        """
        ## Instantiate Memory
        self.memory = Memory(self.session, **self.params["memory"])
        
        ## Instantiate Agents
        self.agents = {}
        action_generator = {}
        for agent_name in self.params["agents"]:
            agent_class = get_class(self.params["agents"][agent_name]["type"])
            self.agents[agent_name] = agent_class(self.session, self.memory, **self.params["agents"][agent_name])
        
        ## Instantiate Explorers
        # All explorers: train/test/eval
        self.explorer = {}
        self.explorer["train"] = Explorer(self.session, agents=self.agents, **self.params["explorer"]["train"])
        self.explorer["test"]  = Explorer(self.session, agents=self.agents, **self.params["explorer"]["test"])
        self.explorer["eval"]  = Explorer(self.session, agents=self.agents, **self.params["explorer"]["eval"])

    ###############################################################
    ### SERIALIZATION ###
    #####################
    def state_dict(self):
        """
        This function will return the states of all internal objects:

        * Agents
        * Explorer (only the ``train`` mode)
        * Memory

        Todo:
            Memory should be dumped in a separate file, since it can get really large.
            Moreover, it should be optional.
        """
        agents_state = {}
        for agent_name in self.agents:
            agents_state[agent_name] = self.agents[agent_name].state_dict()
        
        # explorer_state = {}
        # for explorer_name in self.explorer:
        #     explorer_state[explorer_name] = self.explorer[explorer_name].state_dict()
        ## Only the state of explorer["train"] is important for us.
        explorer_state = self.explorer["train"].state_dict()
        memory_state = self.memory.state_dict()
        return {'agents':agents_state, 'explorer':explorer_state, 'memory':memory_state}
    def load_state_dict(self, state_dict):
        """
        This function will load the states of the internal objects:

        * Agents
        * Explorers (state of ``train`` mode would be loaded for ``test`` and ``eval`` as well)
        * Memory
        """
        agents_state = state_dict['agents']
        for agent_name in agents_state:
            self.agents[agent_name].load_state_dict(agents_state[agent_name])
        
        explorer_state = state_dict['explorer']
        # for explorer_name in explorer_state:
        #     self.explorer[explorer_name].load_state_dict(explorer_state[explorer_name])
        self.explorer["train"].load_state_dict(explorer_state)
        # We do intentionally update the state of test/eval explorers with the state of train.
        self.explorer["test"].load_state_dict(explorer_state)
        self.explorer["eval"].load_state_dict(explorer_state)
    
    ###
    def override(self):
        pass

    #####################
    ###  SAVE RUNNER  ###
    #####################
    # UPON SAVING/LOADING THE RUNNER WITH THE SELF.SAVE FUNCTION:
    #   * save --> self.state_dict --> session.save_states --> torch.save --> states.pt
    #          |-> session.save_runner --> self.__getstate__ --> pickle.dump --> runner.pt
    #   * pickle.load --> __setstate__ 
    #     ... Later on ...
    #     --> self.start --> self.instantiate --> self.load --> session.load_states --> self.load_state_dict --> self.override
    # The __setstate__ and __getstate__ functions are for loading/saving the "runner" through pickle.dump / pickle.load
    # 
    def __getstate__(self):
        """
        This function is used by ``pickle.dump`` when we save the :class:`Runner`.
        This saves the ``params`` and ``state`` of the runner.
        """
        # This is at the time of pickling
        state = {'params':self.params, 'state':self.state}
        return state
    def __setstate__(self, state):
        """
        This function is used by ``pickle.load`` when we load the :class:`Runner`.
        """
        state['state']['loading'] = True
        self.__dict__.update(state)
    ###
        
    def save(self):
        """
        This is a high-level function for saving both the state of objects and the runner object.
        It will use helper functions from :class:`~digideep.pipeline.session.Session`.
        """
        if self.state["i_epoch"] % self.params["runner"]["save_int"] == 0:
            ## 1. state_dict: Saved with torch.save
            self.session.save_states(self.state_dict(), self.state["i_epoch"])
            ## 2. runner: Saved with pickle.dump
            self.session.save_runner(self, self.state["i_epoch"])
    def load(self): # This function does not directly work with files. Instead, it 
        """
        This is a function used by the :func:`start` function to load the states of internal objects 
        from the checkpoint and update the objects state dicts.
        """
        if self.state["loading"]:
            state_dict = self.session.load_states()
            self.load_state_dict(state_dict)
            self.state["loading"] = False
    ###############################################################

    
    def train(self):
        """
        The function that runs the training loop.

        .. code-block:: python
            :caption: The pseudo-code of training loop

            # Do a cycle
            while not done:
                # Explore
                chunk = explorer["train"].update()
                # Store
                memory.store(chunk)
                # Train
                for agent in agents:
                    agents[agent].update()

            log()
            test()
            save()
        
        See Also:
            :ref:`ref-how-runner-works`
        """
        try:
            while self.state["i_epoch"] < self.params["runner"]["n_epochs"]:
                self.state["i_cycle"] = 0
                while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                    with KeepTime("/"):
                        # 1. Do Experiment
                        with KeepTime("/explore"):
                            chunk = self.explorer["train"].update()
                        # 2. Store Result
                        with KeepTime("/store"):
                            self.memory.store(chunk)
                        # 3. Update Agent
                        for agent_name in self.agents:
                            with KeepTime("/update/"+agent_name):
                                self.agents[agent_name].update()
                    self.state["i_cycle"] += 1
                # End of Cycle

                self.state["i_epoch"] += 1
                # NOTE: We may save/test after each cycle or at intervals.

                # 1. Log
                self.log()
                # 2. Perform the test
                self.test()
                # 3. Save
                self.save()
                gc.collect() # Garbage Collection

        except (KeyboardInterrupt, SystemExit):
            logger.fatal('Operation stopped by the user ...')
        finally:
            logger.fatal('End of operation ...')


    def enjoy(self): #i.e. eval
        """This function evaluates the current policy in the environment. It only runs the explorer in a loop.

        .. code-block:: python

            # Do a cycle
            while not done:
                # Explore
                explorer["eval"].update()

            log()
        """
        try:
            self.explorer["eval"].reset()
            while True:
                # Cycles
                self.state["i_cycle"] = 0
                while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                    with KeepTime("/"):
                        # 1. Do Experiment
                        with KeepTime("/explore"):
                            self.explorer["eval"].update()
                    self.log()
                    self.state["i_cycle"] += 1
                # Log
        except (KeyboardInterrupt, SystemExit):
            logger.fatal('Operation stopped by the user ...')
        finally:
            logger.fatal('End of operation ...')

    def test(self):
        # Make the states of the two explorers train/test exactly the same, for the states of the environments.
        if self.params["runner"]["test_act"]:
            if self.state["i_epoch"] % self.params["runner"]["test_int"] == 0:
                self.explorer["test"].load_state_dict(self.explorer["train"].state_dict())
                self.explorer["test"].reset()
                self.explorer["test"].update()

    #####################
    ## Logging Summary ##
    #####################
    def log(self):
        """ The log function prints a summary of:

        * Frame rate and simulated frames.
        * Variables sent to the :class:`~digideep.utility.monitoring.Monitor`.
        * Profiling information, i.e. registered timing information in the :class:`~digideep.utility.profiling.Profiler`.
        """

        n_frame = self.params["explorer"]["train"]["num_workers"] * profiler.get_occurence("/explore/step")
        self.state["i_frame"] += n_frame
        # assert profiler.get_occurence("/explore/step") ==  self.params["explorer"]["train"]["n_steps"] * self.params["runner"]["n_cycles"]
        ## elapsed = profiler.get_time_average("/")
        elapsed = profiler.get_time_overall("/")
        overall = int(n_frame / elapsed)
        
        logger("---------------------------------------------------------")
        logger("Frame={:4.1e} | Epoch({:3d}cy)={:4d} | Overall({:4.1e}F/{:4.1f}s)={:4d}Hz".format(
                self.state["i_frame"],
                self.params["runner"]["n_cycles"],
                self.state["i_epoch"],
                n_frame,
                elapsed,
                overall
                )
              )
        # Printing profiling information:
        logger("PROFILING:\n"+str(profiler))
        # Printing monitoring information:
        logger("MONITORING:\n"+str(monitor))

        profiler.reset()
        monitor.reset()

        # TODO: We can check, and if self.session.args["visdom"] is true send results to visdom as well.

        print("")