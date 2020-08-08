import os, sys
import yaml, json

from digideep.pipeline import Session
from digideep.utility.logging import logger
from digideep.utility.toolbox import get_class, get_module, strict_update

def main(session):
    ##########################################
    ### LOOPING ###
    ###############
    # 1. Loading
    if session.is_loading:
        params = session.update_params({})
        # Summary
        logger.warn("="*50)
        logger.warn("Session:", params["session_name"])
        logger.warn("Message:", params["session_msg"])
        logger.warn("Command:\n\n$", params["session_cmd"], "\n")
        logger.warn("-"*50)

        runner = session.load_runner()
        # params = runner.params
    else:
        ##########################################
        ### LOAD FRESH PARAMETERS ###
        #############################
        # Import method-specific modules
        ParamEngine = get_module(session.args["params"])
        cpanel = strict_update(ParamEngine.cpanel, session.args["cpanel"])
        params = ParamEngine.gen_params(cpanel) ## Generate params from cpanel everytime

        # Storing parameters in the session.
        params = session.update_params(params)
        session.dump_cpanel(cpanel)
        session.dump_params(params)

        # Summary
        logger.warn("="*50)
        logger.warn("Session:", params["session_name"])
        logger.warn("Message:", params["session_msg"])
        logger.warn("Command:\n\n$", params["session_cmd"], "\n")
        logger.warn("-"*50)
        # logger.info("Hyper-Parameters\n\n{}".format(yaml.dump(params, indent=2)) )
        logger.warn("Hyper-Parameters\n\n{}".format(json.dumps(cpanel, indent=4, sort_keys=False)) )
        logger.warn("="*50)
        ##########################################
        
        Runner = get_class(params["runner"]["name"])
        runner = Runner(params)

    # If we are creating the session only, we do not even need to start the runner.
    if session.is_session_only:
        session.save_runner(runner, 0)
        logger.fatal("Session created; exiting ...")
        return
    
    # 2. Initializing: It will load_state_dicts if we are in loading mode
    runner.start(session)
    
    # 3. Train/Enjoy
    if session.is_playing:
        runner.enjoy()
    else:
        runner.train()
    ##########################################

if __name__ == "__main__":
    session = Session(root_path=os.path.dirname(os.path.realpath(__file__)))
    try:
        main(session)
    finally:
        session.finalize()
