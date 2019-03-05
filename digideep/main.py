import os, sys
import yaml, json

from digideep.pipeline import Session, Runner
from digideep.utility.logging import logger
from digideep.utility.toolbox import get_module, strict_update

if __name__=="__main__":
    session = Session(root_path=os.path.dirname(os.path.realpath(__file__)))

    ##########################################
    ### LOOPING ###
    ###############
    # 1. Loading
    if session.is_loading:
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
        # print(ParamEngine)

        # Summary
        # cmd = ' '.join(sys.argv)
        # logger("Command:", cmd, "\n")
        # logger.info("Hyper-Parameters\n\n{}".format(yaml.dump(params, indent=2)) )
        logger.info("Hyper-Parameters\n\n{}".format(json.dumps(cpanel, indent=4, sort_keys=False)) )
        # Storing parameters in the session.
        session.dump_cpanel(cpanel)
        session.dump_params(params)
        ##########################################
        runner = Runner(params)
    
    # 2. Initializing: It will load_state_dicts if we are in loading mode
    runner.start(session)

    # 3. Train/Enjoy
    if session.is_playing:
        runner.enjoy()
    else:
        runner.train()
    ##########################################


