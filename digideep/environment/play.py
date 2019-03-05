"""A module used for visualizing of environments. It is useful for debugging the environment.
See :ref:`ref-play-debug` for details on usage.
"""

import time
import numpy as np
import argparse

import gym
import digideep.environment.dmc2gym

from digideep.utility.toolbox import get_module

# TODO: Load a module from command-line to register user's models.

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-include', metavar=('<pattern>'), nargs='?', const='.*', type=str, help="List by a pattern")
    parser.add_argument('--list-exclude', metavar=('<pattern>'), nargs='?', const='a^', type=str, help="List by a pattern")

    parser.add_argument('--module', metavar=('<module_name>'), default='', type=str, help="The name of the module which will register the model in use.")
    parser.add_argument('--model', metavar=('<model_name>'), default='', type=str, help="The name of the model to play with random actions.")
    parser.add_argument('--runs', metavar=('<n>'), default=5, type=int, help="The number of times to run the simulation.")
    parser.add_argument('--n-step', metavar=('<n>'), default=1024, type=int, help="The number of timesteps to run each episode.")
    parser.add_argument('--delay', metavar=('<ms>'), default=0, type=int, help="The time in milliseconds to delay in each timestep to make simulation slower.")

    parser.add_argument('--no-action', action="store_true", help="The number of timesteps to run each episode.")
    args = parser.parse_args()

    if args.module:
        get_module(args.module)

    if args.list_include or args.list_exclude:
        from gym.envs.registration import registry
        import json, re

        keys = list(registry.env_specs.keys())
        keys.sort()

        if args.list_include:
            pattern_include = re.compile(args.list_include)
            keys = [k for k in keys if pattern_include.match(k)]
        if args.list_exclude:
            pattern_exclude = re.compile(args.list_exclude)
            keys = [k for k in keys if not pattern_exclude.match(k)]
        print("{}".format(json.dumps(keys, indent=4, sort_keys=False)))
        
    else:
        try:
            env = gym.make(args.model)

            for j in range(args.runs):
                print("iter: {}/{}".format(j+1, args.runs))
                env.reset()
                for index in range(args.n_step):
                    env.render()
                    if args.delay > 0:
                        time.sleep(args.delay/1000)
                    if not args.no_action:
                        # Take a random action
                        action = env.action_space.sample()
                        observation, reward, done, info = env.step(action)
        finally:
            print("We are executing finally!")
            env.close()
