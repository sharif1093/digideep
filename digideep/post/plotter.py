"""A module used for post-processing of saved sessions.
"""

import sys, os, glob
import argparse
# import time

## For loading a class by name
# from digideep.utility.toolbox import get_module
from digideep.utility.toolbox import get_class
from digideep.utility.json_encoder import JsonDecoder
from .variable_plotter import VarPlot
from .profile_plotter import ProPlot
from .monitor_plotter import MonPlot



if __name__=="__main__":
    print(">>> PLOTTER is called :)")
    # import sys
    # print(" ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root-dir', metavar=('<path>'), default='/tmp/digideep_sessions', type=str, help="The root directory of sessions.")
    parser.add_argument('-i', '--session-names', metavar=('<path>'), type=str, nargs='+', action='append', required=True, help="Path to all input sessions in the root path. `--session-names session_*_*`")
    parser.add_argument('-o', '--output-dir', metavar=('<path>'), default='', type=str, help="Path to store the output plot.")
    parser.add_argument('-t', '--type', metavar=('<type>'), default='variable', type=str, help="Select the type of graph to be drawn from [variable, profiler]")
    
    parser.add_argument('--options', metavar=('<json dictionary>'), default=r'{}', type=JsonDecoder, help="Set the options as a json dict.")
    args = parser.parse_args()

    # Expand session_names
    for index in range(len(args.session_names)):
        args.session_names[index] = [os.path.relpath(t, args.root_dir) for y in args.session_names[index] for t in sorted(glob.glob(os.path.join(args.root_dir, y)))]

    # print("Commandline was:\n  ", " ".join(sys.argv[:]) )
    print("Added sessions are:\n  ", args.session_names)

    if args.session_names == [[]]:
        print("No sessions were added ...")
        sys.exit(1)
    
    if args.output_dir == '' and len(args.session_names) == 1:
        args.output_dir = os.path.join(args.session_names[0][0], 'plots')
    output_dir = os.path.join(args.root_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Change the PYTHONPATH to load the saved modules for more compatibility.
    # TODO: Why?
    sys.path.insert(0, args.root_dir)

    loaders = []
    for sublist in args.session_names:
        subloaders = []
        for s in sublist:
            subloaders += [get_class(s + "." + "loader")]
        loaders += [subloaders]

    if args.type == "variable":
        pp = VarPlot(loaders, output_dir, **args.options)
        for key in args.options.get("key", ["/reward/train/episodic"]):
            pp.plot(key)
    elif args.type == "monitor":
        pp = MonPlot(loaders, output_dir, **args.options)
        for key in args.options.get("key", ["/cpu/per"]):
            pp.plot(key)
    elif args.type == "profiler":
        pp = ProPlot(loaders, output_dir, **args.options)
        pp.plot()
        
