"""A module used for post-processing of saved sessions.
"""

import sys, os, glob
import argparse
# import time

## For loading a class by name
# from digideep.utility.toolbox import get_module
from digideep.utility.toolbox import get_class
from digideep.utility.json_encoder import JsonDecoder

# print(" ".join(sys.argv))

def type_aliases(t):
    if t=="variable":
        return "digideep.post.VarPlot"
    elif t=="monitor":
        return "digideep.post.MonPlot"
    elif t=="profiler":
        return "digideep.post.ProPlot"
    else:
        return t

def printv(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)

if __name__=="__main__":
    """
    This module is used for plotting the monitored variables (`varlog.json`), profilings (`prolog.json`), and monitorings (`monlog.json`).

    .. code-block:: bash

        >>> python -m digideep.post.plotter 
        >>>

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root-dir', metavar=('<path>'), default='/tmp/digideep_sessions', type=str, help="The root directory of sessions.")
    parser.add_argument('-i', '--session-names', metavar=('<path>'), type=str, nargs='+', action='append', required=True, help="Path to all input sessions in the root path. `--session-names session_*_*`")
    parser.add_argument('-o', '--output-dir', metavar=('<path>'), default='', type=str, help="Path to store the output plot.")
    
    # parser.add_argument('--keys', metavar=('<json>'), default=r'[]',      type=JsonDecoder, help="Choose the key to plot if applicable.")

    parser.add_argument('--type', metavar=('<type>'), default='variable', type=str, help="Select the type of graph to be drawn from [variable, profiler]")
    parser.add_argument('--keyx', metavar=('<name>'), default='',         type=str, help="Choose the key for the x axis.")
    parser.add_argument('--keyy', metavar=('<name>'), default='',         type=str, help="Choose the key for the y axis.")
    parser.add_argument('--opts', metavar=('<json>'), default=r'{}',      type=JsonDecoder, help="Set the options as a json dict.")

    parser.add_argument('-v', '--verbose', action='store_true', help="Whether to print verbose messages.")
    args = parser.parse_args()

    printv("###############################################", verbose=args.verbose)
    printv("##            PLOTTER FOR DIGIDEEP           ##", verbose=args.verbose)
    printv("###############################################", verbose=args.verbose)

    # print("Keys provided are:", args.keys)
    # print("Options are:", args.opts)

    ### Process --session-names:
    for index in range(len(args.session_names)):
        args.session_names[index] = [os.path.relpath(t, args.root_dir) for y in args.session_names[index] for t in sorted(glob.glob(os.path.join(args.root_dir, y)))]
    if args.session_names == [[]]:
        printv("No sessions were added ...", verbose=args.verbose)
        sys.exit(1)
    else:
        # print("Commandline was:\n  ", " ".join(sys.argv[:]) )
        printv("Added sessions are:\n  ", args.session_names, verbose=args.verbose)
    
    # TODO Two things:
    #        1. All /'s should be replaced by .'s
    #        2. No .'s should be present in any of the names.
    for i in range(len(args.session_names)):
        for j in range(len(args.session_names[i])):
            if args.session_names[i][j].find('.') != -1:
                raise Exception("Session names cannot contain dot (.)! We have " + args.session_names[i][j])
            args.session_names[i][j] = args.session_names[i][j].replace("/", ".")

    
    ### TODO: What is the necessity of the following?
    count = 0
    for ss in args.session_names:
        for s in ss:
            count += 1
    
    if args.output_dir == '' and count == 1:
        output_dir = None
    else:
        # If --output-dir is relative then --root-dir is prefixed to it.
        output_dir = os.path.join(args.root_dir, args.output_dir)
    
    # Change the PYTHONPATH to load the saved modules for more compatibility.
    # TODO: Why?
    sys.path.insert(0, args.root_dir)

    ## Get the loaders from SaaM
    loaders = []
    for sublist in args.session_names:
        subloaders = []
        for s in sublist:
            subloaders += [get_class(s + "." + "loader")]
        loaders += [subloaders]

    #######################################
    ##          Actual Plotting          ##
    #######################################
    # Do the plotting
    plotter_class = get_class(type_aliases(args.type))
    pc = plotter_class(loaders, output_dir, **args.opts)
    pc.plot(keyx=args.keyx, keyy=args.keyy)
