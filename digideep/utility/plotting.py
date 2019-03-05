from digideep.utility.visdom_engine.Wrapper import VisdomWrapperPlot
from digideep.utility.filter import MovingAverage

# It is only about plotting lines
class Plotter():
    def __init__(self, **params):
        """
        Params:
          - name (ordered dict): A dict of plot names and which aspect of them to be plotted
              For instance: {"Reward":["Mean", "Max"], "Loss":["Mean", "Median"], "Profile":[]}
          - visibility (dict): Which values to plot
          - win: Name of window in Visdom
          - env: Can specify the environment of the plot.
          - filterargs (dict): Dictionary of arguments of the filter. {"size":2, "window_size":10}
          - visdomargs (dict): Dictionary of arguments of visdom.
        
        Caution:
            We are assuming ordered entries in the name. The orders can be preserved
            by regular ``dict`` in Python 3.7+, or by using ``OrderedDict`` in all other
            versions. BE CAREFUL!
            Also, when trying to create an ``OrderedDict`` with inline loops, we must use
            ``dict`` and this already assumes order preservation of Dicts.
        
        Note:
            When there is only one entry there is no need to using ``OrderedDict``.
        
        Todo:
            Save the state of filter (moving average) and t. Implement ``state_dict`` and ``load_state_dict``.
        """
        # visdomargs = dict(opts = dict(title='Reward', xlabel='Episode', ylabel='Duration'))
        self.params = params
        
        # self.name
        ENV = self.params["env"] if "env" in self.params else "main"
        VISDOMARGS = self.params["visdomargs"] if "visdomargs" in self.params else {}
        self.v = VisdomWrapperPlot('line', env=ENV, win=self.params["win"], **VISDOMARGS)
        
        self.state = {}
        self.state["data"] = MovingAverage(size=len(self.params["name"]), **self.params["filterargs"])
        self.state["t"] = 0
    
    def __call__(self, *args, **kwargs):
        self.append(*args, **kwargs)

    def append(self, y, x=None):
        if x:
            self.state["t"] = x
        else:
            self.state["t"] += 1
        
        # The size of y will be checked in the following for consistency.
        self.state["data"].append(y)
        ymean = self.state["data"].mean
        ystd  = self.state["data"].std
        ymin  = self.state["data"].min
        ymax  = self.state["data"].max
        ymed  = self.state["data"].median

        args = {"t":self.state["t"], "y":y, "ymean":ymean, "ystd":ystd, "ymin":ymin, "ymax":ymax, "ymed":ymed}

        for index, item in enumerate(self.params["name"].items()):
            name, keys = item
            for key in keys:
                fcn = getattr(self, "_plot_"+key.lower())
                fcn(name, index, args)

    
    def _plot_main(self, name, index, args):
        self.v.append(X=args["t"], Y=args["y"][index], name=name)
    def _plot_max(self, name, index, args):
        self.v.append(X=args["t"], Y=args["ymax"][index], name=name+'_Max')
    def _plot_min(self, name, index, args):
        self.v.append(X=args["t"], Y=args["ymin"][index], name=name+'_Min')
    def _plot_mean(self, name, index, args):
        self.v.append(X=args["t"], Y=args["ymean"][index], name=name+'_Mean')
    def _plot_median(self, name, index, args):
        self.v.append(X=args["t"], Y=args["ymed"][i], name=name+'_Median')
    # TODO: Implement error envelopes for 2Sigma, 4Sigma, and 6Sigma
    def _plot_2sigma(self, name, index, args):
        self.v.append(X=args["t"], Y=args["ymean"][index]-1*args["ystd"][index], name=name+'_2Low')
        self.v.append(X=args["t"], Y=args["ymean"][index]+1*args["ystd"][index], name=name+'_2High')
    def _plot_4sigma(self, name, index, args):
        self.v.append(X=args["t"], Y=args["ymean"][index]-2*args["ystd"][index], name=name+'_4Low')
        self.v.append(X=args["t"], Y=args["ymean"][index]+2*args["ystd"][index], name=name+'_4High')
    def _plot_6sigma(self, name, index, args):
        self.v.append(X=args["t"], Y=args["ymean"][index]-3*args["ystd"][index], name=name+'_6Low')
        self.v.append(X=args["t"], Y=args["ymean"][index]+3*args["ystd"][index], name=name+'_6High')

