from .variable_plotter import VarPlot
from .base import get_os_friendly_name

class MonPlot (VarPlot):
    def plot(self, key):
        fig, ax = self.init_plot()
        abscissa_key = self.options.get("abscissa_key", "clock") # episode | epoch | frame | clock
        limits = [None, None, None, None]

        for subloaders in self.loaders:
            monlogs = [l.getMonlogLoader() for l in subloaders]
            new_limit = self.plot_stack(fig, ax, monlogs, key, abscissa_key)
            limits = self.update_plot_limits(limits, new_limit)
        
        self.set_plot_options(fig, ax, abscissa_key, limits)
        self.close_plot(fig, ax, path=self.output_dir, name=get_os_friendly_name(key))

    

