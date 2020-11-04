from .variable_plotter import VarPlot
from .base import get_os_friendly_name

class MonPlot (VarPlot):
    def plot(self, keyx="clock", keyy=None):
        # assert len(keys) == 1, "We only support a single key at the moment."
        # key = keys[0]
        # abscissa_key = self.options.get("abscissa_key", "clock") # episode | epoch | frame | clock

        fig, ax = self.init_plot()
        
        limits = [None, None, None, None]

        for subloaders in self.loaders:
            monlogs = [l.getMonlogLoader() for l in subloaders]
            new_limit = self._plot_stack(fig, ax, monlogs, keyx, keyy)
            limits = self._update_plot_limits(limits, new_limit)
        
        self._set_plot_options(fig, ax, keyx, limits)
        self.close_plot(fig, ax, path=self.output_dir, name=get_os_friendly_name(keyy))

    

# "/cpu/per"
# "/cpu/all/total"
# "/cpu/memory/total"
# "/cpu/memory/used"
# "/cpu/memory/mem"
# "/gpu/load"
# "/gpu/memory/total"
# "/gpu/memory/used"
