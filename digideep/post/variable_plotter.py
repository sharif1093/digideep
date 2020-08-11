import os
import pickle
import numpy as np


## Plotting modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# sns.set_style('whitegrid')

from .base import BasePlotter, get_os_friendly_name

## Generic Functions
def moving_average(array, window_size=5, mode='full'):
    weights = np.ones(window_size) / window_size
    ma = np.convolve(array, weights, mode=mode)
    if mode == 'full':
        return ma[:-window_size+1]
    elif mode=='same':
        return ma
    elif mode=='valid':
        print("Size is reduced!")
        return ma

def trim_to_shortest(signals):
    length = len(signals[0])
    for sig in signals:
        length = min(len(sig), length)
    
    for k in range(len(signals)):
        signals[k] = signals[k][:length]

    return signals

def get_signal_ma(signal, window_size=None):
    num = len(signal)
    window_size = window_size or np.max([int(num/15),5])
    signal_ma = moving_average(signal, window_size=window_size, mode='full')
    return signal_ma

## A class for plotting
# https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib/9890599
# https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/axes_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axes-demo-py
# https://matplotlib.org/users/pyplot_tutorial.html
# https://docs.scipy.org/doc/scipy/reference/signal.html
# Averaging: https://becominghuman.ai/introduction-to-timeseries-analysis-using-python-numpy-only-3a7c980231af
class VarPlot (BasePlotter):
    # def __init__(self, loaders, output_dir = None, **options):
    #     self.loaders = loaders
    #     self.output_dir = output_dir
    #     self.options = options
    #     # Keys: ['epoch', 'frame', 'episode',   'std', 'num', 'min', 'max', 'sum']

    def plot(self, key):
        """
        This function plots the reward function and saves the `.png`, `.pdf`, and `.pkl` files.
        
        To later reload the `.pkl` file, one can use the following (to change format, labels, etc.):

        Example:

            import matplotlib.pyplot as plt
            %matplotlib notebook
            import pickle
            ax = pickle.load( open( "path_to_pkl_figure.pkl", "rb" ) )
            ax.set_title("Alternative title")
            plt.show()
        
        See:
            https://stackoverflow.com/a/12734723

        """
        fig, ax = self.init_plot()
        abscissa_key = self.options.get("abscissa_key", "episode") # episode | epoch | frame | clock
        limits = [None, None, None, None]

        for subloaders in self.loaders:
            varlogs = [l.getVarlogLoader() for l in subloaders]
            new_limit = self.plot_stack(fig, ax, varlogs, key, abscissa_key)
            limits = self.update_plot_limits(limits, new_limit)
        
        self.set_plot_options(fig, ax, abscissa_key, limits)
        self.close_plot(fig, ax, path=self.output_dir, name=get_os_friendly_name(key))


    def update_plot_limits(self, limits, new_limit):
        for i in [0, 2]: # x_min and y_min
            if limits[i] is not None:
                limits[i] = min(limits[i], new_limit[i])
            else:
                limits[i] = new_limit[i]
        for i in [1, 3]: # x_max and y_max
            if limits[i] is not None:
                limits[i] = max(limits[i], new_limit[i])
            else:
                limits[i] = new_limit[i]
        return limits

    def plot_stack(self, fig, ax, varlogs, key, abscissa_key):
        ###########################################
        ### Get ordinate and abscissa from keys ###
        ###########################################
        # Ordinate
        window_size = self.options.get("window_size", 5)
        ordinate_stack_ave = [var[key]["sum"]/var[key]["num"] for var in varlogs]
        ordinate_stack_min = [var[key]["min"] for var in varlogs]
        ordinate_stack_max = [var[key]["max"] for var in varlogs]

        ordinate_stack_ave_ma = [get_signal_ma(signal, window_size=window_size) for signal in ordinate_stack_ave]
        # Choose the shortest signal from all loaders. Because they are all going to be shown on the same plot.
        stack = trim_to_shortest(ordinate_stack_ave_ma)
        ordinate_max = np.max(stack,  axis=0)
        ordinate_min = np.min(stack,  axis=0)
        ordinate_ave = np.mean(stack, axis=0)
        
        # Abscissa
        abscissa_all = [var[key][abscissa_key] for var in varlogs]
        abscissa_all_trimmed = trim_to_shortest(abscissa_all)
        abscissa = abscissa_all_trimmed[0]

        if abscissa_key == "clock":
            abscissa = abscissa - abscissa[0]

        ##################################
        ### Plot and fill the variance ###
        ##################################
        ax.plot(abscissa, ordinate_ave) # linewidth=1.5
        ax.fill_between(abscissa, ordinate_max, ordinate_min, alpha=0.2) # color='gray'

        xlim_min = np.min(abscissa)
        xlim_max = np.max(abscissa)
        ylim_min = np.min(ordinate_min)
        ylim_max = np.max(ordinate_max)
        return (xlim_min, xlim_max, ylim_min, ylim_max)


    def set_plot_options(self, fig, ax, abscissa_key, limits):
        #########################
        ### Set plot settings ###
        #########################
        # Set labels and titles
        title_default  = ""
        title = self.options.get("title",  title_default)

        xlabel_default = abscissa_key
        xlabel = self.options.get("xlabel", xlabel_default)

        ylabel_default = "return (average of episodic rewards)"
        ylabel = self.options.get("ylabel", ylabel_default)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Managing axis limit
        xlim_min = self.options.get("xlim_min", limits[0])
        xlim_max = self.options.get("xlim_max", limits[1])
        ylim_min = self.options.get("ylim_min", limits[2])
        ylim_max = self.options.get("ylim_max", limits[3])

        ax.set(xlim=(xlim_min, xlim_max))
        ax.set(ylim=(ylim_min, ylim_max))

        # Managing x axis ticks
        xticks_num = self.options.get("xticks_num", 4)
        ax.set(xticks=np.linspace(start=xlim_min, stop=xlim_max, num=xticks_num, endpoint=True).astype(np.int))

        if abscissa_key == "episode" or abscissa_key == "epoch":
            xlabels = ['{:d}'.format(int(x)) for x in ax.get_xticks()]
            ax.set_xticklabels(xlabels)
        elif abscissa_key == "frame":
            # TODO: Consider other cases where abscissa can have "K" unit, etc.
            # frame_unit = self.options.get("frame_unit", "m")
            xlabels = ['{:,.2f}'.format(x) + 'M' for x in ax.get_xticks()/1e6]
            ax.set_xticklabels(xlabels)
        elif abscissa_key == "clock":
            xticks = ax.get_xticks()
            xlabels = ['{:.2f}'.format( float(x)/60. ) + 'm' for x in xticks]
            ax.set_xticklabels(xlabels)

        # Set the legends
        legends = self.options.get("legends", None)
        legends_option = self.options.get("legends_option", {})
        # loc="upper left"/"best"/"lower right" | ncol=2 | frameon=false
        if legends is not None:
            ax.legend(labels=legends, **legends_option)
    

