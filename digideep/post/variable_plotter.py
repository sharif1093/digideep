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

def get_signal_ma(signal, window_size=None):
    shape = signal.shape
    # print("   >>>> num =", num)
    assert len(shape) <= 2, "Moving average supports 0 and 1 dimensional arrays."
    if len(shape) == 1:
        signal = signal.reshape(*shape,1)
        # Update shape then:
        shape = signal.shape

    num = shape[0]
    dim = shape[1]

    window_size = window_size or np.max([int(num/15),5])
    signal_ma = np.stack([moving_average(signal[:,index], window_size=window_size, mode='full') for index in range(dim)], axis=1)
    # Shape of signal_ma is (num, dim) now.
    return signal_ma


def trim_to_shortest(signals):
    """
    This function only cuts on the first axis of signals[0].
    Initial shape: signals[stack_dim,      signal_length, signal_dim]
    Final shape:   signals[stack_dim, min(signal_length), signal_dim]
    """
    length = len(signals[0])
    for sig in signals:
        length = min(len(sig), length)
    
    for k in range(len(signals)):
        signals[k] = signals[k][:length]

    return signals



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

    def plot(self, keyx="epoch", keyy=None):
        # keyx: episode, epoch, frame, clock, etc.

        # assert len(keys) == 1, "We only support a single key at the moment."
        # key = keys[0]
        
        fig, ax = self.init_plot()
        
        limits = [None, None, None, None]

        for index, subloaders in enumerate(self.loaders):
            varlogs = [l.getVarlogLoader() for l in subloaders]
            new_limit = self._plot_stack(fig, ax, index, varlogs, keyx, keyy)
            limits = self._update_plot_limits(limits, new_limit)
        
        self._set_plot_options(fig, ax, keyx, limits)
        self.close_plot(fig, ax, path=self.output_dir, name=get_os_friendly_name(keyy))


    def _update_plot_limits(self, limits, new_limit):
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

    def _plot_stack(self, fig, ax, index, log_stack, keyx, keyy):
        ###########################################
        ### Get ordinate and abscissa from keys ###
        ###########################################
        # Ordinate
        window_size_arg = self.options.get("window_size", 5)
        if isinstance(window_size_arg, list):
            if index < len(window_size_arg):
                window_size = window_size_arg[index]
            else:
                window_size = 5
                print("WARNING: We are using default window_size for index", index)
        else:
            window_size = window_size_arg

        # log_stack[0][keyy]
        
        ## Processing abscissa
        abscissa_all = [var[keyy][keyx] for var in log_stack]
        abscissa_all_trimmed = trim_to_shortest(abscissa_all)
        abscissa = abscissa_all_trimmed[0]

        if keyx == "clock":
            abscissa = abscissa - abscissa[0]
        

        ## Processing ordinate
        ordinate_stack_ave = []
        for var in log_stack:
            # Obtain sizes:
            shape_sum = var[keyy]["sum"].shape
            shape_num = var[keyy]["num"].shape

            assert len(shape_sum) <= 2, "We only support scalars and 1d arrays as data."
            assert len(shape_num) <= len(shape_sum), "Shape of numbers should be less than shape of main data"
            assert shape_num[0] == shape_sum[0], "The first dimension of num and sum should be the same."

            shape = list(shape_num)
            while len(shape) < len(shape_sum):
                shape.append(1)

            ordinate_stack_ave += [var[keyy]["sum"]/var[keyy]["num"].reshape(shape)]

        
        
        ## TODO: We can use min and max if there is only one loader and users wants us to do so.
        # ordinate_stack_min = [var[key]["min"] for var in log_stack]
        # ordinate_stack_max = [var[key]["max"] for var in log_stack]

        ordinate_stack_ave_ma = [get_signal_ma(signal, window_size=window_size) for signal in ordinate_stack_ave]
        # Choose the shortest signal from all loaders. Because they are all going to be shown on the same plot.
        stack = np.array(trim_to_shortest(ordinate_stack_ave_ma))
        # stack = trim_to_shortest(ordinate_stack_ave_ma)
        ordinate_max = np.max(stack,  axis=0)
        ordinate_min = np.min(stack,  axis=0)
        ordinate_ave = np.mean(stack, axis=0)

        dim = ordinate_ave.shape[1]
        
        # print("ordinate_max.shape", ordinate_max.shape)
        # print(dim)
        # print("ordinate_min.shape", ordinate_min.shape)
        # print("ordinate_ave.shape", ordinate_ave.shape)
        # exit()

        # print("ordinate_max shape:", ordinate_max.shape)
        # print("stack shape:", stack.shape)
        # print("sum shape:", ordinate_stack_ave[0].shape)
        # print("-"*25)
        # # exit()
        
        ##################################
        ### Plot and fill the variance ###
        ##################################
        for d in range(dim):
            ax.plot(abscissa, ordinate_ave[:,d]) # linewidth=1.5
            ax.fill_between(abscissa, ordinate_max[:,d], ordinate_min[:,d], alpha=0.2) # color='gray'
        
        xlim_min = np.min(abscissa)
        xlim_max = np.max(abscissa)
        ylim_min = np.min(ordinate_min)
        ylim_max = np.max(ordinate_max)
        return (xlim_min, xlim_max, ylim_min, ylim_max)


    def _set_plot_options(self, fig, ax, abscissa_key, limits):
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
    

