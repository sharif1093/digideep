"""
Module to plot a pie chart for the profiled sections of the code.

TODO: Fix legend box size conflict. Make font of legends smaller.
    https://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend

TODO: Create several charts instead of hierarchic pie charts..
"""

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

import warnings

color_pool = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Purples, plt.cm.Greens, plt.cm.pink, plt.cm.gray, plt.cm.Reds,
              plt.cm.OrRd, plt.cm.magma, plt.cm.cividis, plt.cm.afmhot, plt.cm.PuBu, plt.cm.PuRd, plt.cm.ocean,
              plt.cm.autumn, plt.cm.winter]


######################

def aggregate_prolog(prolog):
    cum = {}
    num = {}
    avg = {}
    for k in prolog.keys():
        cum[k] = np.sum(prolog[k]["totals"])
        num[k] = np.sum(prolog[k]["occurs"])
        avg[k] = cum[k] / num[k]

    return cum, num, avg
# def remove_root(D):
#     val = D["/"]
#     del D["/"]
#     return val


#######################
# Routines for making the hierarchical data structure

def find_child_keys(keys, root):
    if root == "":
        return ["/"]
    a = []
    for k in keys:
        if (os.path.split(k)[0] == root) and (k != "/"):
            a.append(k)
    return sorted(a)
    

def retrieve_children(D, root="/", root_value=None, collapse=False, collapse_threshold=0.1, other_threshold=0.01):    
    labels = find_child_keys(D.keys(), root)
    values = []
    
    for l in labels:
        values.append(D[l])
    
    if not labels == []:
        # Sorting based on significance
        values, labels = zip(*sorted(zip(values, labels), reverse=True))
        labels = list(labels)
        values = list(values)
    
    
    # Decide on the total value.
    if root_value:
        T = root_value
    elif root in D:
        T = D[root]
    else:
        raise ValueError("We do not know the total value for", root)    
    
    S = sum(values)
    
    # Add "other" slot if discrepency is larger than 1% total:
    if S > T:
        raise ValueError("How can children overall be greater than total time? S={}, T={}, root={}".format(S, T, root))
    
    # Add "other" way in anyways ... We will remove it at the end if not that big.
    labels.append(os.path.join(root, "other"))
    values.append(T-S)
    
    # Remove those keys that are under 10% of total and add them to "other"
    if collapse:
        to_be_collapsed = []
        for i,l in enumerate(labels[:-1]):
            if values[i] < collapse_threshold * T:
                values[-1] += values[i]
                to_be_collapsed.append(i)
        
        for i in reversed(to_be_collapsed):
            del labels[i]
            del values[i]
    
    # Sort based on values.
    # Also check, if the total value is different than the root, then add a key called "other" at last.
    
    # If other key not big enough then remove it:
    if values[-1] < other_threshold * T:
        values = values[:-1]
        labels = labels[:-1]
    
    return labels, values


# Make a Hierarchy
def make_hierarchy(cum, root, n_levels=1, **kwargs):
    level_labels, level_values, level_colors = {}, {}, {}
    main_values = {}
    
    total = cum[root]
    
    # Making colors
    num_first_generation = len(find_child_keys(cum.keys(), root=root))
    h_vec = np.linspace(0.,1., num_first_generation+1)[:-1] # 
    v_vec = np.linspace(.6,1., n_levels)                    # Based on # of levels
    
    hsv0 = [0.0,1.0,0.4]
    
    level_labels[0] = [root]
    level_values[0] = [1.0]
    level_colors[0] = np.concatenate((matplotlib.colors.hsv_to_rgb(hsv0),[1.]))
    main_values[0] = [total]
    
    # 1) Make a level based on the previous level.
    # 2) Normalize it so it fits under its root.
    # 3) Create the color spectrum
    
    for level in range(1, n_levels+1):
        level_labels[level] = []
        level_values[level] = []
        level_colors[level] = []
        main_values[level]  = []
        
        prev_labels = level_labels[level-1]
        prev_values = main_values[level-1]
        
        for i,l in enumerate(prev_labels):
            labs, vals = retrieve_children(cum, root=l, root_value=prev_values[i], **kwargs)
            level_labels[level] += labs
            main_values[level]  += vals
            # print(np.array(vals))
            # print(prev_values[i])
            level_values[level] += list(np.array(vals) / sum(vals) * prev_values[i])
            
            # Making the color
            if level == 1:
                # hsv_vec = [np.array([h_vec[j], 1., v_vec[level-1]]) for j in range(len(labs))]
                level_colors[level] = [color_pool[j](0.7) for j in range(len(labs))]
            else:
                # Retrieve hue from the very first generation.
                # Based on l.
                # Go back number of levels - 1
                dirname = l
                for level_up in range(level - 2):
                    dirname = os.path.dirname(dirname)
                
                # key = os.path.sep + l.split(os.path.sep)[1:2][0]
                k = level_labels[1].index(dirname)
                h = matplotlib.colors.rgb_to_hsv(level_colors[1][k][:-1])[0]
                
                s_vec = np.linspace(1.,.4, len(labs)) # In each row
                hsv_vec = [np.array([h, s_vec[j], v_vec[level-1]]) for j in range(len(labs))]
            
                # Convert the hsv_vec to rgba    .
                level_colors[level] += [np.concatenate((matplotlib.colors.hsv_to_rgb(hsv_vec[j]),[1.]))
                                        for j in range(len(hsv_vec))]
    
    return level_labels, level_values, level_colors


def whiten_hierarchy(level_labels, level_values, level_colors):
    for level in level_labels:
        for i in range(len(level_labels[level])):
            if os.path.split(level_labels[level][i])[1] == "other":
                level_colors[level][i] = np.array([1.,1.,1.,1.])
                level_labels[level][i] = "_nolegend_"
    return level_labels, level_values, level_colors

            


#################################
### The Profile Plotter Class ###
#################################
class ProPlot (BasePlotter):
    def plot(self, keyx=None, keyy="/"):
        """
        This function plots the reward function and saves the `.ext` and `.pkl` files.
        
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
        # assert len(keys) == 1, "We only support a single key at the moment."
        # root = keys[0]
        # root = self.options.get("root", "/")
        
        root = keyy
        

        fig, ax = self.init_plot()

        if len(self.loaders) > 1:
            raise ValueError("We do not support multiple loaders for profiler plots.")
        loader = self.loaders[0]
        if len(loader) > 1:
            raise ValueError("We do not support multiple subloaders for profiler plots.")
        subloader = self.loaders[0][0]
        
        
        # Preparing data for plotting
        prolog = subloader.getPrologLoader()
        cum, num, avg = aggregate_prolog(prolog)

        if not root in cum:
            raise ValueError("The root:'{}' does not exist in the logged values.".format(root))

        overall_time = cum[root]
        frame_number = num[root]
        average_time = avg[root]

        ## Print some statistics
        # print("*"*41)
        # print("* Time:                  {:6.1f} hours   *".format(overall_time/3600.))
        # print("* Frames:                {:6.1f} million *".format(frame_number/1.e6))
        # print("* Time for 1000x frames: {:6.1f} seconds *".format(1000.*average_time))
        # print("*"*41)

        
        # level_labels, level_values, level_colors = make_hierarchy(cum, "/", overall_time, 3, collapse=True, collapse_threshold=.05)
        level_labels, level_values, level_colors = make_hierarchy(cum, root=root,
                                                                  n_levels=self.options.get("n_levels", 3),
                                                                  collapse=self.options.get("collapse", True),
                                                                  collapse_threshold=self.options.get("collapse_threshold", .1))

        # print("Hierarchy is comprised from:", level_labels)

        # Whiten the other labels and colors
        level_labels, level_values, level_colors = whiten_hierarchy(level_labels, level_values, level_colors)

        bbox_extra_artists = self._plot_pie_chart(fig, ax, level_labels, level_values, level_colors)
        ax.set_title("Time: {:4.2f} hours".format(overall_time/3600.))

        self.close_plot(fig, ax, path=self.output_dir, name="profiler_"+get_os_friendly_name(root), bbox_extra_artists=bbox_extra_artists, save_pickle=False)

    def _plot_pie_chart(self, fig, ax, level_labels, level_values, level_colors):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.3, box.height])
        legends = []

        for level in sorted(list(set(level_values.keys())-{0})):
            patches, texts = ax.pie(level_values[level], colors=level_colors[level],
                                    radius=(2+level)*0.3,
                                    # radius=(5-level)*0.3, # Reverse
                                    # labeldistance=0.7,
                                    startangle=self.options.get("startangle", 0),
                                    counterclock=self.options.get("counterclock", True),
                                    wedgeprops=dict(width=0.3, edgecolor='w'))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                legends += [ax.legend(patches, level_labels[level], loc='center left', bbox_to_anchor=(1+(level-1)*self.options.get("spacing", 0.4), 0.5))]

        # Set aspect ratio to be equal so that pie is drawn as a circle.
        ax.axis('equal')
        plt.tight_layout()

        for l in legends:
            ax.add_artist(l)
        
        bbox_extra_artists = legends
        return bbox_extra_artists
