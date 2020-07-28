import os
import pickle
import numpy as np


## Plotting modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def get_os_friendly_name(path):
    path = path.lstrip("/")
    path = path.replace("/", "_")
    return path

class BasePlotter:
    def __init__(self, loaders, output_dir=None, **options):
        self.loaders = loaders
        
        # Get output path
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.loaders[0][0].getPlotsPath

        self.options = options

    def plot(self):
        pass
    
    
    def init_plot(self):
        ############################
        ### Overall plot options ###
        ############################
        context = self.options.get("context", "notebook") # notebook | paper
        font_scale = self.options.get("font_scale", 2)
        line_width = self.options.get("line_width", 2.5)
        width  = self.options.get("width", 10)
        height = self.options.get("height", 8)

        sns.set_context(context, font_scale=font_scale, rc={"lines.linewidth": line_width})
        # sns.set_context("paper")
        # sns.set(font_scale=1.5)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))  # create figure & 1 axis
        return fig, ax
    
    def close_plot(self, fig, ax, path, name, save_pickle=True, bbox_extra_artists=[]):
        ####################################
        ### Save figure to file as it is ###
        ####################################
        ## Make layout tight
        # plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.tight_layout()

        # Extension can be png/svg/pdf
        filename = os.path.join(path, name+"."+self.options.get("ext", 'svg'))
        
        if bbox_extra_artists:
            fig.savefig(filename, dpi=self.options.get("dpi", 300), bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
        else:
            fig.savefig(filename, dpi=self.options.get("dpi", 300))

        # We do not use bbox_inches to make all figure sizes consistent.
        # fig.savefig(filename, bbox_inches='tight')
        # fig.savefig(filename, bbox_inches='tight', dpi=300)

        if save_pickle:
            # Save pickle files. They will be very easy to modify later.
            pkl_file = os.path.join(path, name+".pkl")
            pickle.dump(ax, open(pkl_file,'wb'))

        # Close the figure we have created.
        plt.close(fig)


# get_os_name(key)
