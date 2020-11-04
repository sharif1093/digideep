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


#############################################################################################
# def human_format(num):
#     magnitude = 0
#     while abs(num) >= 1000:
#         magnitude += 1
#         num /= 1000.0
#     # add more suffixes if you need them
#     return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

# print('the answer is %s' % human_format(7436313))

################################
# kilo  (k) 10^3   thousand    #
# mega  (M) 10^6   million     #
# giga  (G) 10^9   billion     #
# tera  (T) 10^12  trillion    #
# peta  (P) 10^15  quadrillion #
# exa   (E) 10^18  quintillion #
# zetta (Z) 10^21  sextillion  #
# yotta (Y) 10^24  septillion  #
################################
# 1 kilobyte (KiB)	1,024 bytes
# 1 megabyte (MiB)	1,048,576 bytes
# 1 gigabyte (GiB)	1,073,741,824 bytes
# 1 terabyte (TiB)	1,099,511,627,776 bytes
# 1 petabyte (PiB)	1,125,899,906,842,624 bytes
################################
# var cutoff = new SortedList<long, string> { 
#        {59, "{3:S}" }, 
#        {60, "{2:M}" },
#        {60*60-1, "{2:M}, {3:S}"},
#        {60*60, "{1:H}"},
#        {24*60*60-1, "{1:H}, {2:M}"},
#        {24*60*60, "{0:D}"},
#        {Int64.MaxValue , "{0:D}, {1:H}"}
#      };
#############################################################################################



class BasePlotter:
    def __init__(self, loaders, output_dir=None, **options):
        self.loaders = loaders
        
        # Get output path
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.loaders[0][0].getPlotsPath
        # Create output_dir:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.options = options

    def plot(self, keyx=None, keyy=None):
        pass
    
    
    def init_plot(self, nrows=1, ncols=1):
        ############################
        ### Overall plot options ###
        ############################
        context = self.options.get("context", "paper") # notebook | paper
        font_scale = self.options.get("font_scale", 2)
        line_width = self.options.get("line_width", 2.5)
        width  = self.options.get("width", 10)
        height = self.options.get("height", 8)

        sns.set_context(context, font_scale=font_scale, rc={"lines.linewidth": line_width})
        # sns.set_context("paper")
        # sns.set(font_scale=1.5)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))  # create figure & 1 axis
        return fig, ax
    
    def close_plot(self, fig, ax, path, name, save_pickle=True, bbox_extra_artists=[]):
        """This function saves the plot in `.pkl` and other formats.
        
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

        ####################################
        ### Save figure to file as it is ###
        ####################################
        ## Make layout tight
        # plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.tight_layout()

        # Extension can be png/svg/pdf
        dpi = self.options.get("dpi", 300)
        ext = self.options.get("ext", 'svg')
        if isinstance(ext, list):
            filename = []
            for e in ext:
                filename += [os.path.join(path, name+"."+e)]
        else:
            filename = [os.path.join(path, name+"."+ext)]
        
        if bbox_extra_artists:
            for f in filename:
                fig.savefig(f, dpi=dpi, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
        else:
            for f in filename:
                fig.savefig(f, dpi=dpi)
        print("Saved plot:", filename)

        # We do not use bbox_inches to make all figure sizes consistent.
        # fig.savefig(filename, bbox_inches='tight')
        # fig.savefig(filename, bbox_inches='tight', dpi=300)

        if save_pickle:
            # Save pickle files. They will be very easy to modify later.
            pkl_file = os.path.join(path, name+".pkl")
            pickle.dump(ax, open(pkl_file,'wb'))
            print("Saved file:", pkl_file)

        # Close the figure we have created.
        plt.close(fig)
        print()


# get_os_name(key)
