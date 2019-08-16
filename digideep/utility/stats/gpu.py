"""This module is inspired by https://github.com/anderskm/gputil/blob/master/GPUtil/GPUtil.py.

This module provides tools to monitor the GPU utilization using Visdom.
"""

from distutils import spawn
from subprocess import Popen, PIPE
import os, platform
#####################
##### GPU STATS #####
#####################
# GPU = collections.namedtuple("GPU", "id uuid load memoryTotal memoryUsed memoryFree driver gpu_name serial display_active display_mode temp_gpu")
def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number

def get_gpu_lines():
    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi 
        # could not be found from the environment path, 
        # try to find it from system drive with default installation path
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
    else:
        nvidia_smi = "nvidia-smi"
	
    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen([nvidia_smi,"--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu", "--format=csv,noheader,nounits"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except Exception as ex:
        import warnings
        warnings.warn(str(ex))
        return []
    output = stdout.decode('UTF-8')
    # output = output[2:-1] # Remove b' and ' from string added by python
    #print(output)
    ## Parse output
    # Split on line break
    lines = output.split(os.linesep)
    return lines

def get_gpu_count():
    lines = get_gpu_lines()
    numDevices = len(lines)-1
    return numDevices

def get_gpu_stats():
    lines = get_gpu_lines()
    numDevices = len(lines)-1

    # TODO: We don't need all information for statistics. Only those that may change ...
    gpus = dict(id=[], uuid=[], load=[],
                memoryTotal=[], memoryUsed=[], memoryFree=[],
                driver=[], gpu_name=[], serial=[], display_active=[], display_mode=[], temp_gpu=[])

    for i in range(numDevices):
        line = lines[i]
        #print(line)
        vals = line.split(', ')
        #print(vals)
        gpus["id"].append(int(vals[0]))
        gpus["uuid"].append(vals[1])
        gpus["load"].append(safeFloatCast(vals[2]))
        gpus["memoryTotal"].append(safeFloatCast(vals[3]))
        gpus["memoryUsed"].append(safeFloatCast(vals[4]))
        gpus["memoryFree"].append(safeFloatCast(vals[5]))
        gpus["driver"].append(vals[6])
        gpus["gpu_name"].append(vals[7])
        gpus["serial"].append(vals[8])
        gpus["display_active"].append(vals[9])
        gpus["display_mode"].append(vals[10])
        gpus["temp_gpu"].append(safeFloatCast(vals[11]))

    return gpus