"""This module is inspired by https://github.com/anderskm/gputil/blob/master/GPUtil/GPUtil.py.

This module provides tools to monitor the CPU/GPU/Memory utilization using Visdom.
"""

from subprocess import Popen, PIPE
from distutils import spawn
import os
import sys
import platform
# import collections

from digideep.utility.timer import Timer


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

#####################
##### CPU STATS #####
#####################

import psutil, os

MEMORY_IN_MB = 1048576.
def get_cpu_stats():
    process = psutil.Process(os.getpid())
    total_mem = psutil.virtual_memory().total / MEMORY_IN_MB
    used_mem  = psutil.virtual_memory().used  / MEMORY_IN_MB
    mem       = process.memory_info().rss     / MEMORY_IN_MB
    
    cpu_count = psutil.cpu_count()
    total_per_cpu = psutil.cpu_percent(percpu=True, interval=0.0)
    total_cpu = psutil.cpu_percent(percpu=False, interval=0.0)
    # TODO: This cpu usage statistics seems not working at all.
    cpu       = process.cpu_percent()/ cpu_count
    
    # psutil.disk_usage('/').percent
    
    res = dict(total_cpu=total_cpu, total_per_cpu=total_per_cpu, cpu=cpu, cpu_count=cpu_count,
               total_mem=total_mem, used_mem=used_mem, mem=mem)
    return res


###################
## CREATE TIMERS ##
###################
from digideep.utility.plotting import Plotter
# from collections import OrderedDict as od

class StatVizdom():
    def __init__(self, monitor_cpu=True, monitor_gpu=True, percpu=False):
        self.monitor_cpu = monitor_cpu
        self.monitor_gpu = monitor_gpu
        self.percpu = percpu

        if monitor_cpu:
            cpu_count = psutil.cpu_count()
            self.cpu_per_plotter = Plotter(name={"Load_"+str(i):["Main"] for i in range(cpu_count)},
                                           env="monitoring",
                                           win="CPU_Per_Window",
                                           filterargs=dict(window_size=10),
                                           visdomargs=dict(opts=dict(title='CPU Utilization', xlabel='Wall Time', ylabel='Percent')))
            self.cpu_all_plotter = Plotter(name={"Total":["Mean"], "Process":["Mean"]},
                                           env="monitoring",
                                           win="CPU_All_Window",
                                           filterargs=dict(window_size=10),
                                           visdomargs=dict(opts=dict(title='CPU Utilization', xlabel='Wall Time', ylabel='Percent')))
            self.cpu_mem_plotter = Plotter(name={"Total":["Mean"], "Used":["Mean"], "Process":["Mean"]},
                                           env="monitoring",
                                           win="CPU_Mem_Window",
                                           filterargs=dict(window_size=10),
                                           visdomargs=dict(opts=dict(title='CPU Memory Usage', xlabel='Wall Time', ylabel='MB')))
            
            self.timer_cpu = Timer(self.update_cpu, 0.5)
        
        if monitor_gpu:
            gpu_count = get_gpu_count()
            self.gpu_per_plotter = Plotter(name={"Load:"+str(i):["Main"] for i in range(gpu_count)},
                                           env="monitoring",
                                           win="GPU_Per_Window",
                                           filterargs=dict(window_size=10),
                                           visdomargs=dict(opts=dict(title='GPU Utilization', xlabel='Wall Time', ylabel='Percent')))
            
            
            self.gpu_mem_plotter_total = Plotter(name={"Total:"+str(i):["Mean"] for i in range(gpu_count)},
                                                 env="monitoring",
                                                 win="GPU_Mem_Window",
                                                 filterargs=dict(window_size=10),
                                                 visdomargs=dict(opts=dict(title='GPU Memory Usage', xlabel='Wall Time', ylabel='MB')))
            
            self.gpu_mem_plotter_used  = Plotter(name={"Used:"+str(i):["Mean"] for i in range(gpu_count)},
                                                 env="monitoring",
                                                 win="GPU_Mem_Window",
                                                 filterargs=dict(window_size=10))
            self.timer_gpu = Timer(self.update_gpu, 1.13)

        
    def start(self):
        if self.monitor_cpu:
            self.timer_cpu.start()
        if self.monitor_gpu:
            self.timer_gpu.start()
    def shutdown(self):
        if self.monitor_cpu:
            self.timer_cpu.shutdown()
        if self.monitor_gpu:
            self.timer_gpu.shutdown()
    def update_cpu(self):
        res = get_cpu_stats()
        if self.percpu:
            self.cpu_per_plotter.append(res["total_per_cpu"])
        self.cpu_all_plotter.append([res["total_cpu"], res["cpu"]])
        self.cpu_mem_plotter.append([res["total_mem"], res["used_mem"], res["mem"]])

    def update_gpu(self):
        gpus = get_gpu_stats()
        
        try:
            self.gpu_per_plotter.append(gpus["load"])
            self.gpu_mem_plotter_total.append(gpus["memoryTotal"])
            self.gpu_mem_plotter_used.append(gpus["memoryUsed"])
        except:
            # import warnings
            # warnings.warn("There was a problematic value for gpus!")
            pass

if __name__=="__main__":
    # Implement a Unit Test
    pass

