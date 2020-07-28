"""
This module provides tools to monitor the CPU/GPU/Memory utilization using Monitoring.
"""

from digideep.utility.timer import Timer
from digideep.utility.monitoring import Monitor
from .cpu import get_cpu_count, get_cpu_stats
from .gpu import get_gpu_count, get_gpu_stats, get_gpu_lines
import os
# We don't want to use the general monitor.
###################
## CREATE TIMERS ##
###################

class StatLogger():
    def __init__(self, monitor_cpu=True, monitor_gpu=True, output="/tmp/monitor.log", interval=2.0):
        self.output = output
        self.monitor = Monitor()
        self.monitor.set_output_file(self.output)

        self.monitor_cpu = monitor_cpu
        self.monitor_gpu = monitor_gpu

        if not "CUDA_VISIBLE_DEVICES" in os.environ:
            self.monitor_gpu = False

        if self.monitor_cpu:
            self.cpu_count = get_cpu_count()
        if self.monitor_gpu:
            self.gpu_count = get_gpu_count()

        self.interval = interval # seconds
        self.timer = Timer(self._updater, self.interval)

    # Logger control methods
    def start(self):
        self.timer.start()
    def shutdown(self):
        self.timer.shutdown()

    def _updater(self):
        self.monitor.reset()

        if self.monitor_cpu:
            # Get CPU usage (ALL/EACH)
            # Get CPU memory
            cpus = get_cpu_stats()

            self.monitor("/cpu/per", cpus["total_per_cpu"]) # Percent
            self.monitor("/cpu/all/total", cpus["total_cpu"]) # Percent
            # self.monitor("/cpu/all/cpu", cpus["cpu"])

            self.monitor("/cpu/memory/total", cpus["total_mem"]) # MB
            self.monitor("/cpu/memory/used", cpus["used_mem"]) # MB
            self.monitor("/cpu/memory/mem", cpus["mem"])
            
            
        if self.monitor_gpu:
            # Get GPU usage (EACH)
            # Get GPU total memory (EACH)
            # Get GPU memory (EACH)

            gpus = get_gpu_stats()
            try:
                self.monitor("/gpu/load", gpus["load"]) # Percent
                self.monitor("/gpu/memory/total", gpus["memoryTotal"]) # MB
                self.monitor("/gpu/memory/used",  gpus["memoryUsed"]) # MB
            except:
                # import warnings
                # warnings.warn("There was a problematic value for gpus!")
                pass
        
        self.monitor.dump()
        # print(self.monitor)


if __name__=="__main__":
    st = StatLogger()
    st.start()
    while(1):
        # This infinite loop causes one CPU to work full-load.
        continue
    

