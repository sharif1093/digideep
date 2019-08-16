"""This module is inspired by https://github.com/anderskm/gputil/blob/master/GPUtil/GPUtil.py.

This module provides tools to monitor the CPU/Memory utilization using Visdom.
"""

#####################
##### CPU STATS #####
#####################

import psutil, os

def get_cpu_count():
    psutil.cpu_count()

MEMORY_IN_MB = 1048576.
def get_cpu_stats():
    process = psutil.Process(os.getpid())
    total_mem = psutil.virtual_memory().total / MEMORY_IN_MB
    used_mem  = psutil.virtual_memory().used  / MEMORY_IN_MB
    mem       = process.memory_info().rss     / MEMORY_IN_MB
    
    cpu_count = psutil.cpu_count()
    total_per_cpu = psutil.cpu_percent(percpu=True, interval=0.0)
    total_cpu = psutil.cpu_percent(percpu=False, interval=0.0)
    
    # total_per_cpu = psutil.cpu_percent(percpu=True, interval=1.0)
    # total_cpu = psutil.cpu_percent(percpu=False, interval=1.0)

    # TODO: This cpu usage statistics seems not working at all.
    cpu       = process.cpu_percent()/ cpu_count
    
    # psutil.disk_usage('/').percent
    
    res = dict(total_cpu=total_cpu, total_per_cpu=total_per_cpu, cpu=cpu, cpu_count=cpu_count,
               total_mem=total_mem, used_mem=used_mem, mem=mem)
    return res
