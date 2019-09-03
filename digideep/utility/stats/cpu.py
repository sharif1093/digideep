"""This module is inspired by https://github.com/anderskm/gputil/blob/master/GPUtil/GPUtil.py.

This module provides tools to monitor the CPU/Memory utilization.
"""

#####################
##### CPU STATS #####
#####################

import psutil, os, re
import subprocess
import numpy as np

def available_cpu_list():
    """ List of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""
    
    def expand(s):
        if s.count("-") == 1:
            numbers = re.findall(r'(\d+)', s)
            start = int(numbers[0])
            end   = int(numbers[1])
            return list(range(start, end+1))
            
        elif s.count("-") == 0:
            return [int(s)]
        else:
            print("The string cannot have more than one dash mark (-).")

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed_list:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            group = m.group(1)
            # group="0-7,9-10, 14"
            
            m = re.findall(r'(\d+(-\d+)?)', group)
            items = [item[0] for item in m]
            
            cpus = []
            for item in items:
                cpus += expand(item)
            
            return cpus
    except IOError:
        raise IOError("Could not read /proc/self/status")

# This function obtained from: https://stackoverflow.com/a/1006301
# This function only gives the number of available CPUs.
def get_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # #Check nproc. I have found it respecting the visible CPUs in SLURM:
    # try:
    #     m = subprocess.run(['nproc'], stdout=subprocess.PIPE)
    #     if m:
    #         res = int(m.stdout.decode('ascii').replace("\n", ""))
    #         if res > 0:
    #             return res
    # except:
    #     pass
    

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')






MEMORY_IN_MB = 1048576.
def get_cpu_stats():
    # TODO: Only report CPUs that are allowed to be used by current process.
    #       Also overall CPU usage must be an avergae over available CPUs.
    #       The index of available CPUs can be obtained by: 
    process = psutil.Process(os.getpid())
    total_mem = psutil.virtual_memory().total / MEMORY_IN_MB
    used_mem  = psutil.virtual_memory().used  / MEMORY_IN_MB
    mem       = process.memory_info().rss     / MEMORY_IN_MB
    
    cpu_count = get_cpu_count()
    total_per_cpu_raw = psutil.cpu_percent(percpu=True, interval=0.0)
    cpu_list = available_cpu_list()

    total_per_cpu = [total_per_cpu_raw[i] for i in cpu_list]
    total_cpu = np.mean(total_per_cpu)

    total_cpu = psutil.cpu_percent(percpu=False, interval=0.0)
    
    # psutil.disk_usage('/').percent

    # TODO: This cpu usage statistics seems not working at all.
    # cpu       = process.cpu_percent() / psutil.cpu_count()
    # cpu=cpu,
    res = dict(total_cpu=total_cpu, total_per_cpu=total_per_cpu, cpu_count=cpu_count,
               total_mem=total_mem, used_mem=used_mem, mem=mem)
    return res
