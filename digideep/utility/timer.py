#####################
####### TIMER #######
#####################

import threading


class Timer(threading.Thread):
    """Thread that executes a task every N seconds"""
    
    def __init__(self, task, interval=1.0):
        # threading.Thread.__init__(self)
        self._task = task
        self._finished = threading.Event()
        self._interval = interval
        # self._counter = 0
        threading.Thread.__init__(self, name="PeriodicExecutor")
        self.setDaemon(1)
    
    def setInterval(self, interval):
        """Set the number of seconds we sleep between executing our task"""
        self._interval = interval
    
    def shutdown(self):
        """Stop this thread"""
        self._finished.set()
    
    def run(self):
        while 1:
            if self._finished.isSet(): return
            self.task_exec()
            
            # sleep for interval or until shutdown
            self._finished.wait(self._interval)
    
    def task_exec(self):
        """The task done by this thread - override in subclasses"""
        self._task()
        # print("[", self._counter, "]:")
        # self._counter += 1
        