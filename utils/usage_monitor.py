import os
from threading import Thread, Lock
import psutil
import time


class UsageMonitor:

    def __init__(self):
        self._closed = False
        self._active = False
        self._process = psutil.Process(os.getpid())
        self._memory = -1
        self._cpu_usage = -1
        self._lock = Lock()
        self._thread = Thread(
            target=self._track_memory,
            name="MemoryTracker",
        )
        self._thread.start()
        print(f"UsageMonitor pid={os.getpid()}")


    def start(self):
        with self._lock:
            if self._active:
                raise RuntimeError("UsageMonitor is already active.")
            self._active = True
            self._cpu_usage = time.process_time_ns()


    def stop(self):
        with self._lock:
            if not self._active:
                raise RuntimeError("UsageMonitor is inactive.")
            self._active = False

            memory = self._memory
            cpu_usage = time.process_time_ns() - self._cpu_usage

            self._memory = -1
            self._cpu_usage = -1

            return cpu_usage, memory


    def close(self, wait=False):
        with self._lock:
            if self._closed:
                return
            self._closed = True

        if wait:
            self._thread.join()

    
    def _track_memory(self):
        while True:
            with self._lock:
                if self._closed:
                    return
                if self._active:
                    self._memory = max(self._process.memory_info().rss, self._memory)

            time.sleep(0.00025)
