from datetime import datetime, timedelta
from threading import Lock
import time

class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.timestamps = []
        self._lock = Lock()

    def wait(self):
        with self._lock:
            now = datetime.now()
            # Remove timestamps outside the window
            self.timestamps = [ts for ts in self.timestamps 
                             if now - ts < timedelta(seconds=self.period)]
            
            if len(self.timestamps) >= self.calls:
                sleep_time = (self.timestamps[0] + 
                            timedelta(seconds=self.period) - now).total_seconds()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            self.timestamps.append(now)