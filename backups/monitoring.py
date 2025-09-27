from typing import Dict, Any
import time
from functools import wraps
import psutil
import threading
from collections import deque

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'function_calls': {},
            'api_latency': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'errors': {}
        }
        self._lock = threading.Lock()
        self._start_monitoring()

    def _start_monitoring(self):
        def monitor():
            while True:
                self.record_system_metrics()
                time.sleep(60)  # Update every minute
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def record_system_metrics(self):
        with self._lock:
            process = psutil.Process()
            self.metrics['memory_usage'].append({
                'timestamp': time.time(),
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent()
            })

    def record_api_latency(self, api_name: str, latency: float):
        with self._lock:
            self.metrics['api_latency'].append({
                'timestamp': time.time(),
                'api': api_name,
                'latency': latency
            })

    def record_error(self, error_type: str):
        with self._lock:
            self.metrics['errors'][error_type] = self.metrics['errors'].get(error_type, 0) + 1

performance_monitor = PerformanceMonitor()

def monitor_performance(func_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                with performance_monitor._lock:
                    if func_name not in performance_monitor.metrics['function_calls']:
                        performance_monitor.metrics['function_calls'][func_name] = {
                            'count': 0,
                            'total_time': 0,
                            'average_time': 0
                        }
                    metrics = performance_monitor.metrics['function_calls'][func_name]
                    metrics['count'] += 1
                    metrics['total_time'] += duration
                    metrics['average_time'] = metrics['total_time'] / metrics['count']
                return result
            except Exception as e:
                performance_monitor.record_error(type(e).__name__)
                raise
        return wrapper
    return decorator