from typing import Dict, Any, Optional, List
import time
import threading
from collections import deque
import psutil
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from logging_config import logger

@dataclass
class APIMetrics:
    total_calls: int = 0
    total_time: float = 0.0
    errors: int = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def average_latency(self) -> float:
        return self.total_time / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return (self.errors / self.total_calls * 100) if self.total_calls > 0 else 0.0

class PerformanceMonitor:
    def __init__(self, metrics_retention_days: int = 7):
        self.metrics_retention = timedelta(days=metrics_retention_days)
        self._lock = threading.RLock()
        self.api_metrics: Dict[str, APIMetrics] = {}
        self.memory_usage: deque = deque(maxlen=1000)
        self.cpu_usage: deque = deque(maxlen=1000)
        self.operation_times: Dict[str, deque] = {}
        self.error_counts: Dict[str, int] = {}
        self._start_monitoring()

    def _start_monitoring(self):
        """Start background monitoring thread."""
        def monitor_resources():
            while True:
                try:
                    self.record_system_metrics()
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")

        thread = threading.Thread(target=monitor_resources, daemon=True)
        thread.start()

    def record_system_metrics(self):
        """Record system resource usage."""
        try:
            process = psutil.Process()
            with self._lock:
                self.memory_usage.append({
                    'timestamp': time.time(),
                    'memory_percent': process.memory_percent(),
                    'memory_mb': process.memory_info().rss / 1024 / 1024
                })
                self.cpu_usage.append({
                    'timestamp': time.time(),
                    'cpu_percent': process.cpu_percent()
                })
        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")

    def record_api_call(self, api_name: str, latency: float, error: bool = False):
        """Record API call metrics."""
        with self._lock:
            if api_name not in self.api_metrics:
                self.api_metrics[api_name] = APIMetrics()
            
            metrics = self.api_metrics[api_name]
            metrics.total_calls += 1
            metrics.total_time += latency
            metrics.latencies.append(latency)
            if error:
                metrics.errors += 1

    def record_operation(self, operation_name: str, duration: float):
        """Record operation timing."""
        with self._lock:
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = deque(maxlen=1000)
            self.operation_times[operation_name].append({
                'timestamp': time.time(),
                'duration': duration
            })

    def record_error(self, error_type: str):
        """Record error occurrence."""
        with self._lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def get_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        with self._lock:
            current_memory = self.memory_usage[-1] if self.memory_usage else None
            current_cpu = self.cpu_usage[-1] if self.cpu_usage else None
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': {
                    'current_memory_usage': current_memory,
                    'current_cpu_usage': current_cpu,
                    'memory_trend': list(self.memory_usage)[-10:],  # Last 10 readings
                    'cpu_trend': list(self.cpu_usage)[-10:]
                },
                'api_metrics': {
                    name: {
                        'total_calls': metrics.total_calls,
                        'average_latency': metrics.average_latency,
                        'error_rate': metrics.error_rate,
                        'recent_latencies': list(metrics.latencies)[-5:]
                    }
                    for name, metrics in self.api_metrics.items()
                },
                'operation_metrics': {
                    name: {
                        'average_duration': sum(d['duration'] for d in times)/len(times) if times else 0,
                        'recent_operations': list(times)[-5:]
                    }
                    for name, times in self.operation_times.items()
                },
                'error_metrics': self.error_counts
            }

    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.api_metrics.clear()
            self.memory_usage.clear()
            self.cpu_usage.clear()
            self.operation_times.clear()
            self.error_counts.clear()

    def log_metrics(self):
        """Log current metrics."""
        try:
            metrics = self.get_metrics_report()
            logger.info(f"Performance Metrics Report:\n{json.dumps(metrics, indent=2)}")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

# Create global instance
performance_monitor = PerformanceMonitor()

# Decorator for monitoring function performance
def monitor_performance(operation_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_operation(operation_name, duration)
                return result
            except Exception as e:
                performance_monitor.record_error(type(e).__name__)
                raise
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Test monitoring
    @monitor_performance("test_operation")
    def test_function():
        time.sleep(1)
        return "test complete"

    # Run test
    test_function()
    
    # Record API call
    performance_monitor.record_api_call("test_api", 0.5)
    
    # Get and print metrics
    metrics = performance_monitor.get_metrics_report()
    print(json.dumps(metrics, indent=2))