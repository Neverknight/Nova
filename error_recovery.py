from functools import wraps
from typing import Any, Callable, TypeVar, Dict, Optional
import time
import logging
from logging_config import logger

T = TypeVar('T')

class ServiceUnavailableError(Exception):
    pass

class ErrorHandler:
    def __init__(self):
        self.fallback_functions: Dict[str, Callable] = {}
        self.retry_counts: Dict[str, int] = {}
        self.max_retries: Dict[str, int] = {}
        
    def register_fallback(self, function_name: str, fallback_func: Callable) -> None:
        """Register a fallback function for a specific function name."""
        self.fallback_functions[function_name] = fallback_func
        
    def set_max_retries(self, function_name: str, max_retries: int) -> None:
        """Set maximum retry attempts for a specific function."""
        self.max_retries[function_name] = max_retries
        
    def with_fallback(self, function_name: str):
        """Decorator that implements fallback mechanism for functions."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {function_name}: {str(e)}")
                    if function_name in self.fallback_functions:
                        logger.info(f"Attempting fallback for {function_name}")
                        return self.fallback_functions[function_name](*args, **kwargs)
                    raise
            return wrapper
        return decorator
    
    def with_retry(self, 
                   function_name: str,
                   retries: Optional[int] = None,
                   delay: float = 1.0,
                   backoff: float = 2.0,
                   exceptions: tuple = (Exception,)):
        """Decorator that implements retry mechanism with exponential backoff."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                max_attempts = retries or self.max_retries.get(function_name, 3)
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_attempts - 1:  # Last attempt
                            if function_name in self.fallback_functions:
                                logger.warning(f"All retries failed for {function_name}, attempting fallback")
                                return self.fallback_functions[function_name](*args, **kwargs)
                            raise  # Re-raise the last exception if no fallback
                            
                        wait_time = current_delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {function_name}: {str(e)}. "
                            f"Retrying in {wait_time:.1f} seconds..."
                        )
                        time.sleep(wait_time)
                
                raise Exception(f"Unexpected error in retry loop for {function_name}")
            return wrapper
        return decorator

# Create a global instance of ErrorHandler
error_handler = ErrorHandler()

def with_recovery(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Legacy decorator for backward compatibility.
    Recommended to use error_handler.with_retry instead for new code.
    """
    return error_handler.with_retry(
        function_name="legacy_function",
        retries=retries,
        delay=delay,
        backoff=backoff,
        exceptions=exceptions
    )

# Example usage:
if __name__ == "__main__":
    # Example fallback function
    def fallback_weather():
        return {"temperature": 20, "condition": "unknown"}
    
    # Register fallback for weather service
    error_handler.register_fallback("get_weather", fallback_weather)
    
    # Example usage of decorator
    @error_handler.with_fallback("get_weather")
    def get_weather_data():
        raise ServiceUnavailableError("Weather service unavailable")
    
    # Example with retry
    @error_handler.with_retry("get_weather", retries=3, delay=1.0)
    def get_weather_with_retry():
        raise ServiceUnavailableError("Weather service temporarily unavailable")
    
    try:
        # This will use the fallback
        result = get_weather_data()
        print(f"Fallback result: {result}")
        
        # This will retry 3 times then use fallback
        result = get_weather_with_retry()
        print(f"Retry result: {result}")
    except Exception as e:
        print(f"Error: {e}")