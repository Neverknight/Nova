# base_cleanup.py
from typing import List, Callable, Set
import threading
from logging_config import logger
import atexit
import os

class BaseCleanupManager:
    def __init__(self):
        self._cleanup_functions: List[Callable[[], None]] = []
        self._temp_files: Set[str] = set()
        self._temp_dirs: Set[str] = set()
        self._lock = threading.Lock()
        
        # Register cleanup on exit
        atexit.register(self.clean_up)

    def add_temp_file(self, filepath: str) -> None:
        """Register a temporary file for cleanup."""
        with self._lock:
            self._temp_files.add(os.path.abspath(filepath))
            
    def add_temp_dir(self, dirpath: str) -> None:
        """Register a temporary directory for cleanup."""
        with self._lock:
            self._temp_dirs.add(os.path.abspath(dirpath))
            
    def add_cleanup_function(self, func: Callable[[], None]) -> None:
        """Add a cleanup function to be executed during cleanup."""
        with self._lock:
            if func not in self._cleanup_functions:
                self._cleanup_functions.append(func)

    def clean_up(self) -> None:
        """Clean up resources before exiting."""
        try:
            # Execute registered cleanup functions
            for func in self._cleanup_functions:
                try:
                    func()
                except Exception as e:
                    logger.error(f"Error in cleanup function {func.__name__}: {e}")
            
            # Clean up temporary files and directories
            self._cleanup_temp_files()
            self._cleanup_temp_dirs()
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        for filepath in self._temp_files.copy():
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Removed temp file: {filepath}")
            except Exception as e:
                logger.error(f"Error removing temp file {filepath}: {e}")
            self._temp_files.remove(filepath)

    def _cleanup_temp_dirs(self) -> None:
        """Clean up temporary directories."""
        for dirpath in self._temp_dirs.copy():
            try:
                if os.path.exists(dirpath):
                    os.rmdir(dirpath)
                    logger.debug(f"Removed temp directory: {dirpath}")
            except Exception as e:
                logger.error(f"Error removing temp directory {dirpath}: {e}")
            self._temp_dirs.remove(dirpath)

# Create global instance
base_cleanup_manager = BaseCleanupManager()