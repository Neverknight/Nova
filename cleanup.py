from typing import Callable
import cv2
import os
from logging_config import logger
from base_cleanup import base_cleanup_manager

class CleanupManager(base_cleanup_manager.__class__):
    def __init__(self):
        super().__init__()
        self._cleanup_functions.extend([
            self.release_webcam,
            self.destroy_cv2_windows,
            self.reset_ai_clients
        ])

    def release_webcam(self) -> None:
        """Release the webcam if it's open."""
        try:
            from clients import ai_clients
            if ai_clients.web_cam and ai_clients.web_cam.isOpened():
                ai_clients.web_cam.release()
                ai_clients.web_cam = None
                logger.info("Webcam released.")
        except Exception as e:
            logger.error(f"Error releasing webcam: {e}")

    def destroy_cv2_windows(self) -> None:
        """Destroy all OpenCV windows."""
        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows destroyed.")
        except Exception as e:
            logger.error(f"Error destroying OpenCV windows: {e}")

    def reset_ai_clients(self) -> None:
        """Reset AI clients if necessary."""
        try:
            from clients import ai_clients
            if hasattr(ai_clients, 'reset_conversation'):
                ai_clients.reset_conversation()
            logger.info("AI clients reset.")
        except Exception as e:
            logger.error(f"Error resetting AI clients: {e}")

    def clean_up(self) -> None:
        """Main cleanup function that runs all cleanup tasks."""
        try:
            # Run all registered cleanup functions
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

# Create global instance
cleanup_manager = CleanupManager()

# Export the clean_up function
def clean_up():
    """Global cleanup function."""
    cleanup_manager.clean_up()

# Export needed symbols
__all__ = ['cleanup_manager', 'clean_up']