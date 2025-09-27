import logging
import os
import sys
import cv2
import platform
from logging.handlers import RotatingFileHandler

def setup_logging(log_file='nova.log', log_level=logging.INFO):
    logger = logging.getLogger('nova')
    logger.setLevel(log_level)

    # Create formatter that includes more details
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Console handler with less detail
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    console_handler.setLevel(logging.INFO)

    # File handler with full detail
    file_handler = RotatingFileHandler(
        os.path.join('logs', log_file),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log system info at startup
    logger.info("=== Starting Nova ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"Operating System: {platform.platform()}")
    
    return logger

# Create and configure logger
logger = setup_logging()

if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")