from PIL import ImageGrab
import cv2
import os
from datetime import datetime
from typing import Optional, Dict, Any
from logging_config import logger
from clients import ai_clients
from clipboard_vision import vision_prompt
from cleanup import cleanup_manager

def generate_unique_filename(prefix: str, extension: str) -> str:
    """
    Generate a unique filename based on current timestamp.

    Args:
        prefix (str): Prefix for the filename.
        extension (str): File extension.

    Returns:
        str: A unique filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def take_screenshot(nlu_result: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Capture a screenshot of the current screen.

    Args:
        nlu_result (Optional[Dict[str, Any]]): NLU result, not used but included for consistency.

    Returns:
        Optional[str]: Path to the saved screenshot, or None if an error occurs.
    """
    screenshot_path = generate_unique_filename("screenshot", "jpg")
    cleanup_manager.add_temp_file(screenshot_path)  # Register for cleanup
    try:
        path = generate_unique_filename("screenshot", "jpg")
        screenshot = ImageGrab.grab()
        rgb_screenshot = screenshot.convert('RGB')
        rgb_screenshot.save(path, quality=15)
        logger.info(f'Screenshot saved to {path}')
        return path
    except Exception as e:
        logger.error(f'Error taking screenshot: {e}')
        return None

def analyze_screenshot(screenshot_path: str) -> str:
    """
    Analyze the contents of a screenshot using vision_prompt.

    Args:
        screenshot_path (str): Path to the screenshot image.

    Returns:
        str: Description of the screenshot contents.
    """
    try:
        prompt = "Describe what you see in this screenshot."
        return vision_prompt(prompt, screenshot_path)
    except Exception as e:
        logger.error(f"Error analyzing screenshot: {e}")
        return "I'm sorry, I couldn't analyze the image due to an error."

def web_cam_capture() -> Optional[str]:
    """
    Capture an image from the webcam.

    Returns:
        Optional[str]: Path to the saved webcam image, or None if an error occurs.
    """
    try:
        if ai_clients.web_cam is None:
            logger.warning('No webcam available')
            return None

        if not ai_clients.web_cam.isOpened():
            logger.error('Error: Camera not opened successfully')
            return None

        path = generate_unique_filename("webcam", "jpg")
        ret, frame = ai_clients.web_cam.read()
        if ret:
            cv2.imwrite(path, frame)
            logger.info(f'Webcam capture saved to {path}')
            return path
        else:
            logger.error('Error: Failed to capture image from webcam')
            return None
    except Exception as e:
        logger.error(f"Error in web_cam_capture: {e}")
        return None

def analyze_image(image_path: str) -> str:
    return vision_prompt("Describe what you see in this image.", image_path)

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that the specified directory exists, creating it if necessary.

    Args:
        directory (str): The directory path to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def set_output_directory(directory: str) -> None:
    """
    Set the output directory for screenshots and webcam captures.

    Args:
        directory (str): The directory path to use for output files.
    """
    global OUTPUT_DIRECTORY
    OUTPUT_DIRECTORY = directory
    ensure_directory_exists(OUTPUT_DIRECTORY)
    logger.info(f"Set output directory to: {OUTPUT_DIRECTORY}")

# Set a default output directory
OUTPUT_DIRECTORY = "captures"
set_output_directory(OUTPUT_DIRECTORY)