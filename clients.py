import os
from typing import List, Dict, Union, Optional
import openai
from faster_whisper import WhisperModel
import cv2
from logging_config import logger
from config import get_config
from ai_interface import ai_interface
from capability_analyzer import update_system_message

# Get configuration
config = get_config()

# System Message for GPT-4
try:
    SYS_MSG_GPT4: str = update_system_message()
except Exception as e:
    logger.error(f"Error generating system message: {e}")
    SYS_MSG_GPT4 = f"You are {config.ASSISTANT_NAME}, a helpful AI assistant that can engage in natural conversation and help with various tasks."

# Model configurations
GENERATION_CONFIG_GPT4: Dict[str, Union[float, int]] = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

SAFETY_SETTINGS_GPT4: List[Dict[str, str]] = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
]

class AIClients:
    def __init__(self):
        self.whisper_model = self._init_whisper()
        self.web_cam = None  # Initialize as None first
        try:
            self.web_cam = self._init_webcam()
        except Exception as e:
            logger.warning(f"Webcam initialization failed: {e}. Webcam features will be disabled.")

    @staticmethod
    def _init_whisper() -> WhisperModel:
        try:
            num_cores = os.cpu_count()
            return WhisperModel('base', device='cpu', compute_type='int8', 
                              cpu_threads=num_cores // 2, num_workers=num_cores // 2)
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise

    def _init_webcam(self) -> Optional[cv2.VideoCapture]:
        try:
            for index in range(2):
                cam = cv2.VideoCapture(index)
                if cam.isOpened():
                    logger.info(f"Successfully initialized webcam with index {index}")
                    return cam
            logger.warning("No working webcam found")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize webcam: {e}")
            return None

    def call_gpt4(self, *args, **kwargs) -> str:
        """Delegate to AI interface."""
        return ai_interface.call_gpt4(*args, **kwargs)

    def reset_conversation(self) -> None:
        """Delegate to AI interface."""
        ai_interface.reset_conversation()

ai_clients = AIClients()

__all__ = ['ai_clients', 'SYS_MSG_GPT4', 'GENERATION_CONFIG_GPT4', 'SAFETY_SETTINGS_GPT4']