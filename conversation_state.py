import threading
from threading import Lock
from typing import Optional
from config import get_config
from logging_config import logger

config = get_config()

class ConversationState:
    def __init__(self):
        self.is_active: bool = False
        self.timer: Optional[threading.Timer] = None
        self._lock = threading.RLock() 
        self._state = threading.local() 

    def reset(self) -> None:
        with self._lock:
            self.is_active = False
            if self.timer is not None:
                self.timer.cancel()
            logger.info("Conversation state reset.")

    def start(self) -> None:
        with self._lock:
            self.is_active = True
            if self.timer is not None:
                self.timer.cancel()
            self.timer = threading.Timer(config.CONVERSATION_TIMEOUT, self.reset)
            self.timer.start()
            logger.info("Conversation state started.")

    def is_active_conversation(self) -> bool:
        """
        Check if the conversation is currently active.

        Returns:
            bool: True if the conversation is active, False otherwise.
        """
        return self.is_active

# Create a global instance of ConversationState
conversation_state = ConversationState()

def reset_conversation_state() -> None:
    """
    Reset the global conversation state.
    """
    conversation_state.reset()

def start_conversation_state() -> None:
    """
    Start or restart the global conversation state.
    """
    conversation_state.start()

def is_active_conversation() -> bool:
    """
    Check if the global conversation is currently active.

    Returns:
        bool: True if the conversation is active, False otherwise.
    """
    return conversation_state.is_active_conversation()