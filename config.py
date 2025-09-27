# config.py
import os
from typing import Dict, Any
from dotenv import load_dotenv
import logging
from functools import lru_cache

# Create a logger for this module specifically
logger = logging.getLogger('nova.config')

# Load environment variables from .env file
load_dotenv()

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Assistant identity
        self._ASSISTANT_NAME = os.getenv('ASSISTANT_NAME', 'Nova')
        self._WAKE_WORD = os.getenv('WAKE_WORD', 'nova')
        
        # Set up directory structure
        self._DIRECTORIES = {
            'data': 'data',
            'logs': 'logs',
            'temp': 'temp',
            'captures': 'captures'
        }
        
        # Create directories if they don't exist
        for dir_path in self._DIRECTORIES.values():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"Created directory: {dir_path}")

        # Set up file paths
        self._DB_PATH = os.path.join(self._DIRECTORIES['data'], 'memory.db')
        self._LOG_PATH = os.path.join(self._DIRECTORIES['logs'], 'nova.log')
        self._TEMP_PATH = self._DIRECTORIES['temp']
        self._OUTPUT_DIRECTORY = self._DIRECTORIES['captures']

        # API Keys
        self._OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        self._OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', '')
        self._AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY', '')
        self._AZURE_SPEECH_REGION = os.getenv('AZURE_SPEECH_REGION', '')
        
        # Default settings
        self._DEFAULT_WEATHER_LOCATION = os.getenv('DEFAULT_WEATHER_LOCATION', 
                                                 'Greenville, South Carolina')
        self._CONVERSATION_TIMEOUT = int(os.getenv('CONVERSATION_TIMEOUT', '10'))
        self._RETRY_DELAY = int(os.getenv('RETRY_DELAY', '10'))
        self._MAX_MEMORY_SIZE = int(os.getenv('MAX_MEMORY_SIZE', '100'))
        self._MEMORY_TTL_HOURS = int(os.getenv('MEMORY_TTL_HOURS', '1'))

        self._initialized = True
        logger.info(f"Config initialized with ASSISTANT_NAME: {self._ASSISTANT_NAME}")

    @property
    def ASSISTANT_NAME(self) -> str:
        return self._ASSISTANT_NAME

    @property
    def WAKE_WORD(self) -> str:
        return self._WAKE_WORD

    @property
    def DIRECTORIES(self) -> Dict[str, str]:
        return self._DIRECTORIES.copy()

    @property
    def DB_PATH(self) -> str:
        return self._DB_PATH

    @property
    def LOG_PATH(self) -> str:
        return self._LOG_PATH

    @property
    def TEMP_PATH(self) -> str:
        return self._TEMP_PATH

    @property
    def OUTPUT_DIRECTORY(self) -> str:
        return self._OUTPUT_DIRECTORY

    @property
    def OPENAI_API_KEY(self) -> str:
        return self._OPENAI_API_KEY

    @property
    def OPENWEATHERMAP_API_KEY(self) -> str:
        return self._OPENWEATHERMAP_API_KEY

    @property
    def AZURE_SPEECH_KEY(self) -> str:
        return self._AZURE_SPEECH_KEY

    @property
    def AZURE_SPEECH_REGION(self) -> str:
        return self._AZURE_SPEECH_REGION

    @property
    def DEFAULT_WEATHER_LOCATION(self) -> str:
        return self._DEFAULT_WEATHER_LOCATION

    @property
    def CONVERSATION_TIMEOUT(self) -> int:
        return self._CONVERSATION_TIMEOUT

    @property
    def RETRY_DELAY(self) -> int:
        return self._RETRY_DELAY

    @property
    def MAX_MEMORY_SIZE(self) -> int:
        return self._MAX_MEMORY_SIZE

    @property
    def MEMORY_TTL_HOURS(self) -> int:
        return self._MEMORY_TTL_HOURS

    def __str__(self) -> str:
        """String representation of config (excluding sensitive data)."""
        return f"Config(ASSISTANT_NAME='{self.ASSISTANT_NAME}', WAKE_WORD='{self.WAKE_WORD}')"

@lru_cache(maxsize=None)
def get_config() -> Config:
    """Get the configuration instance (cached)."""
    return Config()

# For backwards compatibility
config = get_config()

if __name__ == "__main__":
    # Test the configuration
    test_config = get_config()
    print(f"Config test:")
    print(f"ASSISTANT_NAME: {test_config.ASSISTANT_NAME}")
    print(f"WAKE_WORD: {test_config.WAKE_WORD}")
    print(f"Full config: {test_config}")
    
    # Test singleton behavior
    another_config = get_config()
    print(f"\nSingleton test:")
    print(f"Same instance: {test_config is another_config}")