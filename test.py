import os
import requests
from logging_config import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ElevenLabsTTS:
    def __init__(self):
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        
        if not self.api_key:
            raise ValueError("ElevenLabs API Key not found in environment variables")
        
        self.voice_id = "XB0fDUnXU5powFXDhCwa"  # Correct voice ID for Charlotte
        self.base_url = "https://api.elevenlabs.io/v1/text-to-speech"
        logger.info("Initialized ElevenLabs TTS with Charlotte voice")

    def synthesize_speech(self, text, output_file="output.mp3"):
        url = f"{self.base_url}/{self.voice_id}/stream"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            with open(output_file, "wb") as file:
                file.write(response.content)
            logger.info(f"Audio saved to {output_file}")
            self.cleanup(output_file)
            return output_file
        else:
            logger.error(f"Failed to synthesize speech: {response.status_code}, {response.text}")
            raise Exception(f"ElevenLabs TTS failed with status code {response.status_code}")

    def cleanup(self, file_path):
        """Remove the audio file after use."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up audio file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up audio file: {e}")

# Instantiate the ElevenLabs TTS class for usage
tts_engine = None
try:
    tts_engine = ElevenLabsTTS()
except Exception as e:
    logger.error(f"Failed to initialize TTS engine: {e}")

def speak(text: str) -> None:
    """Speak the given text using ElevenLabs TTS."""
    global tts_engine
    logger.info(f"Speak function called with text: '{text[:50]}...'")
    try:
        if tts_engine is None:
            logger.error("TTS engine not initialized")
            return
            
        tts_engine.synthesize_speech(text)
        logger.info("Speak function completed successfully")
    except Exception as e:
        logger.error(f"Error in speak function: {e}", exc_info=True)

if __name__ == "__main__":
    # Test the TTS system
    print("Testing ElevenLabs TTS...")
    speak("Hello, this is a test of the ElevenLabs Text to Speech system.")
    print("Test complete.")
