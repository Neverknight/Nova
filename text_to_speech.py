import os
import requests
from logging_config import logger
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile
import threading
from queue import Queue, Empty
import subprocess
import atexit
import time
from base_cleanup import base_cleanup_manager

# Load environment variables
load_dotenv()

class AudioPlayer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = Queue()
        self.is_playing = threading.Event()
        self.stop_event = threading.Event()
        self.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                file_path = self.queue.get(timeout=0.5)
                if file_path:
                    self.is_playing.set()
                    subprocess.run(
                        ['powershell', '-c', f'$player = New-Object System.Media.SoundPlayer("{file_path}"); $player.PlaySync(); $player.Dispose()'],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    self.is_playing.clear()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio player thread: {e}")
                self.is_playing.clear()

    def play(self, file_path):
        self.queue.put(file_path)
        self.is_playing.wait()
        while self.is_playing.is_set():
            time.sleep(0.1)

    def stop(self):
        self.stop_event.set()
        if not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                pass
        self.join(timeout=1)

class ElevenLabsTTS:
    def __init__(self):
        logger.info("Initializing ElevenLabsTTS...")
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        
        if not self.api_key:
            raise ValueError("ElevenLabs API Key not found in environment variables")
        
        self.voice_id = "XB0fDUnXU5powFXDhCwa"  # Charlotte voice
        self.base_url = "https://api.elevenlabs.io/v1/text-to-speech"
        
        # Create temp directory and register it for cleanup
        self.temp_dir = tempfile.mkdtemp()
        base_cleanup_manager.add_temp_dir(self.temp_dir)
        
        self.audio_player = AudioPlayer()
        logger.info("Initialized ElevenLabs TTS with Charlotte voice")

    def synthesize_speech(self, text: str) -> str:
        logger.info("Starting speech synthesis...")
        output_file = os.path.join(self.temp_dir, "temp_speech.mp3")
        # Register the temp file for cleanup
        base_cleanup_manager.add_temp_file(output_file)

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
        
        try:
            logger.info("Making API request to ElevenLabs...")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            with open(output_file, "wb") as file:
                file.write(response.content)
            logger.info(f"Successfully saved audio to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            raise

# Create audio player instance
audio_player = AudioPlayer()

# Global TTS engine instance
tts_engine = None
try:
    logger.info("Creating TTS engine instance...")
    tts_engine = ElevenLabsTTS()
except Exception as e:
    logger.error(f"Failed to initialize TTS engine: {e}")

def speak(text: str) -> None:
    """Text to speech function."""
    # Skip speech in test mode
    if os.environ.get('TEST_MODE'):
        return
        
    logger.info(f"Speak function called with text: '{text[:50]}...'")
    
    try:
        if tts_engine is None:
            logger.error("TTS engine not initialized")
            return
            
        # Generate speech
        mp3_file = tts_engine.synthesize_speech(text)
        
        # Convert to WAV
        audio = AudioSegment.from_mp3(mp3_file)
        wav_file = os.path.join(tts_engine.temp_dir, "temp_speech.wav")
        # Register the temp WAV file for cleanup
        base_cleanup_manager.add_temp_file(wav_file)
        audio.export(wav_file, format="wav")
        
        # Play audio in separate thread
        logger.info("Starting audio playback")
        tts_engine.audio_player.play(wav_file)
        logger.info("Audio playback completed")
        
    except Exception as e:
        logger.error(f"Error in speak function: {e}", exc_info=True)
    finally:
        logger.info("Speak function finished")

# Register cleanup
atexit.register(audio_player.stop)

if __name__ == "__main__":
    # Test the TTS system
    print("Testing ElevenLabs TTS...")
    speak("Hello, this is a test of the ElevenLabs Text to Speech system.")
    print("Test complete.")