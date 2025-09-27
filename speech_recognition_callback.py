import os
import re
import threading
import time
from typing import Optional
import speech_recognition as sr
from conversation_state import conversation_state, reset_conversation_state, start_conversation_state, is_active_conversation
from function_calls import function_call, gpt4_prompt
from weather_functions import weather_service, extract_location
from text_to_speech import speak
from screenshot_webcam import take_screenshot, web_cam_capture
from clipboard_vision import get_clipboard_text, vision_prompt
from logging_config import logger
from config import get_config
from clients import ai_clients

config = get_config()

# Initialize Speech Recognition
r = sr.Recognizer()
source = sr.Microphone()

def wav_to_text(audio_path: str) -> str:
    try:
        segments, _ = ai_clients.whisper_model.transcribe(audio_path)
        text = ''.join(segment.text for segment in segments)
        return text
    except Exception as e:
        logger.error(f"Error in wav_to_text: {e}")
        return ""

def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    try:
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)
        clean_prompt = extract_prompt(prompt_text, config.WAKE_WORD) if not is_active_conversation() else prompt_text.strip()

        if clean_prompt:
            logger.info(f'USER: {clean_prompt}')
            response = function_call(clean_prompt)
            
            if response['function'] == 'take_screenshot':
                logger.info('Taking screenshot.')
                screenshot_path = take_screenshot()
                visual_context = vision_prompt(prompt=clean_prompt, photo_path=screenshot_path)
            elif response['function'] == 'capture_webcam':
                logger.info('Capturing webcam.')
                webcam_image_path = web_cam_capture()
                if os.path.exists(webcam_image_path):
                    visual_context = vision_prompt(prompt=clean_prompt, photo_path=webcam_image_path)
                else:
                    logger.error('Webcam capture failed, image not found.')
                    visual_context = None
            elif response['function'] == 'extract_clipboard':
                logger.info('Extracting clipboard text.')
                paste = get_clipboard_text()
                clean_prompt = f'{clean_prompt} \n\n    CLIPBOARD CONTENT: {paste}'
                visual_context = None
            elif response['function'] in ['get_current_weather', 'get_forecast']:
                location = response['location'] or extract_location(clean_prompt) or "Greenville, South Carolina"
                logger.info(f"Fetching weather for {location}")
                if response['function'] == 'get_current_weather':
                    weather_data = weather_service.get_weather(location)
                    if weather_data:
                        celsius = weather_data['temperature']
                        fahrenheit = weather_service.celsius_to_fahrenheit(celsius)
                        weather_info = (
                            f"The current weather in {location} is {weather_data['description']} with a temperature of "
                            f"{celsius:.1f}°C ({fahrenheit:.1f}°F). "
                            f"The humidity is {weather_data['humidity']}% and the wind speed is {weather_data['wind_speed']} m/s."
                        )
                    else:
                        weather_info = f"I'm sorry, I couldn't fetch the weather data for {location}."
                else:  # get_forecast
                    forecast_data = weather_service.get_forecast(location, forecast_type=response['forecast_type'])
                    if forecast_data:
                        weather_info = f"Here's the forecast for {location}:\n"
                        for day in forecast_data[:5]:  # Limit to 5 days for brevity
                            celsius = day['temperature']
                            fahrenheit = weather_service.celsius_to_fahrenheit(celsius)
                            weather_info += (
                                f"{day['timestamp'].strftime('%A, %B %d')}: {day['description']}, "
                                f"temperature {celsius:.1f}°C ({fahrenheit:.1f}°F), "
                                f"humidity {day['humidity']}%, wind speed {day['wind_speed']} m/s\n"
                            )
                    else:
                        weather_info = f"I'm sorry, I couldn't fetch the forecast data for {location}."
                clean_prompt = f'{clean_prompt} \n\n    WEATHER INFO: {weather_info}'
                visual_context = None
            else:
                visual_context = None

            gpt4_response = gpt4_prompt(prompt=clean_prompt, img_context=visual_context)
            logger.info(f'ASSISTANT: {gpt4_response}')
            speak(gpt4_response)

            # Set active conversation state
            start_conversation_state()

    except Exception as e:
        logger.error(f"Error in callback: {e}")

def extract_prompt(transcribed_text: str, wake_word: str) -> Optional[str]:
    if is_active_conversation():
        # Restart the timer with every new prompt to keep the conversation active
        start_conversation_state()
        logger.info("Active conversation detected, using full transcribed text.")
        return transcribed_text.strip()
    
    # Ensure the transcribed text is not empty
    if not transcribed_text.strip():
        logger.warning("Transcribed text is empty.")
        return None

    # Pattern to recognize the wake word followed by any text.
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    # If a match is found, start the conversation state and return the extracted prompt text.
    if match:
        prompt = match.group(1).strip()
        start_conversation_state()
        logger.info(f"Wake word detected. Extracted prompt: '{prompt}'")
        return prompt
    else:
        logger.warning("Wake word not detected or no prompt found after wake word.")
        return None

def start_listening() -> None:
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    logger.info(f'\nSay {config.WAKE_WORD} followed with your prompt. \n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(.5)