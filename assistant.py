import os
from text_to_speech import speak as tts_speak

def speak(text: str) -> None:
    # Skip speech in test mode
    if os.environ.get('TEST_MODE'):
        return
    tts_speak(text)

import time
import json
import platform
import signal
import sys
import threading
import speech_recognition as sr
import date_utils
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from collections import deque
from config import get_config
from clients import ai_clients
from conversation_state import reset_conversation_state, start_conversation_state, is_active_conversation
from function_calls import function_call, gpt4_prompt, analyze_text, execute_function_based_on_response
from screenshot_webcam import take_screenshot, web_cam_capture, analyze_screenshot, analyze_image
from clipboard_vision import get_clipboard_text, vision_prompt
from text_to_speech import speak, tts_engine
from weather_functions import weather_service, format_weather_response, format_forecast_response, extract_location
from cleanup import clean_up
from logging_config import logger
from memory import remember_interaction, get_relevant_memories
from advanced_nlu import process_nlu
from dialogue_manager import DialogueManager, DialogueState
from system_control import system_controller

os.environ['TEST_MODE'] = 'true'

def shutdown_handler(signum, frame):
    """Handle shutdown gracefully"""
    print("\nShutting down Astra...")
    logger.info("Shutdown initiated")
    if tts_engine:
        tts_engine.cleanup()
    clean_up()
    sys.exit(0)

# Register shutdown handlers
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# Get configuration
config = get_config()

# Initialize the DialogueManager
dialogue_manager = DialogueManager()

# Initialize Speech Recognition
r = sr.Recognizer()
source = sr.Microphone()

def initialize_assistant():
    """Initialize the assistant with test mode awareness"""
    # Check if we're in test mode
    is_test_mode = os.environ.get('TEST_MODE') == 'true'
    
    if is_test_mode:
        logger.info("Initializing assistant in test mode")
        # Skip background process initialization
        return
        
    # Normal initialization continues here
    config = get_config()

    
    # Set default location if not already set
    if not hasattr(config, 'DEFAULT_WEATHER_LOCATION') or not config.DEFAULT_WEATHER_LOCATION:
        config.DEFAULT_WEATHER_LOCATION = "Greenville, South Carolina"
    
    # Register task handlers
    dialogue_manager.register_task_handler("get_weather", lambda nlu_result: format_weather_response(
        weather_service.get_weather(extract_location(nlu_result) or config.DEFAULT_WEATHER_LOCATION),
        extract_location(nlu_result) or config.DEFAULT_WEATHER_LOCATION,
        None
    ))
    dialogue_manager.register_task_handler("get_forecast", lambda nlu_result: format_forecast_response(
        weather_service.get_forecast(
            extract_location(nlu_result) or config.DEFAULT_WEATHER_LOCATION, 
            forecast_type='daily',
            days=1 if date_utils.extract_date(nlu_result) == 'tomorrow' else 0
        ),
        extract_location(nlu_result) or config.DEFAULT_WEATHER_LOCATION,
        'daily',
        date_utils.extract_date(nlu_result),
        None
    ))
    dialogue_manager.register_task_handler("get_time", lambda nlu_result: f"The current time is {datetime.now().strftime('%I:%M %p')}.")
    dialogue_manager.register_task_handler("take_screenshot", take_screenshot)
    dialogue_manager.register_task_handler("analyze_screenshot", analyze_screenshot)
    dialogue_manager.register_task_handler("capture_webcam", lambda nlu_result: web_cam_capture())
    dialogue_manager.register_task_handler("analyze_webcam", lambda nlu_result: analyze_image(web_cam_capture()))
    dialogue_manager.register_task_handler("extract_clipboard", lambda nlu_result: get_clipboard_text())
    dialogue_manager.register_task_handler("general_conversation", lambda nlu_result: ai_clients.call_gpt4(nlu_result['text']))
    
    logger.info(f"Assistant initialized with task handlers. Default location set to {config.DEFAULT_WEATHER_LOCATION}")

def extract_location(nlu_result: Dict[str, Any]) -> Optional[str]:
    """Extract location entity from NLU result."""
    for entity, entity_type in nlu_result['entities']:
        if entity_type == 'GPE':
            return entity
    return None

def rate_limiter(max_calls: int, period: int):
    """
    Rate limiting decorator that limits the number of calls to a function
    within a specified time period.
    
    Args:
        max_calls (int): Maximum number of calls allowed within the period
        period (int): Time period in seconds
    
    Returns:
        Callable: Decorated function with rate limiting
    """
    def decorator(func: Callable) -> Callable:
        # Use deque to store timestamps of calls
        calls = deque(maxlen=max_calls)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            now = datetime.now()
            
            # Remove timestamps older than the period
            while calls and now - calls[0] > timedelta(seconds=period):
                calls.popleft()
            
            # If we haven't reached the limit, make the call
            if len(calls) < max_calls:
                calls.append(now)
                return func(*args, **kwargs)
            
            # If we have reached the limit, check when we can make the next call
            next_call = calls[0] + timedelta(seconds=period)
            if now < next_call:
                sleep_time = (next_call - now).total_seconds()
                logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                calls.append(datetime.now())
                return func(*args, **kwargs)
            
            # If we get here, we can make the call
            calls.append(now)
            return func(*args, **kwargs)
            
        return wrapper
    return decorator

def setup_signal_handlers():
    """Setup proper signal handlers for both Windows and Unix systems."""
    if platform.system() == 'Windows':
        try:
            import win32api
            def handler(sig, func=None):
                if sig == signal.SIGTERM:
                    shutdown_handler(sig, None)
                return True
            win32api.SetConsoleCtrlHandler(handler, True)
        except ImportError:
            logger.warning("win32api not available, Windows signal handling disabled")
    
    # Register standard signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

def process_input(input_text: str) -> bool:
    """Process user input and return success status."""
    try:
        sanitized_input = input_text.strip()
        logger.info(f"Sanitized input: {sanitized_input}")
        
        if not sanitized_input:
            speak("I'm sorry, but I didn't receive any input. Could you please try again?")
            return True  # Continue the loop

        # Get response from dialogue manager
        response = dialogue_manager.process_input(sanitized_input)
        
        # Output response
        print(f"Assistant: {response}")
        speak(response)

        # Queue interaction for memory storage
        remember_interaction(
            sanitized_input, 
            response, 
            dialogue_manager._get_context_dict()
        )

        return True

    except Exception as e:
        logger.error(f"Error processing input: {e}", exc_info=True)
        error_message = "I'm sorry, but I encountered an error while processing your request. Is there something else I can help you with?"
        print(f"Assistant: {error_message}")
        speak(error_message)
        return True  # Continue the loop despite error

def initialize_assistant():
    """Initialize the assistant with test mode awareness"""
    # Check if we're in test mode
    is_test_mode = os.environ.get('TEST_MODE') == 'true'
    
    if is_test_mode:
        logger.info("Initializing assistant in test mode")
        # Skip background process initialization
        return
        
    # Normal initialization continues here
    config = get_config()
    
    # Set default location if not already set
    if not hasattr(config, 'DEFAULT_WEATHER_LOCATION') or not config.DEFAULT_WEATHER_LOCATION:
        config.DEFAULT_WEATHER_LOCATION = "Greenville, South Carolina"
    
    # Initialize the dialogue manager with task handlers
    dialogue_manager = DialogueManager()
    
    # Register task handlers
    dialogue_manager.register_task_handler("get_weather", lambda nlu_result: format_weather_response(
        weather_service.get_weather(extract_location(nlu_result) or config.DEFAULT_WEATHER_LOCATION),
        extract_location(nlu_result) or config.DEFAULT_WEATHER_LOCATION,
        None
    ))
    
    dialogue_manager.register_task_handler("get_time", lambda nlu_result: f"The current time is {datetime.now().strftime('%I:%M %p')}.")
    dialogue_manager.register_task_handler("take_screenshot", take_screenshot)
    dialogue_manager.register_task_handler("analyze_screenshot", analyze_screenshot)
    dialogue_manager.register_task_handler("capture_webcam", lambda nlu_result: web_cam_capture())
    dialogue_manager.register_task_handler("analyze_webcam", lambda nlu_result: analyze_image(web_cam_capture()))
    dialogue_manager.register_task_handler("extract_clipboard", lambda nlu_result: get_clipboard_text())
    dialogue_manager.register_task_handler("general_conversation", lambda nlu_result: ai_clients.call_gpt4(nlu_result['text']))
    
    # Register system control handlers
    dialogue_manager.register_task_handler("launch_app", lambda nlu_result: system_controller.launch_application(nlu_result.get('app_name')))
    dialogue_manager.register_task_handler("list_files", lambda nlu_result: system_controller.list_directory(nlu_result.get('location')))
    dialogue_manager.register_task_handler("system_info", lambda nlu_result: system_controller.get_system_info())
    dialogue_manager.register_task_handler("search_files", lambda nlu_result: system_controller.search_files(nlu_result.get('query')))
    dialogue_manager.register_task_handler("navigate", lambda nlu_result: system_controller.navigate_to(nlu_result.get('location')))
    
    logger.info(f"Assistant initialized with task handlers. Default location set to {config.DEFAULT_WEATHER_LOCATION}")

@rate_limiter(max_calls=30, period=60)  # 30 calls per minute
def voice_callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio)
        logger.info(f"Recognized: {text}")
        
        # If conversation is not active, check for wake word
        if not is_active_conversation():
            if config.WAKE_WORD.lower() in text.lower():
                logger.info("Wake word detected - starting conversation")
                start_conversation_state()
                # Remove wake word from the text
                text = text.lower().replace(config.WAKE_WORD.lower(), '').strip()
                if text:  # If there's remaining text after wake word
                    process_input(text)
        else:
            # Conversation is active, process any input directly
            process_input(text)
            
    except sr.UnknownValueError:
        logger.info("Google Speech Recognition could not understand audio")
        # Don't reset conversation state for recognition errors
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}")
        # Don't reset conversation state for recognition errors
    except Exception as e:
        logger.error(f"Error in voice callback: {e}")
        reset_conversation_state()  # Reset state on general errors

def voice_mode():
    print(f"Starting voice mode. Say '{config.WAKE_WORD}' to start a conversation.")
    # Reset conversation state when starting voice mode
    reset_conversation_state()
    
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    
    stop_listening = r.listen_in_background(source, voice_callback)
    
    try:
        while True:
            time.sleep(0.1)
            # Check if conversation has timed out
            if is_active_conversation():
                logger.debug("Conversation is active")
            else:
                logger.debug("Waiting for wake word...")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting voice mode.")
    finally:
        stop_listening(wait_for_stop=False)
        reset_conversation_state()

def text_mode():
    """Run the assistant in text mode."""
    print("Starting text mode. Type 'exit' to end the session.")
    
    while True:
        try:
            # Print prompt and get input
            print("\nYou: ", end='', flush=True)
            user_input = input().strip()
            
            # Handle exit condition
            if user_input.lower() == 'exit':
                print("Exiting text mode.")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Process input and check if we should continue
            if not process_input(user_input):
                break
            
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting text mode.")
            break
        except EOFError:
            print("\nEOF received. Exiting text mode.")
            break
        except Exception as e:
            logger.error(f"Error in text mode: {e}", exc_info=True)
            print("\nAn error occurred. Please try again.")
            continue

# Main block with platform check
if __name__ == "__main__":
    setup_signal_handlers()
    try:
        # Initialize the assistant
        initialize_assistant()
        
        print(f"Starting {config.ASSISTANT_NAME}...")
        mode = input("Choose mode (voice/text): ").lower().strip()
        if mode == "voice":
            print(f"\nSay '{config.WAKE_WORD}' to start a conversation.")
            voice_mode()
        elif mode == "text":
            print(f"\nStarting {config.ASSISTANT_NAME} in text mode. Type 'exit' to end the session.")
            text_mode()
        else:
            print("Invalid mode selected. Exiting.")
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")
        print("An unexpected error occurred. The program will now exit.")
    finally:
        clean_up()

print(f"\nThank you for using {config.ASSISTANT_NAME}. Goodbye!")

# IMPLEMENT FIXES FROM CLAUDE, CONTINUE TESTING