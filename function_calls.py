from clients import ai_clients, SYS_MSG_GPT4
from logging_config import logger
from weather_functions import weather_service, format_weather_response, format_forecast_response, get_next_day_date
from screenshot_webcam import take_screenshot, web_cam_capture
from clipboard_vision import get_clipboard_text, vision_prompt
from datetime import datetime
from typing import Dict, Any, List, Optional
from textblob import TextBlob
from memory import get_relevant_memories
from config import get_config
import json
import spacy
import re

config = get_config()

FORECAST_TYPES = {"hourly", "daily", "weekly"}

# Load a larger spaCy model
nlp = spacy.load('en_core_web_lg')

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze the input text for entities, sentiment, and key phrases.
    
    Args:
        text (str): The input text to analyze.
    
    Returns:
        Dict[str, Any]: A dictionary containing the analysis results.
    """
    doc = nlp(text)
    blob = TextBlob(text)
    
    # Extract named entities
    entities = {ent.label_: ent.text for ent in doc.ents}
    
    # Perform sentiment analysis
    sentiment = blob.sentiment.polarity
    
    # Extract key phrases (noun chunks)
    key_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    return {
        "entities": entities,
        "sentiment": sentiment,
        "key_phrases": key_phrases
    }

def validate_input(text: str) -> str:
    """Sanitize and validate user input."""
    # Remove any potential harmful characters
    cleaned = re.sub(r'[^\w\s\-\.,?!]', '', text)
    # Limit input length
    cleaned = cleaned[:1000]  # Reasonable maximum length
    return cleaned.strip()

def function_call(nlu_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    intents = nlu_result['intents']
    entities = nlu_result['entities']
    context = nlu_result['context']

    # Handle short greetings or farewells
    if len(nlu_result['text'].split()) <= 5 and any(intent in ['greeting', 'farewell'] for intent in nlu_result['intents']):
        return [{'function': 'general_query', 'location': None, 'forecast_type': None, 'specific_date': None, 'specific_info': None}]

    sys_msg = (
        'You are an AI function calling model for a multi-modal assistant. Analyze the user\'s query and determine '
        'the most appropriate functions to call based on the provided NLU result. Respond with a JSON array containing objects with the following fields:\n'
        '1. "function": The function to call (one of "get_current_weather", "get_forecast", "take_screenshot", "capture_webcam", "extract_clipboard", "handle_negative_sentiment", or "general_query")\n'
        f'2. "location": The location mentioned in the query, or "{config.DEFAULT_WEATHER_LOCATION}" if not specified\n'
        '3. "forecast_type": For forecasts, specify "hourly", "daily", or "weekly", or null if not applicable\n'
        '4. "specific_date": If specific dates or days of the week are mentioned, include them as a list of strings, or null if not specified\n'
        '5. "specific_info": Any specific information requested (e.g., "temperature", "humidity", "sunset"), or null if not specified\n'
        'Ensure your response is a valid JSON array of objects. Only include the "take_screenshot" function if the user explicitly requests to see what\'s on their screen or take a screenshot.'
    )

    prompt = (
        f"NLU Result:\n"
        f"Intents: {intents}\n"
        f"Entities: {entities}\n"
        f"Context: {context}\n"
        f"Based on these NLU results, determine the most appropriate functions to call."
    )

    messages = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': prompt}
    ]

    try:
        # Validate input
        if not isinstance(nlu_result, dict):
            raise ValueError("Invalid NLU result format")
        
        text = nlu_result.get('text', '')
        if not text:
            raise ValueError("Empty input text")
            
        cleaned_text = validate_input(text)
        nlu_result['text'] = cleaned_text
        
        response = ai_clients.call_gpt4(messages)
        logger.info(f"Function call response: {response}")
        
        # Extract JSON from the response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = response[json_start:json_end]
            try:
                parsed_response = json.loads(json_str)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON: {json_str}")
                return [default_response()]
        else:
            logger.error(f"No JSON array found in the response: {response}")
            return [default_response()]
        
        if not isinstance(parsed_response, list):
            parsed_response = [parsed_response]
        
        for func in parsed_response:
            if 'location' not in func or not func['location']:
                func['location'] = config.DEFAULT_WEATHER_LOCATION

        # Remove 'take_screenshot' only if not explicitly requested
        if 'screen' not in nlu_result['text'].lower():
            parsed_response = [func for func in parsed_response if func['function'] != 'take_screenshot']
        
        # If no specific functions were identified, return only one general query
        if all(func['function'] == 'general_query' for func in parsed_response):
            return [parsed_response[0]]
        
        # Remove 'general_query' if other specific functions are present
        specific_functions = [func for func in parsed_response if func['function'] != 'general_query']
        
        # Prioritize get_current_weather over get_forecast
        if any(func['function'] == 'get_current_weather' for func in specific_functions):
            specific_functions = [func for func in specific_functions if func['function'] != 'get_forecast']
        
        # Prioritize weather-related functions
        weather_functions = [func for func in specific_functions if func['function'] in ['get_current_weather', 'get_forecast']]
        other_functions = [func for func in specific_functions if func['function'] not in ['get_current_weather', 'get_forecast']]
        
        # Combine weather functions (if any) with other functions
        prioritized_functions = weather_functions + other_functions

        # If asking about sunset, ensure it's included in specific_info
        if any(word in nlu_result['text'].lower() for word in ['sunset', 'sun down', 'sundown']):
            for func in prioritized_functions:
                if func['function'] in ['get_current_weather', 'get_forecast']:
                    func['specific_info'] = 'sunset'

        return prioritized_functions if prioritized_functions else [default_response()]

    except Exception as e:
        logger.error(f"Unexpected error in function_call: {e}")
        return [default_response()]

def execute_function_based_on_response(response: Dict[str, Any], nlu_result: Dict[str, Any]) -> str:
    function = response.get('function', 'general_query')
    location = response.get('location') or config.DEFAULT_WEATHER_LOCATION
    forecast_type = response.get('forecast_type')
    specific_date = response.get('specific_date')
    specific_info = response.get('specific_info')

    # Handle the case where specific_date is a list
    if isinstance(specific_date, list):
        specific_date = specific_date[0] if specific_date else None

    results = []

    try:
        if function == "take_screenshot":
            screenshot_path = take_screenshot()
            if screenshot_path:
                visual_context = vision_prompt(prompt=nlu_result['text'], photo_path=screenshot_path)
                results.append(f"I've taken a screenshot of your screen. {visual_context}")
            else:
                results.append("I'm sorry, I couldn't take a screenshot.")

        elif function == "get_current_weather":
            weather_data = weather_service.get_weather(location)
            if weather_data:
                specific_info = "sunset time" if any(word in nlu_result['text'].lower() for word in ['sunset', 'sun down', 'sundown']) else None
                result = format_weather_response(weather_data, location, specific_info)
            else:
                result = f"I'm sorry, I couldn't fetch the current weather data for {location}. Please try again later."

        elif function == "get_forecast":
            if isinstance(specific_date, list) and specific_date:
                specific_date = get_next_day_date(specific_date[0])
            elif isinstance(specific_date, str):
                specific_date = get_next_day_date(specific_date)

            forecast_data = weather_service.get_forecast(location, forecast_type=forecast_type)
            if forecast_data:
                formatted_forecast = format_forecast_response(forecast_data, location, forecast_type, specific_date, specific_info)
                interpretation_prompt = (
                    f"Based on this forecast for {location}: {json.dumps(formatted_forecast, indent=2)}\n"
                    f"User question: {nlu_result['text']}\n"
                    "Provide a conversational response that addresses the user's question and interprets the weather data. "
                    "Make the response friendly and natural, as if you're having a casual conversation about the weather. "
                    f"The location is {location}, do not ask for a location."
                )
                results.append(gpt4_prompt(interpretation_prompt, weather_data=formatted_forecast, context=nlu_result['context']))
            else:
                results.append(f"I'm sorry, I couldn't fetch the forecast data for {location}. Please try again later.")

        elif function == "capture_webcam":
            webcam_image_path = web_cam_capture()
            if webcam_image_path:
                result += f"I've captured a webcam image and saved it as {webcam_image_path}. "
                visual_context = vision_prompt(prompt=nlu_result['text'], photo_path=webcam_image_path)
                result += gpt4_prompt(nlu_result['text'], img_context=visual_context, context=nlu_result['context'])
            else:
                result += "I'm sorry, I couldn't capture a webcam image. "

        elif function == "extract_clipboard":
            clipboard_text = get_clipboard_text()
            if clipboard_text:
                result += f"The clipboard contains: {clipboard_text} "
            else:
                result += "The clipboard is empty. "

        elif function == "handle_negative_sentiment":
            result += handle_negative_sentiment(nlu_result['text'], nlu_result.get('sentiment', 0))

        elif function == "general_query":
            result = gpt4_prompt(nlu_result['text'], context=nlu_result['context'])

    except Exception as e:
        logger.error(f"Error in execute_function_based_on_response: {e}")
        return f"I apologize, but I encountered an error while processing your request about {function}. Please try asking again or rephrase your question."

    # Combine results into a coherent response
    combined_response = " ".join(results)
    final_prompt = (
        f"Combine the following information into a coherent and natural response, without repeating information: {combined_response}\n"
        f"Remember, the location for weather queries is always {location} unless explicitly stated otherwise. "
        "Do not mention the default location or ask for a location unless the user specifically asked about a different location. "
        "Do not repeat or summarize the weather information if it's already provided. "
        "Ensure the response flows naturally and doesn't sound like separate pieces of information stuck together."
    )
    try:
        return gpt4_prompt(final_prompt)
    except Exception as e:
        logger.error(f"Error in final gpt4_prompt: {e}")
        return combined_response  # Return the combined response without additional processing

def handle_negative_sentiment(text: str, sentiment: float) -> str:
    prompt = (
        f"The user has expressed a negative sentiment (score: {sentiment}). "
        f"Their input was: '{text}'. Please provide an empathetic response "
        "that acknowledges their feelings and offers support or assistance."
    )
    return gpt4_prompt(prompt)

def validate_and_clean_response(response: Dict[str, Any]) -> Dict[str, Any]:
    # Update the list of valid functions
    VALID_FUNCTIONS = {
        "get_current_weather",
        "get_forecast",
        "take_screenshot",
        "capture_webcam",
        "extract_clipboard",
        "handle_negative_sentiment",
        "general_query"
    }

    FORECAST_TYPES = {"hourly", "daily", "weekly"}
    
    cleaned_response = {}

    # Validate and clean 'function'
    function = response.get('function', 'general_query')
    cleaned_response['function'] = function if function in VALID_FUNCTIONS else 'general_query'

    # Validate and clean 'location'
    location = response.get('location')
    cleaned_response['location'] = location if isinstance(location, str) else None

    # Validate and clean 'forecast_type'
    forecast_type = response.get('forecast_type')
    cleaned_response['forecast_type'] = forecast_type if forecast_type in FORECAST_TYPES else None

    # Validate and clean 'specific_date'
    specific_date = response.get('specific_date')
    if isinstance(specific_date, list):
        cleaned_response['specific_date'] = specific_date[0] if specific_date else None
    elif isinstance(specific_date, str):
        cleaned_response['specific_date'] = specific_date
    else:
        cleaned_response['specific_date'] = None

    # Validate and clean 'specific_info'
    specific_info = response.get('specific_info')
    cleaned_response['specific_info'] = specific_info if isinstance(specific_info, str) else None

    # Validate and clean 'sentiment'
    sentiment = response.get('sentiment')
    cleaned_response['sentiment'] = float(sentiment) if isinstance(sentiment, (int, float)) else 0.0

    # Validate and clean 'key_phrases'
    key_phrases = response.get('key_phrases', [])
    cleaned_response['key_phrases'] = [str(phrase) for phrase in key_phrases] if isinstance(key_phrases, list) else []

    return cleaned_response

def default_response() -> Dict[str, Any]:
    return {
        "function": "general_query",
        "location": None,
        "forecast_type": None,
        "specific_date": None,
        "specific_info": None,
        "sentiment": 0.0,
        "key_phrases": []
    }

def gpt4_prompt(prompt: str, weather_data: Optional[Dict[str, Any]] = None, img_context: str = None, context: str = None) -> str:
    system_message = SYS_MSG_GPT4
    if context:
        system_message += f"\n\nConversation context: {context}"
    if weather_data:
        system_message += f"\n\nCurrent weather data: {json.dumps(weather_data, indent=2)}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    if img_context:
        messages[1]["content"] += f"\n\nImage context: {img_context}"
    
    try:
        response = ai_clients.call_gpt4(messages)
        return response
    except Exception as e:
        logger.error(f"Error in gpt4_prompt: {e}")
        return "I'm sorry, I encountered an error while processing your request. Could you please try again?"

__all__ = ['function_call', 'gpt4_prompt', 'execute_function_based_on_response']