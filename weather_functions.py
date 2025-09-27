import requests
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, timedelta
import spacy
from logging_config import logger
from config import get_config
from clients import ai_clients
from memory import get_relevant_memories
import json
from rate_limiter import RateLimiter
from retry import retry_with_backoff

# Initialize SpaCy
nlp = spacy.load('en_core_web_sm')

# Get configuration
config = get_config()

class WeatherService:
    BASE_URL = "http://api.openweathermap.org/data/2.5"
    
    def __init__(self):
        self.api_key = config.OPENWEATHERMAP_API_KEY
        self.rate_limiter = RateLimiter(calls=60, period=60)  # 60 calls per minute

    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        return kelvin - 273.15

    @staticmethod
    def celsius_to_fahrenheit(celsius: float) -> float:
        return celsius * 9/5 + 32

    @retry_with_backoff(retries=3)
    def get_weather(self, location: str) -> Dict[str, Any]:
        data = self._fetch_weather_data(location)
        if data:
            return {
                'description': data['weather'][0]['description'],
                'temperature': self.kelvin_to_celsius(data['main']['temp']),
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'sunset': data['sys']['sunset'],
                'sunrise': data['sys']['sunrise']
            }
        return None

    def get_forecast(self, location: str, forecast_type: str = 'daily', days: int = 1) -> List[Dict[str, Any]]:
        data = self._fetch_forecast_data(location)
        if data:
            target_date = datetime.now().date() + timedelta(days=days)
            forecast_data = []
            for item in data['list']:
                item_date = datetime.fromtimestamp(item['dt']).date()
                if item_date == target_date:
                    forecast_data.append({
                        'date': datetime.fromtimestamp(item['dt']),
                        'description': item['weather'][0]['description'],
                        'temperature': self.kelvin_to_celsius(item['main']['temp']),
                        'humidity': item['main']['humidity'],
                        'wind_speed': item['wind']['speed']
                    })
            return forecast_data
        return None

    def _fetch_weather_data(self, location: str) -> Optional[Dict]:
        try:
            self.rate_limiter.wait()
            url = f"{self.BASE_URL}/weather?q={location}&appid={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None

    def _fetch_forecast_data(self, location: str) -> Optional[Dict]:
        try:
            url = f"{self.BASE_URL}/forecast?q={location}&appid={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching forecast data: {e}")
            return None

def format_weather_response(weather_data: Dict[str, Any], location: str, specific_info: Optional[str] = None) -> str:
    if not weather_data:
        return f"I'm sorry, but I couldn't fetch the weather data for {location}."

    celsius = weather_data['temperature']
    fahrenheit = WeatherService.celsius_to_fahrenheit(celsius)
    
    formatted_weather = {
        "location": location,
        "description": weather_data['description'],
        "temperature": {
            "celsius": round(celsius, 1),
            "fahrenheit": round(fahrenheit, 1)
        },
        "humidity": weather_data['humidity'],
        "wind_speed": weather_data['wind_speed']
    }
    
    if specific_info in ["sunset time", "sunrise time"]:
        time_key = 'sunset' if specific_info == "sunset time" else 'sunrise'
        time_str = datetime.fromtimestamp(weather_data[time_key]).strftime('%I:%M %p')
        formatted_weather[time_key] = time_str
    
    context = (
        f"Current weather information: {json.dumps(formatted_weather, indent=2)}\n"
        f"User question: {specific_info if specific_info else 'general weather information'}\n"
        "Please provide a conversational weather report that addresses the user's question "
        "and includes the relevant weather details. If no specific information was requested, "
        "provide a friendly overview of the current weather conditions. "
        f"{'If the location is ' + config.DEFAULT_WEATHER_LOCATION + ', mention that this is the default location.' if location == config.DEFAULT_WEATHER_LOCATION else ''}"
    )
    
    response = ai_clients.call_gpt4([{"role": "user", "content": context}])
    return response

def format_forecast_response(forecast_data: List[Dict[str, Any]], location: str, forecast_type: str, specific_date: Optional[str], specific_info: Optional[str]) -> str:
    if not forecast_data:
        return f"I'm sorry, I couldn't retrieve the forecast data for {location}. This could be due to a temporary issue with the weather service."

    # Format the forecast information
    date_str = forecast_data[0]['date'].strftime("%A, %B %d, %Y")
    avg_temp = sum(day['temperature'] for day in forecast_data) / len(forecast_data)
    avg_humidity = sum(day['humidity'] for day in forecast_data) / len(forecast_data)
    avg_wind_speed = sum(day['wind_speed'] for day in forecast_data) / len(forecast_data)
    
    conditions = list(set(day['description'] for day in forecast_data))
    main_condition = max(conditions, key=conditions.count)

    formatted_data = {
        "location": location,
        "date": date_str,
        "conditions": main_condition,
        "temperature": {
            "celsius": round(avg_temp, 1),
            "fahrenheit": round(WeatherService.celsius_to_fahrenheit(avg_temp), 1)
        },
        "humidity": round(avg_humidity),
        "wind_speed": round(avg_wind_speed, 1)
    }

    context = (
        f"Forecast information: {json.dumps(formatted_data, indent=2)}\n"
        f"User question: {specific_info if specific_info else 'general forecast information'}\n"
        "Please provide a conversational forecast report that addresses the user's question "
        "and includes the relevant weather details. If no specific information was requested, "
        "provide a friendly overview of the forecast conditions. "
        f"{'If the location is ' + config.DEFAULT_WEATHER_LOCATION + ', mention that this is the default location.' if location == config.DEFAULT_WEATHER_LOCATION else ''}"
    )

    response = ai_clients.call_gpt4([{"role": "user", "content": context}])
    return response

def get_next_day_date(day_name: str) -> str:
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    target_day = day_name.lower().split()[-1]  # Get the last word in case of "upcoming Friday"
    if target_day not in days:
        return None
    today = datetime.now().weekday()
    day = days.index(target_day)
    days_ahead = day - today
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

def extract_location(text: Union[str, Dict[str, Any]]) -> Optional[str]:
    if isinstance(text, dict):
        # If text is a dictionary (nlu_result), extract the 'text' field
        text = text.get('text', '')
    
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE (Geo-Political Entity) covers locations like cities, countries, states
            return ent.text
    return None

# Create an instance of WeatherService
weather_service = WeatherService()

__all__ = ['WeatherService', 'weather_service', 'format_weather_response', 'format_forecast_response', 'extract_location', 'get_next_day_date']