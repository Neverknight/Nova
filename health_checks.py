from typing import Dict, Tuple
import requests
import openai
from clients import ai_clients
import cv2
import config

def check_internet_connection() -> Tuple[bool, str]:
    try:
        requests.get("https://api.openai.com", timeout=5)
        return True, "Internet connection is available"
    except requests.RequestException:
        return False, "No internet connection"

def check_openai_api() -> Tuple[bool, str]:
    try:
        ai_clients.openai_client.models.list()
        return True, "OpenAI API is accessible"
    except Exception as e:
        return False, f"OpenAI API check failed: {str(e)}"

def check_weather_api() -> Tuple[bool, str]:
    try:
        response = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={config.OPENWEATHERMAP_API_KEY}",
            timeout=5
        )
        response.raise_for_status()
        return True, "Weather API is accessible"
    except Exception as e:
        return False, f"Weather API check failed: {str(e)}"

def check_webcam() -> Tuple[bool, str]:
    if ai_clients.web_cam is None:
        return False, "Webcam is not initialized"
    return ai_clients.web_cam.isOpened(), "Webcam is available"

def run_health_checks() -> Dict[str, Tuple[bool, str]]:
    return {
        "internet": check_internet_connection(),
        "openai_api": check_openai_api(),
        "weather_api": check_weather_api(),
        "webcam": check_webcam()
    }