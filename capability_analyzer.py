import inspect
from typing import Dict, Set, List
import importlib
from advanced_nlu import IntentClassifier
from config import get_config
from system_control import SystemController
from logging_config import logger

config = get_config()

class CapabilityAnalyzer:
    def __init__(self):
        self.capabilities = {
            'core_functions': set(),
            'system_control': set(),
            'task_management': set(),
            'communication': set()
        }
        self.limitations = set()

    def analyze_intents(self) -> None:
        """Analyze available intents from IntentClassifier."""
        try:
            classifier = IntentClassifier()
            for intent, data in classifier.intents.items():
                category = self._categorize_intent(intent)
                patterns = data.get('patterns', [])
                keywords = data.get('keywords', [])
                capability = f"{intent}: {', '.join(patterns[:2])}... ({', '.join(keywords[:3])}...)"
                self.capabilities[category].add(capability)
        except Exception as e:
            logger.error(f"Error analyzing intents: {e}")

    def analyze_system_capabilities(self) -> None:
        """Analyze system control capabilities."""
        try:
            controller = SystemController()
            # Get methods that don't start with _
            methods = [method for method in dir(controller) 
                      if not method.startswith('_') and 
                      callable(getattr(controller, method))]
            
            for method in methods:
                func = getattr(controller, method)
                doc = inspect.getdoc(func)
                if doc:
                    self.capabilities['system_control'].add(f"{method}: {doc.split('.')[0]}")
        except Exception as e:
            logger.error(f"Error analyzing system control: {e}")

    def _categorize_intent(self, intent: str) -> str:
        """Categorize an intent into a capability category."""
        if any(word in intent.lower() for word in ['task', 'reminder', 'schedule']):
            return 'task_management'
        elif any(word in intent.lower() for word in ['system', 'file', 'app', 'launch']):
            return 'system_control'
        elif any(word in intent.lower() for word in ['weather', 'time', 'screenshot']):
            return 'core_functions'
        else:
            return 'communication'

    def generate_system_message(self) -> str:
        """Generate a comprehensive system message based on analyzed capabilities."""
        message = [
            f'You are a multi-modal AI voice assistant named {config.ASSISTANT_NAME}. Your current capabilities include:\n'
        ]

        for category, caps in self.capabilities.items():
            if caps:
                message.append(f"\n{category.replace('_', ' ').title()}:")
                for cap in sorted(caps):
                    message.append(f"- {cap}")

        message.append('\nImportant Notes:')
        message.append('- When discussing capabilities, only mention these specific functions.')
        message.append('- If asked to perform a task outside these capabilities, politely explain that you cannot do so.')
        message.append('- For visual tasks, you use advanced image recognition to interpret and describe contents.')
        message.append('- Always strive to provide accurate and helpful responses within the scope of your actual abilities.')

        return '\n'.join(message)

def update_system_message() -> str:
    """Update the system message based on current capabilities."""
    try:
        analyzer = CapabilityAnalyzer()
        analyzer.analyze_intents()
        analyzer.analyze_system_capabilities()  # Changed from analyze_system_control
        return analyzer.generate_system_message()
    except Exception as e:
        logger.error(f"Error updating system message: {e}")
        return "I am Nova, a multi-modal AI assistant. I can help you with various tasks and engage in natural conversation."

if __name__ == "__main__":
    # Test the capability analyzer
    print("\nGenerating System Message...")
    print("-" * 50)
    print(update_system_message())