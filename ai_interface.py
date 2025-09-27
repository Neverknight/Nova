import openai
from typing import List, Dict, Any, Optional, Union
from logging_config import logger
from config import get_config

config = get_config()

class AIInterface:
    def __init__(self):
        self.openai_client = self._init_openai()
        self.conversation_history = []

    @staticmethod
    def _init_openai() -> openai.OpenAI:
        try:
            return openai.OpenAI(api_key=config.OPENAI_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def call_gpt4(self, prompt_or_messages: Union[str, List[Dict[str, str]]], 
                 img_context: Optional[str] = None) -> str:
        try:
            if isinstance(prompt_or_messages, str):
                prompt = f'USER PROMPT: {prompt_or_messages}\n\nIMAGE CONTEXT: {img_context}' if img_context else prompt_or_messages
                messages = [{'role': 'user', 'content': prompt}]
            else:
                messages = prompt_or_messages

            chat_completion = self.openai_client.chat.completions.create(
                model='gpt-4',
                messages=messages
            )
            response = chat_completion.choices[0].message.content
            self.conversation_history.append({'role': 'assistant', 'content': response})
            return response
        except Exception as e:
            logger.error(f"Error in call_gpt4: {e}")
            return "I'm sorry, I couldn't process your request."

    def reset_conversation(self) -> None:
        self.conversation_history = []

# Create global instance
ai_interface = AIInterface()