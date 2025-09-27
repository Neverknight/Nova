import os
import pyperclip
from PIL import Image
from typing import Optional
import base64
from clients import ai_clients
from logging_config import logger

def get_clipboard_text() -> Optional[str]:
    """
    Retrieve text from the clipboard.

    Returns:
        Optional[str]: The text from the clipboard, or None if no text is available or an error occurs.
    """
    try:
        clipboard_content = pyperclip.paste()
        if isinstance(clipboard_content, str):
            return clipboard_content
        else:
            logger.info('No clipboard text to copy')
            return None
    except Exception as e:
        logger.error(f"Error in get_clipboard_text: {e}")
        return None

def vision_prompt(prompt: str, photo_path: str) -> str:
    try:
        with open(photo_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe this image naturally, as if you're telling someone what you see. Don't start with phrases like 'The image shows' or 'I can see'. Just describe it directly: {prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        response = ai_clients.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=300
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in vision_prompt: {e}")
        return "I'm sorry, I couldn't analyze the image."

def is_image_file(file_path: str) -> bool:
    """
    Check if a given file is a valid image file.

    Args:
        file_path (str): The path to the file to be checked.

    Returns:
        bool: True if the file is a valid image, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def process_clipboard_image(prompt: str) -> Optional[str]:
    temp_path = "temp_clipboard_image.png"
    try:
        image = Image.grabclipboard()
        if image is not None:
            image.save(temp_path)
            if is_image_file(temp_path):
                return vision_prompt(prompt, temp_path)
            else:
                logger.error("Invalid image file in clipboard")
                return None
        else:
            logger.info("No image found in clipboard")
            return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Failed to remove temporary file {temp_path}: {e}")