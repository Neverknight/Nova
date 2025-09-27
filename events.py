from typing import Callable, Any
import asyncio
from logging_config import logger

class EventSystem:
    def __init__(self):
        self._handlers = {}

    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def emit(self, event_type: str, data: Any = None):
        """Emit an event to all registered handlers."""
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")

# Create global event system
event_system = EventSystem()