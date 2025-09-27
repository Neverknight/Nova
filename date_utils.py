from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
import re
from logging_config import logger

class DateTimeUtils:
    """Centralized date and time handling utilities"""
    
    @staticmethod
    def extract_date(text_or_nlu_result: Union[str, Dict[str, Any]]) -> Optional[str]:
        """
        Extract date information from text or NLU result.
        
        Args:
            text_or_nlu_result: Either raw text or NLU result dictionary
            
        Returns:
            Optional[str]: Extracted date in YYYY-MM-DD format, or None if no date found
        """
        try:
            # Handle NLU result input
            if isinstance(text_or_nlu_result, dict):
                entities = text_or_nlu_result.get('entities', [])
                for entity, entity_type in entities:
                    if entity_type == 'DATE':
                        return DateTimeUtils._normalize_date(entity)
                return None
            
            # Handle raw text input
            text = text_or_nlu_result.lower()
            
            # Check for common date patterns
            patterns = {
                'tomorrow': lambda: datetime.now() + timedelta(days=1),
                'next week': lambda: datetime.now() + timedelta(weeks=1),
                r'next (monday|tuesday|wednesday|thursday|friday|saturday|sunday)': 
                    lambda match: DateTimeUtils._get_next_weekday(match.group(1)),
                r'in (\d+) days?': lambda match: datetime.now() + timedelta(days=int(match.group(1))),
                r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})': DateTimeUtils._parse_date_format
            }
            
            for pattern, handler in patterns.items():
                match = re.search(pattern, text)
                if match:
                    result_date = handler(match) if callable(handler) else handler
                    return result_date.strftime('%Y-%m-%d')
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting date: {e}")
            return None

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """Normalize various date formats to YYYY-MM-DD"""
        try:
            # Handle relative dates
            relative_dates = {
                'today': datetime.now(),
                'tomorrow': datetime.now() + timedelta(days=1),
                'day after tomorrow': datetime.now() + timedelta(days=2)
            }
            
            if date_str.lower() in relative_dates:
                return relative_dates[date_str.lower()].strftime('%Y-%m-%d')
            
            # Try parsing with various formats
            formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%B %d, %Y',
                '%d %B %Y',
                '%Y/%m/%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            return date_str
            
        except Exception as e:
            logger.error(f"Error normalizing date: {e}")
            return date_str

    @staticmethod
    def _get_next_weekday(day_name: str) -> datetime:
        """Get the date of the next occurrence of a weekday"""
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        target_day = weekdays[day_name.lower()]
        current = datetime.now()
        days_ahead = target_day - current.weekday()
        
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
            
        return current + timedelta(days=days_ahead)

    @staticmethod
    def _parse_date_format(match) -> datetime:
        """Parse dates in format DD/MM/YY or similar"""
        day, month, year = map(int, match.groups())
        if year < 100:
            year += 2000 if year < 50 else 1900
        return datetime(year, month, day)

    @staticmethod
    def format_relative_time(target_datetime: datetime) -> str:
        """
        Format a datetime into a human-readable relative time string.
        
        Args:
            target_datetime: The datetime to format
            
        Returns:
            str: Human-readable relative time (e.g., "in 2 hours", "tomorrow at 3 PM")
        """
        now = datetime.now()
        delta = target_datetime - now
        
        if delta.days == 0:
            if delta.seconds < 3600:
                minutes = delta.seconds // 60
                return f"in {minutes} minutes"
            else:
                return f"at {target_datetime.strftime('%I:%M %p')}"
        elif delta.days == 1:
            return f"tomorrow at {target_datetime.strftime('%I:%M %p')}"
        elif delta.days < 7:
            return f"on {target_datetime.strftime('%A')} at {target_datetime.strftime('%I:%M %p')}"
        else:
            return f"on {target_datetime.strftime('%B %d')} at {target_datetime.strftime('%I:%M %p')}"

date_utils = DateTimeUtils()