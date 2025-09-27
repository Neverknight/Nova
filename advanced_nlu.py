import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
import subprocess
from task_management import TaskCategory, TaskPriority, TaskDependency
from logging_config import logger
from error_recovery import error_handler
from collections import defaultdict
from system_control import system_controller

# Make sure spaCy model is installed
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    logger.warning("Downloading spaCy model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

class DateTimeExtractor:
    def __init__(self):
        self.relative_patterns = {
            r'in (\d+) minutes?': lambda x: timedelta(minutes=int(x)),
            r'in (\d+) hours?': lambda x: timedelta(hours=int(x)),
            r'in (\d+) days?': lambda x: timedelta(days=int(x)),
            r'in (\d+) weeks?': lambda x: timedelta(weeks=int(x)),
            r'in (\d+) months?': lambda x: timedelta(days=int(x) * 30),  # Approximate
        }
        
        self.specific_patterns = {
            r'at (\d{1,2}(?::\d{2})?\s*(?:am|pm))': self._parse_time,
            r'on ([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)': self._parse_date,
            r'next (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)': self._parse_next_weekday,
        }
        
        self.recurring_patterns = {
            r'every (\d+) (minutes?|hours?|days?|weeks?|months?)': self._create_cron,
            r'every (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)': self._create_weekly_cron,
            r'daily at (\d{1,2}(?::\d{2})?\s*(?:am|pm))': self._create_daily_cron,
        }

    def extract_datetime(self, text: str) -> Dict[str, Any]:
        """Extract all datetime related information from text."""
        result = {
            'due_date': None,
            'recurring': None,
            'remind_before': []
        }
        
        # Check for relative patterns
        for pattern, delta_func in self.relative_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['due_date'] = datetime.now() + delta_func(match.group(1))
                break
        
        # Check for specific patterns
        for pattern, parser_func in self.specific_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                parsed = parser_func(match.group(1))
                if parsed:
                    result['due_date'] = parsed
                break
        
        # Check for recurring patterns
        for pattern, cron_func in self.recurring_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['recurring'] = cron_func(*match.groups())
                break
        
        # Extract reminder times
        remind_patterns = [
            r'remind me (\d+) (minutes?|hours?|days?) before',
            r'remind me at (\d{1,2}(?::\d{2})?\s*(?:am|pm))',
        ]
        
        for pattern in remind_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(2) if len(match.groups()) > 1 else None:
                    # Relative reminder
                    unit = match.group(2).rstrip('s')
                    value = int(match.group(1))
                    delta = {
                        'minute': timedelta(minutes=value),
                        'hour': timedelta(hours=value),
                        'day': timedelta(days=value)
                    }[unit]
                    result['remind_before'].append(delta)
                else:
                    # Specific time reminder
                    reminder_time = self._parse_time(match.group(1))
                    if reminder_time:
                        result['remind_before'].append(
                            result['due_date'] - reminder_time if result['due_date'] else reminder_time
                        )
        
        return result

    @staticmethod
    def _parse_time(time_str: str) -> Optional[datetime]:
        try:
            return datetime.strptime(time_str.strip().lower(), '%I:%M %p')
        except ValueError:
            try:
                return datetime.strptime(time_str.strip().lower(), '%I %p')
            except ValueError:
                return None

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        try:
            return datetime.strptime(date_str.strip(), '%B %d')
        except ValueError:
            return None

    @staticmethod
    def _parse_next_weekday(day: str) -> datetime:
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        target_day = weekdays[day.lower()]
        current = datetime.now()
        days_ahead = target_day - current.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return current + timedelta(days=days_ahead)

    @staticmethod
    def _create_cron(value: str, unit: str) -> str:
        """Create a cron expression for recurring tasks."""
        unit = unit.rstrip('s').lower()
        if unit == 'minute':
            return f"*/{value} * * * *"
        elif unit == 'hour':
            return f"0 */{value} * * *"
        elif unit == 'day':
            return f"0 0 */{value} * *"
        elif unit == 'week':
            return f"0 0 * * 0/{value}"
        elif unit == 'month':
            return f"0 0 1 */{value} *"
        return None

    @staticmethod
    def _create_weekly_cron(day: str) -> str:
        weekdays = {
            'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4,
            'friday': 5, 'saturday': 6, 'sunday': 0
        }
        return f"0 0 * * {weekdays[day.lower()]}"

    @staticmethod
    def _create_daily_cron(time_str: str) -> str:
        time = datetime.strptime(time_str.strip().lower(), '%I:%M %p')
        return f"{time.minute} {time.hour} * * *"

class IntentClassifier:
    def __init__(self):
        self.intents = {
            "create_task": {
                "patterns": [
                    "remind me to", "schedule", "create task", "add task", "set reminder",
                    "need to", "have to", "must", "should", "todo", "to do"
                ],
                "keywords": ["reminder", "schedule", "task", "deadline", "due"]
            },
            "modify_task": {
                "patterns": [
                    "change", "modify", "update", "reschedule", "postpone", "move",
                    "edit", "adjust"
                ],
                "keywords": ["different", "new", "another", "instead"]
            },
            "delete_task": {
                "patterns": [
                    "cancel", "delete", "remove", "forget about", "drop"
                ],
                "keywords": ["nevermind", "don't need", "do not need"]
            },
            "query_tasks": {
                "patterns": [
                    "what tasks", "show tasks", "list tasks", "view tasks",
                    "what do i have", "what's scheduled", "what is scheduled"
                ],
                "keywords": ["upcoming", "pending", "today", "tomorrow", "next"]
            },
            "task_status": {
                "patterns": [
                    "status of", "progress on", "how's the", "how is the",
                    "update on"
                ],
                "keywords": ["progress", "status", "completion", "finished"]
            },
            "launch_app": {
                "patterns": [
                    "open", "launch", "start", "run", "execute"
                ],
                "keywords": ["application", "program", "app", "software"]
            },
            "list_files": {
                "patterns": [
                    "list", "show", "display", "what's in", "contents of"
                ],
                "keywords": ["files", "folder", "directory", "documents"]
            },
            "system_info": {
                "patterns": [
                    "system", "computer", "pc", "laptop", "status"
                ],
                "keywords": ["cpu", "memory", "disk", "battery", "usage"]
            },
            "search_files": {
                "patterns": [
                    "find", "search", "locate", "where is"
                ],
                "keywords": ["file", "document", "folder"]
            },
            "navigate": {
                "patterns": [
                    "go to", "change to", "switch to", "navigate"
                ],
                "keywords": ["folder", "directory", "location"]
            }
        }
        
        # Add existing intents from the current system
        self.intents.update({
            "greeting": {
                "patterns": ["hello", "hi", "hey", "good morning", "good afternoon"],
                "keywords": ["greetings", "morning", "afternoon", "evening"]
            },
            "farewell": {
                "patterns": ["goodbye", "bye", "see you", "talk to you later"],
                "keywords": ["farewell", "night", "later"]
            },
            "weather_query": {
                "patterns": ["what's the weather", "how's the weather", "forecast"],
                "keywords": ["temperature", "rain", "sunny", "cloudy", "humidity"]
            },
            "screenshot": {
                "patterns": ["take a screenshot", "capture screen", "snap screen"],
                "keywords": ["screenshot", "capture", "screen"]
            }
        })
        
        self.confidence_thresholds = {
            "create_task": 0.6,
            "modify_task": 0.7,
            "delete_task": 0.7,
            "query_tasks": 0.6,
            "task_status": 0.6,
            "greeting": 0.5,
            "farewell": 0.5,
            "weather_query": 0.6,
            "screenshot": 0.7,
            "launch_app": 0.6,
            "list_files": 0.6,
            "system_info": 0.6,
            "search_files": 0.7,
            "navigate": 0.6
        }

class CommandProcessor:
    def __init__(self):
        self.command_patterns = {
            'open': [
                r'open\s+(?P<app>[\w\s]+)',
                r'launch\s+(?P<app>[\w\s]+)',
                r'start\s+(?P<app>[\w\s]+)',
                r'run\s+(?P<app>[\w\s]+)'
            ],
            'close': [
                r'close\s+(?P<app>[\w\s]+)',
                r'exit\s+(?P<app>[\w\s]+)',
                r'quit\s+(?P<app>[\w\s]+)',
                r'terminate\s+(?P<app>[\w\s]+)'
            ],
            'list': [
                r'what(?:\'s)?\s+in\s+(?:my\s+)?(?P<location>[\w\s]+)\s+folder',
                r'show\s+(?:me\s+)?(?:my\s+)?(?P<location>[\w\s]+)\s+(?:folder|directory)',
                r'list\s+(?:my\s+)?(?P<location>[\w\s]+)\s+(?:folder|directory)',
                r'contents\s+of\s+(?:my\s+)?(?P<location>[\w\s]+)\s+(?:folder|directory)'
            ]
        }
        
        # Compile all patterns
        self.compiled_patterns = {
            command: [re.compile(pattern, re.IGNORECASE) 
                     for pattern in patterns]
            for command, patterns in self.command_patterns.items()
        }
    
    def process_command(self, text: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        Process a command and return command type, target, and confidence.
        
        Returns:
            Tuple of (command_type, target, confidence)
        """
        text = text.lower().strip()
        
        best_match = (None, None, 0.0)  # command_type, target, confidence
        
        for command_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.match(text)
                if match:
                    # Calculate confidence based on match length and specificity
                    match_length = match.end() - match.start()
                    specificity = len(pattern.pattern)
                    confidence = (match_length / len(text)) * (specificity / 100)
                    
                    # Get target from named group
                    target = None
                    for group_name in ['app', 'location']:
                        if group_name in match.groupdict():
                            target = match.group(group_name).strip()
                            break
                    
                    if confidence > best_match[2]:
                        best_match = (command_type, target, confidence)
                        
        return best_match

class EnhancedNLU:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.intent_classifier = IntentClassifier()
        self.datetime_extractor = DateTimeExtractor()
        self.command_processor = CommandProcessor()
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        
        # Initialize task-specific components
        self.task_categories = {str(cat).lower(): cat for cat in TaskCategory}
        self.task_priorities = {str(pri).lower(): pri for pri in TaskPriority}
        
        self._setup_matchers()

    def _setup_matchers(self):
        """Setup pattern matchers for entity recognition."""
        # Add patterns for task categories
        for category in self.task_categories:
            self.phrase_matcher.add(f"CATEGORY_{category}", [self.nlp(category)])
        
        # Add patterns for priorities
        for priority in self.task_priorities:
            self.phrase
            self.phrase_matcher.add(f"PRIORITY_{priority}", [self.nlp(priority)])
        
        # Add patterns for dependencies
        self.matcher.add("DEPENDENCY", [
            [{"LOWER": "after"}, {"OP": "?"}, {"ENT_TYPE": "TASK"}],
            [{"LOWER": "before"}, {"OP": "?"}, {"ENT_TYPE": "TASK"}],
            [{"LOWER": "depends"}, {"LOWER": "on"}, {"ENT_TYPE": "TASK"}],
        ])

    @error_handler.with_retry("process_text", retries=2)
    def process_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text input and extract structured information.
        
        Args:
            text: Input text to process
            context: Optional context from previous interactions
            
        Returns:
            Dict containing extracted information
        """
        doc = self.nlp(text.lower())
        
        # Extract intents
        intents = self._extract_intents(doc)
        
        # Extract datetime information
        datetime_info = self.datetime_extractor.extract_datetime(text)
        
        # Extract task-specific information
        task_info = self._extract_task_info(doc)
        
        # Try to process as a command first
        command_type, target, cmd_confidence = self.command_processor.process_command(text)
        if cmd_confidence > 0.6:
            command_info = {
                "type": command_type,
                "target": target,
                "confidence": cmd_confidence
            }
        else:
            command_info = None
        
        # Combine all extracted information
        result = {
            "text": text,
            "intents": intents,
            "datetime": datetime_info,
            "task_info": task_info,
            "command": command_info,
            "entities": self._extract_entities(doc),
            "context": context
        }
        
        # Calculate confidence scores
        result["confidence"] = self._calculate_confidence(result)
        
        return result

    def _extract_intents(self, doc: Doc) -> List[Dict[str, float]]:
        """Extract intents with confidence scores."""
        intents = []
        
        # Check each intent
        for intent_name, intent_data in self.intent_classifier.intents.items():
            confidence = 0.0
            
            # Check for exact pattern matches
            for pattern in intent_data["patterns"]:
                if pattern in doc.text.lower():
                    confidence = max(confidence, 0.8)
            
            # Check for keyword matches
            keyword_matches = sum(1 for keyword in intent_data["keywords"] 
                                if keyword in doc.text.lower())
            if keyword_matches:
                confidence = max(confidence, 
                               min(0.6 + (keyword_matches * 0.1), 0.9))
            
            # Use spaCy's similarity for additional confidence
            doc_vector = doc.vector
            pattern_vectors = [self.nlp(pattern).vector 
                             for pattern in intent_data["patterns"]]
            if pattern_vectors:
                max_similarity = max(
                    cosine_similarity(
                        [doc_vector], 
                        [pattern_vector]
                    )[0][0] 
                    for pattern_vector in pattern_vectors
                )
                confidence = max(confidence, float(max_similarity))
            
            if confidence >= self.intent_classifier.confidence_thresholds[intent_name]:
                intents.append({
                    "intent": intent_name,
                    "confidence": confidence
                })
        
        return sorted(intents, key=lambda x: x["confidence"], reverse=True)

    def _extract_task_info(self, doc: Doc) -> Dict[str, Any]:
        """Extract task-related information."""
        task_info = {
            "category": None,
            "priority": None,
            "dependencies": [],
            "description": None
        }
        
        # Extract category and priority using phrase matcher
        matches = self.phrase_matcher(doc)
        for match_id, start, end in matches:
            match_type = self.nlp.vocab.strings[match_id]
            if match_type.startswith("CATEGORY_"):
                category = match_type.split("_")[1]
                task_info["category"] = self.task_categories[category]
            elif match_type.startswith("PRIORITY_"):
                priority = match_type.split("_")[1]
                task_info["priority"] = self.task_priorities[priority]
        
        # Extract dependencies
        for match_id, start, end in self.matcher(doc):
            match_type = self.nlp.vocab.strings[match_id]
            if match_type == "DEPENDENCY":
                dep_span = doc[start:end]
                task_info["dependencies"].append({
                    "type": "requires" if "after" in dep_span.text else "blocks",
                    "task": dep_span.text
                })
        
        # Extract description (text minus the matched patterns)
        used_spans = set()
        for match_id, start, end in matches:
            used_spans.add((start, end))
        for match_id, start, end in self.matcher(doc):
            used_spans.add((start, end))
        
        # Create description from unused tokens
        description_tokens = []
        for i, token in enumerate(doc):
            if not any(start <= i < end for start, end in used_spans):
                description_tokens.append(token.text)
        
        if description_tokens:
            task_info["description"] = " ".join(description_tokens).strip()
        
        return task_info

    def _extract_entities(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract named entities and custom entities."""
        entities = []
        
        # Extract spaCy's built-in entities
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Extract custom entities
        custom_entities = self._extract_custom_entities(doc)
        entities.extend(custom_entities)
        
        return entities

    def _extract_custom_entities(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract custom entities specific to our domain."""
        custom_entities = []
        
        # Pattern for extracting time durations
        duration_pattern = re.compile(
            r'(\d+)\s*(minute|hour|day|week|month)s?',
            re.IGNORECASE
        )
        
        # Find all duration mentions
        for match in duration_pattern.finditer(doc.text):
            custom_entities.append({
                "text": match.group(0),
                "label": "DURATION",
                "start": match.start(),
                "end": match.end(),
                "value": {
                    "amount": int(match.group(1)),
                    "unit": match.group(2).lower()
                }
            })
        
        # Extract task references
        task_references = self._extract_task_references(doc)
        custom_entities.extend(task_references)
        
        return custom_entities

    def _extract_task_references(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract references to tasks."""
        references = []
        
        # Pattern for task references
        task_patterns = [
            [{"LOWER": "that"}, {"LOWER": "task"}],
            [{"LOWER": "the"}, {"OP": "*"}, {"LOWER": "task"}],
            [{"LOWER": "my"}, {"OP": "*"}, {"LOWER": "task"}],
            [{"LOWER": "the"}, {"OP": "*"}, {"LOWER": "reminder"}],
        ]
        
        matcher = Matcher(self.nlp.vocab)
        matcher.add("TASK_REFERENCE", task_patterns)
        
        matches = matcher(doc)
        for match_id, start, end in matches:
            references.append({
                "text": doc[start:end].text,
                "label": "TASK_REFERENCE",
                "start": doc[start].idx,
                "end": doc[end-1].idx + len(doc[end-1])
            })
        
        return references

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the extraction."""
        confidence_scores = []
        
        # Intent confidence
        if result["intents"]:
            confidence_scores.append(result["intents"][0]["confidence"])
        
        # Command confidence
        if result["command"]:
            confidence_scores.append(result["command"]["confidence"])
        
        # DateTime confidence
        if result["datetime"]["due_date"] or result["datetime"]["recurring"]:
            confidence_scores.append(0.8)
        
        # Task info confidence
        task_info = result["task_info"]
        if task_info["description"]:
            confidence_scores.append(0.7)
        if task_info["category"]:
            confidence_scores.append(0.9)
        if task_info["priority"]:
            confidence_scores.append(0.9)
        
        # Entity confidence
        if result["entities"]:
            confidence_scores.append(0.8)
        
        # Calculate weighted average
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        return 0.0

    def resolve_references(self, text: str, context: Dict[str, Any]) -> str:
        """Resolve references like 'it', 'that task', etc."""
        doc = self.nlp(text)
        
        # Replace pronouns if we have context
        if "current_task" in context:
            pronoun_pattern = re.compile(r'\b(it|that|this)\b', re.IGNORECASE)
            text = pronoun_pattern.sub(context["current_task"], text)
        
        return text

def process_input(text: str) -> str:
    """Process user input with improved understanding."""
    try:
        # First try to process as a command
        command_type, target, confidence = command_processor.process_command(text)
        
        if confidence > 0.6:  # Confidence threshold
            if command_type == 'open':
                success, message = system_controller.launch_application(target)
                return message
                
            elif command_type == 'close':
                success, message = system_controller.control_application('close', target)
                return message
                
            elif command_type == 'list':
                return system_controller.list_directory(target)
        
        # If not a command or low confidence, pass to dialogue manager
        # Import here to avoid circular import
        from dialogue_manager import dialogue_manager
        return dialogue_manager.process_input(text)
        
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        return "I encountered an error processing your request. Could you please rephrase it?"

# Create global instances
command_processor = CommandProcessor()
nlu_engine = EnhancedNLU()

def process_nlu(text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process input text and return structured information."""
    try:
        # Resolve any references using context
        if context:
            text = nlu_engine.resolve_references(text, context)
        
        # Process the text
        result = nlu_engine.process_text(text, context)
        
        # Log the processing result
        logger.debug(f"NLU Processing Result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        return {
            "error": str(e),
            "text": text,
            "intents": [],
            "datetime": {},
            "task_info": {},
            "command": None,
            "entities": [],
            "confidence": 0.0
        }

# Update __all__ to expose the new functionality
__all__ = [
    'nlu_engine',
    'command_processor',
    'process_nlu',
    'process_input',
    'DateTimeExtractor',
    'IntentClassifier',
    'CommandProcessor',
    'EnhancedNLU'
]

if __name__ == "__main__":
    # Test all functionality
    test_inputs = [
        "remind me to call John tomorrow at 3pm",
        "schedule a high priority meeting for next Monday at 10am",
        "open calculator",
        "what's in my downloads folder",
        "close notepad",
        "create a shopping task for groceries due in 2 days"
    ]
    
    print("\nTesting NLU Engine:")
    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        result = process_nlu(test_input)
        print("Result:")
        print(f"Intents: {result['intents']}")
        print(f"DateTime: {result['datetime']}")
        print(f"Task Info: {result['task_info']}")
        print(f"Command: {result['command']}")
        print(f"Confidence: {result['confidence']}")