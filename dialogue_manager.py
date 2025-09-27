import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple, Callable, Union
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
import json
import threading
from queue import Queue
from collections import defaultdict
from task_management import TaskCategory, TaskPriority, TaskStatus
from logging_config import logger
from error_recovery import error_handler
from config import get_config
from system_control import system_controller
from base_cleanup import base_cleanup_manager
from events import event_system

# Get configuration
config = get_config()

class DialogueState(Enum):
    IDLE = auto()
    EXECUTING_TASK = auto()
    AWAITING_CONFIRMATION = auto()
    COLLECTING_INFO = auto()
    ERROR_RECOVERY = auto()

@dataclass
class RequiredTaskInfo:
    description: bool = False
    due_date: bool = False
    category: bool = False
    priority: bool = False
    dependencies: List[str] = field(default_factory=list)
    notification_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskDependency:
    task_id: str
    dependency_type: str  # 'blocks', 'requires', 'suggests'
    condition: Optional[str] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DialogueContext:
    state: DialogueState = field(default=DialogueState.IDLE)
    current_task: Optional[str] = None
    task_result: Any = None
    missing_info: RequiredTaskInfo = field(default_factory=RequiredTaskInfo)
    collected_info: Dict[str, Any] = field(default_factory=dict)
    last_intent: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    active_tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending_notifications: Queue = field(default_factory=Queue)
    error_context: Optional[Dict[str, Any]] = None

class DialogueManager:
    def __init__(self):
        # Core components
        self._state_lock = threading.RLock()
        self._context_lock = threading.RLock()
        self._context = DialogueContext()
        self._task_handlers = {}
        self._nlu_engine = None
        self._memory_system = None
        self._scheduler = None
        self._max_history = 10

        # Initialize command and task handlers
        self._command_handlers = {
            'open': self._handle_open_command,
            'close': self._handle_close_command,
            'list': self._handle_list_command,
            'search': self._handle_search_command,
            'navigate': self._handle_navigate_command,
            'system_info': self._handle_system_info_command,
            'compound': self._handle_compound_command
        }

        # Task-specific settings
        self.task_timeout = timedelta(minutes=30)
        self.notification_interval = timedelta(minutes=5)
        self._pending_tasks = {}
        self._task_dependencies = defaultdict(list)
        self._task_notifications = defaultdict(list)

        # Error recovery settings
        self._max_retries = 3
        self._retry_delay = 1.0
        self._error_handlers = defaultdict(list)

        logger.info(f"DialogueManager initialized with ASSISTANT_NAME: {config.ASSISTANT_NAME}")
        
        # Start background task processing
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Initialize and start background task processing."""
        self._task_processor = threading.Thread(
            target=self._process_task_queue,
            daemon=True
        )
        self._notification_processor = threading.Thread(
            target=self._process_notifications,
            daemon=True
        )
        
        self._task_processor.start()
        self._notification_processor.start()

    def _lazy_init(self):
        """Lazy initialization of components to avoid circular imports."""
        if self._nlu_engine is None:
            from advanced_nlu import nlu_engine
            self._nlu_engine = nlu_engine

        if self._memory_system is None:
            from memory import memory_system
            self._memory_system = memory_system

        if self._scheduler is None:
            from task_management import task_scheduler
            self._scheduler = task_scheduler

    async def process_input(self, text: str) -> str:
        """
        Process user input with comprehensive error handling and state management.
        
        Args:
            text: User input text
            
        Returns:
            str: Response to the user
        """
        try:
            # Initialize components if needed
            self._lazy_init()
            
            # Validate and sanitize input
            sanitized_input = self._sanitize_input(text)
            if not sanitized_input:
                return "I didn't receive any input. Could you please try again?"

            # Get current context
            context = self._get_context_dict()
            
            # Process with NLU
            nlu_result = await self._process_with_nlu(sanitized_input, context)
            
            # Handle based on current state
            response = await self._process_by_state(nlu_result)

            # Update conversation history and memory
            await self._update_interaction_records(sanitized_input, response, nlu_result)

            return response

        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            self.state = DialogueState.ERROR_RECOVERY
            return await self._handle_error("process_input", e)

    def _sanitize_input(self, text: str) -> str:
        """Sanitize and validate user input."""
        if not text:
            return ""
            
        # Remove potential harmful characters
        sanitized = "".join(char for char in text if char.isprintable())
        
        # Enforce maximum length
        max_length = 1000  # Reasonable maximum
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            
        return sanitized.strip()

    async def _process_with_nlu(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with NLU system with error handling."""
        try:
            # Basic NLU processing
            nlu_result = self._nlu_engine.process_text(text, context)
            
            # Enhance with datetime processing if needed
            if self._needs_datetime_processing(nlu_result):
                nlu_result = await self._enhance_with_datetime(nlu_result)
                
            # Add command processing
            if self._might_be_command(text):
                command_result = await self._process_command_intent(text)
                if command_result:
                    nlu_result['command'] = command_result
                    
            # Add compound command processing
            if 'and' in text.lower():
                compound_result = await self._process_compound_command(text)
                if compound_result:
                    nlu_result['compound_commands'] = compound_result
                    
            return nlu_result
            
        except Exception as e:
            logger.error(f"Error in NLU processing: {e}")
            return {
                'text': text,
                'error': str(e),
                'intents': [],
                'entities': [],
                'confidence': 0.0
            }

    async def _process_by_state(self, nlu_result: Dict[str, Any]) -> str:
        """Process input based on current dialogue state."""
        current_state = self.state
        
        try:
            if current_state == DialogueState.ERROR_RECOVERY:
                return await self._handle_error_recovery_state(nlu_result)
                
            elif current_state == DialogueState.COLLECTING_INFO:
                return await self._handle_info_collection_state(nlu_result)
                
            elif current_state == DialogueState.AWAITING_CONFIRMATION:
                return await self._handle_confirmation_state(nlu_result)
                
            elif current_state == DialogueState.EXECUTING_TASK:
                return await self._handle_executing_task_state(nlu_result)
                
            else:  # IDLE state
                return await self._handle_idle_state(nlu_result)
                
        except Exception as e:
            logger.error(f"Error processing state {current_state}: {e}")
            self.state = DialogueState.ERROR_RECOVERY
            return await self._handle_error(f"process_state_{current_state.name}", e)

    async def _handle_error_recovery_state(self, nlu_result: Dict[str, Any]) -> str:
        """Handle input while in error recovery state."""
        if self._can_recover_from_error(nlu_result):
            self.state = DialogueState.IDLE
            return await self._process_normal_input(nlu_result)
        else:
            retry_count = self._context.error_context.get('retry_count', 0) + 1
            if retry_count < self._max_retries:
                self._context.error_context['retry_count'] = retry_count
                return f"I'm still having trouble understanding. Could you try rephrasing? (Attempt {retry_count}/{self._max_retries})"
            else:
                self.state = DialogueState.IDLE
                self._context.error_context = None
                return "I'm sorry, but I'm unable to process this request. Let's try something else."

    async def _handle_info_collection_state(self, nlu_result: Dict[str, Any]) -> str:
        """Handle input while collecting required information."""
        try:
            # Update collected information
            self._update_collected_info(nlu_result)
            
            # Check if we have all required information
            if not any(asdict(self._context.missing_info).values()):
                # Execute task with collected information
                return await self._execute_task_with_info(nlu_result)
            
            # Generate request for next piece of missing information
            return self._generate_info_request()
            
        except Exception as e:
            logger.error(f"Error collecting information: {e}")
            return await self._handle_error("info_collection", e)

    async def _handle_confirmation_state(self, nlu_result: Dict[str, Any]) -> str:
        """Handle input while awaiting confirmation."""
        try:
            confirmation = self._extract_confirmation(nlu_result)
            
            if confirmation is True:
                return await self._execute_confirmed_action()
            elif confirmation is False:
                self.state = DialogueState.IDLE
                return "Alright, I've cancelled that. What else can I help you with?"
            else:
                return "I didn't quite understand. Please say 'yes' to confirm or 'no' to cancel."
                
        except Exception as e:
            logger.error(f"Error in confirmation handling: {e}")
            return await self._handle_error("confirmation", e)

    def _extract_confirmation(self, nlu_result: Dict[str, Any]) -> Optional[bool]:
        """Extract confirmation intent from NLU result."""
        text = nlu_result['text'].lower()
        
        confirmation_keywords = {'yes', 'yeah', 'sure', 'okay', 'confirm', 'correct'}
        denial_keywords = {'no', 'nope', 'cancel', 'wrong', 'incorrect'}
        
        if any(keyword in text for keyword in confirmation_keywords):
            return True
        elif any(keyword in text for keyword in denial_keywords):
            return False
            
        return None

    async def _execute_confirmed_action(self) -> str:
        """Execute the action that was awaiting confirmation."""
        try:
            if not self._context.current_task:
                raise ValueError("No task waiting for confirmation")
                
            task_info = self._context.collected_info.copy()
            handler = self._task_handlers.get(self._context.current_task)
            
            if not handler:
                raise ValueError(f"No handler for task: {self._context.current_task}")
                
            self.state = DialogueState.EXECUTING_TASK
            result = await handler(task_info)
            
            self.state = DialogueState.IDLE
            self._clear_task_context()
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing confirmed action: {e}")
            return await self._handle_error("execute_confirmed", e)

    async def _execute_task_with_info(self, nlu_result: Dict[str, Any]) -> str:
        """Execute task with collected information."""
        try:
            task_info = self._context.collected_info.copy()
            task_info.update(nlu_result.get('task_info', {}))
            
            handler = self._task_handlers.get(self._context.current_task)
            if not handler:
                raise ValueError(f"No handler for task: {self._context.current_task}")
                
            result = await handler(task_info)
            
            self.state = DialogueState.IDLE
            self._clear_task_context()
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return await self._handle_error("execute_task", e)

    def _clear_task_context(self):
        """Clear task-related context."""
        self._context.current_task = None
        self._context.collected_info.clear()
        self._context.missing_info = RequiredTaskInfo()
        self._context.task_result = None

    async def _process_normal_input(self, nlu_result: Dict[str, Any]) -> str:
        """Process input in normal (IDLE) state."""
        try:
            # Check for commands first
            if nlu_result.get('command') and nlu_result['command']['confidence'] > 0.6:
                return await self._handle_command(
                    nlu_result['command']['type'],
                    nlu_result['command']['target']
                )

            # Check for compound commands
            if nlu_result.get('compound_commands'):
                return await self._handle_compound_command(nlu_result['compound_commands'])

            # Check for intents
            if nlu_result['intents']:
                primary_intent = nlu_result['intents'][0]
                if primary_intent['confidence'] > 0.6:
                    return await self._handle_intent(primary_intent['intent'], nlu_result)

            # Default to general conversation
            return await self._handle_general_conversation(nlu_result)

        except Exception as e:
            logger.error(f"Error in normal input processing: {e}")
            return await self._handle_error("normal_processing", e)
        
    async def _handle_command(self, command_type: str, target: str) -> str:
        """Handle system commands with full error recovery."""
        try:
            handler = self._command_handlers.get(command_type)
            if handler:
                return await handler(target)
            return f"I'm not sure how to {command_type} {target}."
        except Exception as e:
            logger.error(f"Error handling command {command_type}: {e}")
            return await self._handle_error("command_execution", e)

    async def _handle_open_command(self, target: str) -> str:
        """Handle open/launch commands with validation."""
        try:
            if not self._is_safe_application(target):
                return f"I'm sorry, but I cannot open {target} for security reasons."
                
            success, message = await system_controller.launch_application(target)
            if success:
                await self._memory_system.store_memory({
                    'type': 'command',
                    'action': 'open',
                    'target': target,
                    'result': 'success'
                })
            return message
        except Exception as e:
            logger.error(f"Error opening {target}: {e}")
            return f"I encountered an error trying to open {target}"

    async def _handle_close_command(self, target: str) -> str:
        """Handle close/exit commands with process validation."""
        try:
            if not self._is_safe_application(target):
                return f"I'm sorry, but I cannot close {target} for security reasons."
                
            success, message = await system_controller.control_application('close', target)
            if success:
                await self._memory_system.store_memory({
                    'type': 'command',
                    'action': 'close',
                    'target': target,
                    'result': 'success'
                })
            return message
        except Exception as e:
            logger.error(f"Error closing {target}: {e}")
            return f"I encountered an error trying to close {target}"

    def _is_safe_application(self, app_name: str) -> bool:
        """Validate if an application is safe to open/close."""
        unsafe_keywords = {'cmd', 'powershell', 'registry', 'regedit', 'taskmgr', 'services'}
        return not any(keyword in app_name.lower() for keyword in unsafe_keywords)

    async def _handle_compound_command(self, commands: List[Dict[str, Any]]) -> str:
        """Handle multiple commands linked with 'and'."""
        responses = []
        for command in commands:
            try:
                response = await self._handle_command(
                    command['type'],
                    command['target']
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Error in compound command: {e}")
                responses.append(f"Error executing {command['type']} {command['target']}")

        return " And ".join(responses)

    async def _process_compound_command(self, text: str) -> List[Dict[str, Any]]:
        """Process text that might contain multiple commands."""
        commands = []
        parts = text.lower().split(' and ')
        
        for part in parts:
            result = await self._process_command_intent(part)
            if result:
                commands.append(result)
                
        return commands if len(commands) > 1 else None

    async def _handle_intent(self, intent: str, nlu_result: Dict[str, Any]) -> str:
        """Handle recognized intents with full context awareness."""
        try:
            # Check for task-specific handlers first
            if intent in self._task_handlers:
                return await self._handle_task_intent(intent, nlu_result)

            # Handle built-in intents
            if intent == "greeting":
                return await self._handle_greeting(nlu_result)
            elif intent == "farewell":
                return await self._handle_farewell(nlu_result)
            elif intent == "weather_query":
                return await self._handle_weather(nlu_result)
            elif intent == "help":
                return await self._handle_help(nlu_result)
            elif intent == "status":
                return await self._handle_status(nlu_result)

            # Fall back to task processing
            return await self._process_task_intent(intent, nlu_result)

        except Exception as e:
            logger.error(f"Error handling intent {intent}: {e}")
            return await self._handle_error("intent_handling", e)

    async def _handle_task_intent(self, intent: str, nlu_result: Dict[str, Any]) -> str:
        """Handle task-related intents with proper validation."""
        try:
            # Validate required information
            if not self._has_required_info(intent, nlu_result):
                self.state = DialogueState.COLLECTING_INFO
                self._context.current_task = intent
                return self._generate_info_request()

            # Check dependencies
            if not await self._check_task_dependencies(intent, nlu_result):
                return "Some required tasks need to be completed first."

            # Execute the task
            handler = self._task_handlers[intent]
            result = await handler(nlu_result)

            # Store the result
            await self._store_task_result(intent, result, nlu_result)

            return result

        except Exception as e:
            logger.error(f"Error in task handling: {e}")
            return await self._handle_error("task_handling", e)

    async def _check_task_dependencies(self, intent: str, nlu_result: Dict[str, Any]) -> bool:
        """Check if all dependencies for a task are met."""
        dependencies = self._task_dependencies.get(intent, [])
        
        for dep in dependencies:
            if not await self._is_dependency_satisfied(dep):
                return False
                
        return True

    async def _is_dependency_satisfied(self, dependency: TaskDependency) -> bool:
        """Check if a specific dependency is satisfied."""
        try:
            if dependency.dependency_type == 'blocks':
                task = self._pending_tasks.get(dependency.task_id)
                return task and task.get('status') == 'completed'
                
            elif dependency.dependency_type == 'requires':
                if dependency.condition:
                    return await self._evaluate_condition(
                        dependency.condition,
                        dependency.task_id
                    )
                return True
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependency: {e}")
            return False

    async def _evaluate_condition(self, condition: str, task_id: str) -> bool:
        """Evaluate a dependency condition."""
        try:
            task = self._pending_tasks.get(task_id)
            if not task:
                return False
                
            # Basic condition evaluation
            if condition == 'completed':
                return task.get('status') == 'completed'
            elif condition.startswith('result_equals_'):
                expected = condition.split('_')[-1]
                return str(task.get('result')) == expected
                
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False

    async def _store_task_result(self, 
                               intent: str, 
                               result: Any, 
                               nlu_result: Dict[str, Any]) -> None:
        """Store task result with proper metadata."""
        try:
            # Store in memory system
            await self._memory_system.store_memory({
                'type': 'task_result',
                'intent': intent,
                'result': result,
                'context': nlu_result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update pending tasks
            if intent in self._pending_tasks:
                self._pending_tasks[intent].update({
                    'status': 'completed',
                    'result': result,
                    'completion_time': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error storing task result: {e}")

    def _has_required_info(self, intent: str, nlu_result: Dict[str, Any]) -> bool:
        """Check if all required information is present."""
        try:
            required_info = RequiredTaskInfo()
            
            if intent == "create_task":
                task_info = nlu_result.get('task_info', {})
                datetime_info = nlu_result.get('datetime', {})
                
                required_info.description = not bool(task_info.get('description'))
                required_info.due_date = not bool(datetime_info.get('due_date'))
                required_info.category = not bool(task_info.get('category'))
                required_info.priority = not bool(task_info.get('priority'))
                
            elif intent == "modify_task":
                required_info.description = True
                
            self._context.missing_info = required_info
            return not any(asdict(required_info).values())
            
        except Exception as e:
            logger.error(f"Error checking required info: {e}")
            return False

    def _generate_info_request(self) -> str:
        """Generate request for missing information."""
        try:
            missing = asdict(self._context.missing_info)
            
            if missing['description']:
                return "What would you like me to do?"
            elif missing['due_date']:
                return "When would you like this done?"
            elif missing['category']:
                categories = ', '.join(str(cat) for cat in TaskCategory)
                return f"What category should this be? ({categories})"
            elif missing['priority']:
                priorities = ', '.join(str(pri) for pri in TaskPriority)
                return f"What priority level? ({priorities})"
                
            return "Could you provide more details?"
            
        except Exception as e:
            logger.error(f"Error generating info request: {e}")
            return "I need more information. Could you provide more details?"

    def _update_collected_info(self, nlu_result: Dict[str, Any]) -> None:
        """Update collected information from NLU result."""
        try:
            with self._context_lock:
                # Update task info
                if 'task_info' in nlu_result:
                    self._context.collected_info.update(nlu_result['task_info'])
                
                # Update datetime info
                if 'datetime' in nlu_result:
                    self._context.collected_info.update(nlu_result['datetime'])
                
                # Update any entity information
                if 'entities' in nlu_result:
                    for entity in nlu_result['entities']:
                        if entity['label'] in ['PERSON', 'ORG', 'GPE']:
                            self._context.collected_info[entity['label']] = entity['text']
                            
        except Exception as e:
            logger.error(f"Error updating collected info: {e}")

    async def _process_task_queue(self):
        """Process queued tasks in background thread."""
        while True:
            try:
                task = await self._get_next_task()
                if task:
                    await self._execute_background_task(task)
                    
                await asyncio.sleep(1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in task queue processing: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def _execute_background_task(self, task: Dict[str, Any]) -> None:
        """Execute a background task with proper error handling."""
        try:
            task_id = task['id']
            handler = self._task_handlers.get(task['type'])
            
            if not handler:
                raise ValueError(f"No handler for task type: {task['type']}")
                
            self._pending_tasks[task_id]['status'] = 'running'
            result = await handler(task['data'])
            
            await self._handle_task_completion(task_id, result)
            
        except Exception as e:
            logger.error(f"Error executing background task: {e}")
            await self._handle_task_failure(task['id'], str(e))

    async def _handle_task_completion(self, task_id: str, result: Any) -> None:
        """Handle successful task completion."""
        try:
            self._pending_tasks[task_id].update({
                'status': 'completed',
                'result': result,
                'completion_time': datetime.now().isoformat()
            })
            
            # Notify any waiting tasks
            await self._notify_dependent_tasks(task_id)
            
            # Send completion notification if needed
            if self._should_notify_completion(task_id):
                await self._send_completion_notification(task_id, result)
                
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")

    async def _handle_task_failure(self, task_id: str, error: str) -> None:
        """Handle task failure with retry logic."""
        try:
            task = self._pending_tasks[task_id]
            retry_count = task.get('retry_count', 0)
            
            if retry_count < self._max_retries:
                # Schedule retry
                task.update({
                    'status': 'retry_pending',
                    'retry_count': retry_count + 1,
                    'last_error': error,
                    'retry_after': datetime.now() + timedelta(
                        seconds=self._retry_delay * (2 ** retry_count)
                    )
                })
            else:
                # Mark as failed
                task.update({
                    'status': 'failed',
                    'error': error,
                    'failure_time': datetime.now().isoformat()
                })
                
                await self._send_failure_notification(task_id, error)
                
        except Exception as e:
            logger.error(f"Error handling task failure: {e}")

    async def _notify_dependent_tasks(self, task_id: str) -> None:
        """Notify tasks that depend on the completed task."""
        try:
            for dep_id, dep in self._task_dependencies.items():
                if task_id in [d.task_id for d in dep]:
                    await self._check_dependency_satisfaction(dep_id)
                    
        except Exception as e:
            logger.error(f"Error notifying dependent tasks: {e}")

    async def _check_dependency_satisfaction(self, task_id: str) -> None:
        """Check if all dependencies for a task are now satisfied."""
        try:
            if await self._check_task_dependencies(task_id, {}):
                task = self._pending_tasks.get(task_id)
                if task and task['status'] == 'waiting_dependencies':
                    await self._schedule_task(task_id)
                    
        except Exception as e:
            logger.error(f"Error checking dependency satisfaction: {e}")

    async def _schedule_task(self, task_id: str) -> None:
        """Schedule a task for execution."""
        try:
            task = self._pending_tasks[task_id]
            
            # Check if task should be scheduled now or later
            if 'schedule_time' in task:
                schedule_time = datetime.fromisoformat(task['schedule_time'])
                if schedule_time > datetime.now():
                    await self._scheduler.schedule_task(
                        task_id,
                        schedule_time,
                        task
                    )
                    return
            
            # Execute immediately if no scheduling needed
            task['status'] = 'pending'
            await self._execute_background_task(task)
            
        except Exception as e:
            logger.error(f"Error scheduling task: {e}")

    async def _process_notifications(self):
        """Process pending notifications in background thread."""
        while True:
            try:
                # Process due notifications
                current_time = datetime.now()
                notifications = self._get_due_notifications(current_time)
                
                for notification in notifications:
                    await self._send_notification(notification)
                    
                await asyncio.sleep(1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error processing notifications: {e}")
                await asyncio.sleep(5)  # Back off on error

    def _get_due_notifications(self, current_time: datetime) -> List[Dict[str, Any]]:
        """Get notifications that are due for delivery."""
        try:
            due_notifications = []
            
            with self._context_lock:
                remaining_notifications = []
                
                while not self._context.pending_notifications.empty():
                    notif = self._context.pending_notifications.get_nowait()
                    
                    if notif['delivery_time'] <= current_time:
                        due_notifications.append(notif)
                    else:
                        remaining_notifications.append(notif)
                
                # Put back notifications that aren't due yet
                for notif in remaining_notifications:
                    self._context.pending_notifications.put_nowait(notif)
                    
            return due_notifications
            
        except Exception as e:
            logger.error(f"Error getting due notifications: {e}")
            return []

    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """Send a notification with proper error handling."""
        try:
            notification_type = notification.get('type', 'general')
            message = notification.get('message', '')
            
            if notification_type == 'task_completion':
                await self._handle_completion_notification(notification)
            elif notification_type == 'task_reminder':
                await self._handle_reminder_notification(notification)
            elif notification_type == 'error':
                await self._handle_error_notification(notification)
            else:
                await self._handle_general_notification(notification)
                
            # Store notification in memory
            await self._memory_system.store_memory({
                'type': 'notification',
                'notification_type': notification_type,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'metadata': notification.get('metadata', {})
            })
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    async def _handle_error(self, error_type: str, error: Exception) -> str:
        """Handle errors with proper recovery options."""
        try:
            # Log the error
            logger.error(f"Error in {error_type}: {error}", exc_info=True)
            
            # Store error context
            self._context.error_context = {
                'type': error_type,
                'error': str(error),
                'timestamp': datetime.now().isoformat(),
                'retry_count': 0
            }
            
            # Get appropriate error message
            message = self._get_error_message(error_type, error)
            
            # Schedule error notification if needed
            await self._schedule_error_notification(error_type, error)
            
            # Set error recovery state
            self.state = DialogueState.ERROR_RECOVERY
            
            return message
            
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            return "I encountered an error. Could you please try again?"

    def _get_error_message(self, error_type: str, error: Exception) -> str:
        """Get appropriate error message based on error type and context."""
        error_messages = {
            'process_input': "I had trouble understanding that. Could you rephrase?",
            'command_execution': "I couldn't execute that command. Want to try something else?",
            'task_handling': "I had trouble with that task. Should we try again?",
            'info_collection': "I'm having trouble collecting the information. Let's start over.",
            'confirmation': "I'm not sure about your confirmation. Please say 'yes' or 'no'.",
        }
        
        return error_messages.get(
            error_type,
            "I encountered an unexpected error. Would you like to try again?"
        )

    async def _schedule_error_notification(self, error_type: str, error: Exception) -> None:
        """Schedule error notification if needed."""
        try:
            if self._should_notify_error(error_type):
                notification = {
                    'type': 'error',
                    'error_type': error_type,
                    'message': str(error),
                    'delivery_time': datetime.now(),
                    'metadata': {
                        'state': self.state.name,
                        'current_task': self._context.current_task
                    }
                }
                
                self._context.pending_notifications.put_nowait(notification)
                
        except Exception as e:
            logger.error(f"Error scheduling error notification: {e}")

    def _should_notify_error(self, error_type: str) -> bool:
        """Determine if an error should trigger a notification."""
        critical_errors = {
            'task_failure',
            'system_error',
            'security_violation',
            'data_corruption'
        }
        
        return error_type in critical_errors

    async def _handle_general_conversation(self, nlu_result: Dict[str, Any]) -> str:
        """Handle general conversation with context awareness."""
        try:
            # Get relevant context from memory
            context = await self._get_conversation_context(nlu_result)
            
            # Import here to avoid circular import
            from ai_interface import ai_interface
            
            # Add memory context to prompt
            enhanced_prompt = self._enhance_prompt_with_context(
                nlu_result['text'],
                context
            )
            
            # Get response from AI
            response = await ai_interface.call_gpt4(enhanced_prompt)
            
            # Store conversation in memory
            await self._store_conversation(nlu_result['text'], response, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in general conversation: {e}")
            return "I'm having trouble with our conversation. Could you try again?"

    async def _get_conversation_context(self, nlu_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant context for the conversation."""
        try:
            # Get recent interactions
            recent_history = self._conversation_history[-5:]
            
            # Get relevant memories
            relevant_memories = await self._memory_system.get_relevant_memories(
                nlu_result['text'],
                limit=3
            )
            
            # Build context dictionary
            context = {
                'recent_history': recent_history,
                'relevant_memories': relevant_memories,
                'current_state': self.state.name,
                'current_task': self._context.current_task,
                'collected_info': self._context.collected_info
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {}

    def _enhance_prompt_with_context(self, text: str, context: Dict[str, Any]) -> str:
        """Enhance user prompt with relevant context."""
        try:
            enhanced_prompt = text
            
            # Add relevant historical context if available
            if context.get('relevant_memories'):
                memory_context = "\nRelevant past interactions:\n"
                for memory in context['relevant_memories']:
                    memory_context += f"- {memory['content']}\n"
                enhanced_prompt = f"{memory_context}\nCurrent query: {text}"
            
            # Add current task context if relevant
            if context.get('current_task'):
                task_context = f"\nCurrent task: {context['current_task']}"
                if context.get('collected_info'):
                    task_context += f"\nCollected information: {json.dumps(context['collected_info'], indent=2)}"
                enhanced_prompt = f"{enhanced_prompt}\n{task_context}"
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return text

    async def _store_conversation(self, 
                                user_input: str, 
                                response: str, 
                                context: Dict[str, Any]) -> None:
        """Store conversation in memory system."""
        try:
            await self._memory_system.store_memory({
                'type': 'conversation',
                'user_input': user_input,
                'assistant_response': response,
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'state': self.state.name
            })
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")

    def _get_context_dict(self) -> Dict[str, Any]:
        """Get current context as a dictionary with thread safety."""
        with self._context_lock:
            return {
                'state': self.state.name,
                'current_task': self._context.current_task,
                'conversation_history': self._conversation_history[-5:] if self._conversation_history else [],
                'collected_info': self._context.collected_info.copy(),
                'pending_tasks': list(self._pending_tasks.keys()),
                'error_context': self._context.error_context
            }

    def clear_context(self) -> None:
        """Clear conversation context with thread safety."""
        with self._context_lock:
            self._conversation_history.clear()
            self._context.collected_info.clear()
            self._context.current_task = None
            self._context.task_result = None
            self._context.error_context = None
            self.state = DialogueState.IDLE
            logger.debug("Conversation context cleared")

    async def shutdown(self) -> None:
        """Perform cleanup and shutdown."""
        try:
            # Clear all context
            self.clear_context()
            
            # Cancel any pending tasks
            for task_id in self._pending_tasks:
                await self._handle_task_failure(task_id, "System shutdown")
            
            # Clear notifications
            while not self._context.pending_notifications.empty():
                self._context.pending_notifications.get_nowait()
            
            logger.info("DialogueManager shut down successfully")
        except Exception as e:
            logger.error(f"Error during DialogueManager shutdown: {e}")

    def __str__(self) -> str:
        """String representation of DialogueManager state."""
        return (
            f"DialogueManager(state={self.state.name}, "
            f"current_task={self._context.current_task}, "
            f"pending_tasks={len(self._pending_tasks)}, "
            f"conversation_history={len(self._conversation_history)})"
        )

# Create global instance
dialogue_manager = DialogueManager()

# Register cleanup
base_cleanup_manager.add_cleanup_function(dialogue_manager.shutdown)

# Export needed symbols
__all__ = ['dialogue_manager', 'DialogueState', 'DialogueManager', 'RequiredTaskInfo']