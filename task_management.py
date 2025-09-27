import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum, auto
import sqlite3
import json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from logging_config import logger
from config import get_config
from error_recovery import error_handler
from collections import defaultdict
from monitoring import monitor_performance

class TaskCategory(Enum):
    WORK = "work"
    PERSONAL = "personal"
    SHOPPING = "shopping"
    HEALTH = "health"
    BILLS = "bills"
    OTHER = "other"

class TaskPriority(Enum):
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskDependency:
    task_id: str
    dependency_type: str  # 'blocks', 'requires', 'suggests'
    condition: Optional[str] = None  # Optional condition that must be met

@dataclass
class TaskNotification:
    task_id: str
    message: str
    notification_time: datetime
    notification_type: str  # 'reminder', 'due_soon', 'overdue'
    delivered: bool = False

class EnhancedTaskScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, List[TaskDependency]] = defaultdict(list)
        self.notifications: Dict[str, List[TaskNotification]] = defaultdict(list)
        self.categories: Dict[TaskCategory, Set[str]] = defaultdict(set)
        self._initialize_scheduler()

    def _initialize_scheduler(self):
        """Initialize the scheduler with error handling."""
        try:
            self.scheduler.start()
            # Add maintenance job to clean up completed tasks
            self.scheduler.add_job(
                self._cleanup_old_tasks,
                IntervalTrigger(hours=24),
                id='maintenance_cleanup'
            )
            logger.info("Task scheduler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
            raise

    def get_tasks(self, **filters) -> List[Dict[str, Any]]:
        """
        Get tasks matching the given filters.
        
        Args:
            **filters: Arbitrary filters to apply (e.g., status=TaskStatus.PENDING)
            
        Returns:
            List[Dict[str, Any]]: List of matching tasks
        """
        try:
            tasks = []
            for task_id, task in self.tasks.items():
                if self._matches_filters(task, filters):
                    tasks.append(task)
            return tasks
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            raise

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific task by ID.
        
        Args:
            task_id: The ID of the task to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: The task info if found, None otherwise
        """
        try:
            return self.tasks.get(task_id)
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return None

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all pending tasks.
        
        Returns:
            List[Dict[str, Any]]: List of pending tasks
        """
        return self.get_tasks(status=TaskStatus.PENDING)

    def _matches_filters(self, task: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if a task matches all given filters.
        
        Args:
            task: The task to check
            filters: Dictionary of filter conditions
            
        Returns:
            bool: True if task matches all filters, False otherwise
        """
        try:
            for key, value in filters.items():
                task_value = task.get(key)
                
                # Handle special cases for enum comparisons
                if isinstance(value, Enum) and isinstance(task_value, Enum):
                    if task_value != value:
                        return False
                # Handle datetime comparisons
                elif isinstance(value, datetime) and isinstance(task_value, datetime):
                    if task_value != value:
                        return False
                # Regular comparison
                elif task_value != value:
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error matching filters: {e}")
            return False

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a task with new values.
        
        Args:
            task_id: The ID of the task to update
            updates: Dictionary of fields to update
            
        Returns:
            Optional[Dict[str, Any]]: Updated task info if successful, None otherwise
        """
        try:
            if task_id not in self.tasks:
                return None

            task = self.tasks[task_id]
            
            # Update fields
            for key, value in updates.items():
                if key in task:
                    task[key] = value
            
            task['last_updated'] = datetime.now()
            
            # Reschedule if due_date changed
            if 'due_date' in updates:
                self._schedule_task(task_id, updates['due_date'], task.get('recurring'))
            
            return task
            
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
            return None

    async def cancel_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Cancel a task.
        
        Args:
            task_id: The ID of the task to cancel
            
        Returns:
            Optional[Dict[str, Any]]: Cancelled task info if successful, None otherwise
        """
        try:
            if task_id not in self.tasks:
                return None

            task = self.tasks[task_id]
            task['status'] = TaskStatus.CANCELLED
            task['completed_at'] = datetime.now()
            
            # Remove scheduled jobs
            try:
                self.scheduler.remove_job(f"task_{task_id}")
            except:
                pass

            return task
            
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return None

    def __len__(self) -> int:
        """Return the total number of tasks."""
        return len(self.tasks)

    def __bool__(self) -> bool:
        """Return True if there are any tasks."""
        return bool(self.tasks)

    async def send_notification(self, message: str, notification_type: str) -> None:
        """Send a notification without direct speech dependency."""
        try:
            logger.info(f"Task notification ({notification_type}): {message}")
            # Instead of directly calling speak, emit an event or use a callback
            if hasattr(self, 'notification_callback'):
                await self.notification_callback(message)
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    def set_notification_callback(self, callback):
        """Set the callback for notifications."""
        self.notification_callback = callback

    @monitor_performance("add_task")
    async def add_task(self,
                      description: str,
                      due_date: datetime,
                      category: TaskCategory = TaskCategory.OTHER,
                      priority: TaskPriority = TaskPriority.MEDIUM,
                      recurring: Optional[str] = None,
                      dependencies: Optional[List[TaskDependency]] = None,
                      remind_before: Optional[List[timedelta]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new task with enhanced features.
        
        Args:
            description: Task description
            due_date: When the task is due
            category: Task category (work, personal, etc.)
            priority: Task priority level
            recurring: Cron expression for recurring tasks
            dependencies: List of task dependencies
            remind_before: List of timedeltas for multiple reminders
            metadata: Additional task metadata
        
        Returns:
            str: Task ID
        """
        task_id = str(uuid4())
        
        task_info = {
            'id': task_id,
            'description': description,
            'due_date': due_date,
            'category': category,
            'priority': priority,
            'status': TaskStatus.PENDING,
            'recurring': recurring,
            'created_at': datetime.now(),
            'metadata': metadata or {},
            'completion_percentage': 0,
            'last_updated': datetime.now()
        }
        
        try:
            # Store the task
            self.tasks[task_id] = task_info
            self.categories[category].add(task_id)
            
            # Store dependencies if any
            if dependencies:
                self.dependencies[task_id].extend(dependencies)
                # Check for circular dependencies
                if self._has_circular_dependencies(task_id):
                    raise ValueError("Circular dependency detected")
            
            # Schedule the main task
            self._schedule_task(task_id, due_date, recurring)
            
            # Schedule reminders
            if remind_before:
                self._schedule_reminders(task_id, due_date, remind_before)
            
            logger.info(f"Added task: {description} (ID: {task_id})")
            return task_id
            
        except Exception as e:
            logger.error(f"Error adding task: {e}")
            # Cleanup in case of failure
            self._cleanup_failed_task(task_id)
            raise

    def _schedule_task(self, task_id: str, due_date: datetime, recurring: Optional[str] = None):
        """Schedule a task with the job scheduler."""
        if recurring:
            trigger = CronTrigger.from_crontab(recurring)
        else:
            trigger = DateTrigger(run_date=due_date)
            
        self.scheduler.add_job(
            self.execute_task,
            trigger=trigger,
            args=[task_id],
            id=f"task_{task_id}",
            replace_existing=True
        )

    def _schedule_reminders(self, task_id: str, due_date: datetime, remind_before: List[timedelta]):
        """Schedule multiple reminders for a task."""
        for reminder_delta in remind_before:
            reminder_time = due_date - reminder_delta
            if reminder_time > datetime.now():
                notification = TaskNotification(
                    task_id=task_id,
                    message=f"Reminder: {self.tasks[task_id]['description']} is due in {self._format_timedelta(reminder_delta)}",
                    notification_time=reminder_time,
                    notification_type='reminder'
                )
                self.notifications[task_id].append(notification)
                
                self.scheduler.add_job(
                    self.send_reminder,
                    'date',
                    run_date=reminder_time,
                    args=[task_id, notification],
                    id=f"reminder_{task_id}_{reminder_delta.total_seconds()}",
                    replace_existing=True
                )

    @monitor_performance("execute_task")
    async def execute_task(self, task_id: str) -> None:
        """Execute a scheduled task with dependency checking."""
        try:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return
                
            # Check dependencies
            if not await self._check_dependencies(task_id):
                task['status'] = TaskStatus.BLOCKED
                self._schedule_dependency_check(task_id)
                return
                
            task['status'] = TaskStatus.IN_PROGRESS
            logger.info(f"Executing task: {task['description']}")
            
            # Send notification
            await self.send_notification(
                f"Time for your task: {task['description']}",
                'task_execution'
            )
            
            task['status'] = TaskStatus.COMPLETED
            task['completed_at'] = datetime.now()
            
            # Handle recurring tasks
            if task['recurring']:
                await self._schedule_next_occurrence(task)
                
            # Update dependent tasks
            await self._update_dependent_tasks(task_id)
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            if task_id in self.tasks:
                self.tasks[task_id]['status'] = TaskStatus.FAILED
                self.tasks[task_id]['error'] = str(e)

    async def _check_dependencies(self, task_id: str) -> bool:
        """Check if all dependencies are met for a task."""
        if task_id not in self.dependencies:
            return True
            
        for dep in self.dependencies[task_id]:
            if dep.dependency_type == 'blocks':
                if self.tasks[dep.task_id]['status'] != TaskStatus.COMPLETED:
                    return False
            elif dep.dependency_type == 'requires':
                if dep.condition:
                    # Evaluate condition
                    if not await self._evaluate_condition(dep.condition, dep.task_id):
                        return False
        return True

    def _has_circular_dependencies(self, task_id: str, visited: Set[str] = None) -> bool:
        """Check for circular dependencies using DFS."""
        if visited is None:
            visited = set()
            
        if task_id in visited:
            return True
            
        visited.add(task_id)
        
        for dep in self.dependencies.get(task_id, []):
            if self._has_circular_dependencies(dep.task_id, visited):
                return True
                
        visited.remove(task_id)
        return False

    async def update_task_progress(self, task_id: str, progress: int) -> None:
        """Update the completion percentage of a task."""
        if task_id in self.tasks:
            self.tasks[task_id]['completion_percentage'] = min(max(progress, 0), 100)
            self.tasks[task_id]['last_updated'] = datetime.now()
            
            if progress == 100:
                await self.complete_task(task_id)

    async def complete_task(self, task_id: str) -> None:
        """Mark a task as completed and handle dependencies."""
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = TaskStatus.COMPLETED
            self.tasks[task_id]['completed_at'] = datetime.now()
            
            # Update dependent tasks
            await self._update_dependent_tasks(task_id)

    def get_tasks_by_category(self, category: TaskCategory) -> List[Dict[str, Any]]:
        """Get all tasks in a specific category."""
        return [self.tasks[task_id] for task_id in self.categories[category]]

    def get_overdue_tasks(self) -> List[Dict[str, Any]]:
        """Get all overdue tasks."""
        now = datetime.now()
        return [
            task for task in self.tasks.values()
            if task['status'] == TaskStatus.PENDING
            and task['due_date'] < now
        ]

    def _cleanup_failed_task(self, task_id: str) -> None:
        """Clean up a failed task and its schedules."""
        try:
            if task_id in self.tasks:
                del self.tasks[task_id]
            for category in self.categories.values():
                category.discard(task_id)
            if task_id in self.dependencies:
                del self.dependencies[task_id]
            if task_id in self.notifications:
                del self.notifications[task_id]
                
            # Remove scheduled jobs
            try:
                self.scheduler.remove_job(f"task_{task_id}")
            except:
                pass
                
            # Remove reminder jobs
            for job in self.scheduler.get_jobs():
                if job.id.startswith(f"reminder_{task_id}"):
                    try:
                        self.scheduler.remove_job(job.id)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error cleaning up failed task {task_id}: {e}")

    async def _cleanup_old_tasks(self) -> None:
        """Clean up old completed tasks."""
        try:
            current_time = datetime.now()
            threshold = current_time - timedelta(days=30)  # Keep completed tasks for 30 days
            
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if (task['status'] in [TaskStatus.COMPLETED, TaskStatus.CANCELLED] and
                    task.get('completed_at', current_time) < threshold):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                self._cleanup_failed_task(task_id)
                
            logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
        except Exception as e:
            logger.error(f"Error in task cleanup: {e}")

    @staticmethod
    def _format_timedelta(delta: timedelta) -> str:
        """Format a timedelta into a human-readable string."""
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        
        parts = []
        if days:
            parts.append(f"{days} {'day' if days == 1 else 'days'}")
        if hours:
            parts.append(f"{hours} {'hour' if hours == 1 else 'hours'}")
        if minutes:
            parts.append(f"{minutes} {'minute' if minutes == 1 else 'minutes'}")
            
        if not parts:
            return "less than a minute"
            
        return ", ".join(parts)

# Create a global instance of EnhancedTaskScheduler
task_scheduler = EnhancedTaskScheduler()

# Example usage
if __name__ == "__main__":
    async def test_task_scheduler():
        # Create a work task with multiple reminders
        work_task_id = await task_scheduler.add_task(
            description="Complete project presentation",
            due_date=datetime.now() + timedelta(days=2),
            category=TaskCategory.WORK,
            priority=TaskPriority.HIGH,
            remind_before=[
                timedelta(days=1),
                timedelta(hours=4),
                timedelta(hours=1)
            ],
            metadata={"project": "Q4 Review"}
        )
        
        # Create a dependent task
        dependency = TaskDependency(work_task_id, "requires")
        dependent_task_id = await task_scheduler.add_task(
            description="Send presentation to team",
            due_date=datetime.now() + timedelta(days=2, hours=1),
            category=TaskCategory.WORK,
            priority=TaskPriority.MEDIUM,
            dependencies=[dependency]
        )
        
        # Update task progress
        await task_scheduler.update_task_progress(work_task_id, 50)
        
        # Get tasks by category
        work_tasks = task_scheduler.get_tasks_by_category(TaskCategory.WORK)
        print(f"Work tasks: {work_tasks}")
        
        # Get overdue tasks
        overdue = task_scheduler.get_overdue_tasks()
        print(f"Overdue tasks: {overdue}")

    asyncio.run(test_task_scheduler())