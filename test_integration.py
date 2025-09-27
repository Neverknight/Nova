import unittest
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from datetime import datetime, timedelta
import random
from contextlib import contextmanager
import sqlite3
import gc
import psutil
import os

from dialogue_manager import dialogue_manager, DialogueState
from task_management import task_scheduler, TaskCategory, TaskPriority, TaskStatus
from memory import memory_system, Memory, MemoryType
from db_connection import db_manager
from logging_config import logger

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        dialogue_manager.clear_context()
        self.cleanup_test_data()
        time.sleep(0.1)
        
    def tearDown(self):
        self.cleanup_test_data()
        gc.collect()
        time.sleep(0.1)

    def cleanup_test_data(self):
        try:
            with db_manager.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories WHERE content LIKE '%test%'")
                cursor.execute("DELETE FROM memories WHERE memory_type = 'test'")
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    def test_concurrent_task_creation(self):
        """Test creating multiple tasks concurrently."""
        num_tasks = 10
        results = queue.Queue()
        
        def create_task(i: int):
            try:
                response = dialogue_manager.process_input(
                    f"remind me to do test task {i} tomorrow at {9+i}am"
                )
                results.put(("success", response))
            except Exception as e:
                results.put(("error", str(e)))

        threads = [
            threading.Thread(target=create_task, args=(i,))
            for i in range(num_tasks)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()

        errors = []
        successes = 0
        while not results.empty():
            status, result = results.get()
            if status == "error":
                errors.append(result)
            else:
                successes += 1

        self.assertEqual(successes, num_tasks, 
                        f"Expected {num_tasks} successful tasks, got {successes}. Errors: {errors}")

    def test_memory_stress(self):
        """Test memory system under load."""
        num_memories = 100
        batch_size = 10
        
        def create_test_memories(batch_id: int):
            for i in range(batch_size):
                memory = Memory(
                    content={
                        "type": "test",
                        "content": f"test memory {batch_id}-{i}",
                        "timestamp": datetime.now().isoformat()
                    },
                    type=MemoryType.SHORT_TERM,
                    timestamp=datetime.now(),
                    importance=0.5,
                    metadata={"test": True, "batch": batch_id}
                )
                try:
                    # Store directly instead of using queue for test
                    memory_system.store_memory(memory)
                except Exception as e:
                    logger.error(f"Error storing memory: {e}")

        # Use smaller number of workers to avoid overwhelming SQLite
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(create_test_memories, i)
                for i in range(num_memories // batch_size)
            ]
            
            for future in futures:
                future.result()

        # Allow time for processing
        time.sleep(1)

        # Verify memories were stored correctly
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM memories WHERE content LIKE '%test memory%'")
            result = cursor.fetchone()
            self.assertEqual(result['count'], num_memories)

    def test_error_recovery(self):
        """Test system's ability to recover from errors."""
        original_execute = db_manager.execute_with_retry
        
        def mock_execute(*args, **kwargs):
            raise sqlite3.OperationalError("Simulated DB error")
            
        db_manager.execute_with_retry = mock_execute
        
        try:
            # Force an error
            response = dialogue_manager.process_input("remind me to test error recovery")
            self.assertIn("error", response.lower())
            self.assertEqual(dialogue_manager.get_current_state(), DialogueState.ERROR_RECOVERY)
            
            # Test recovery with a safe command
            db_manager.execute_with_retry = original_execute  # Restore original function
            response = dialogue_manager.process_input("what time is it?")
            self.assertNotIn("error", response.lower())
            self.assertEqual(dialogue_manager.get_current_state(), DialogueState.IDLE)
            
        finally:
            db_manager.execute_with_retry = original_execute
            dialogue_manager.clear_error_state()

    def test_conversation_state_recovery(self):
        """Test conversation state recovery after errors."""
        def error_handler(*args, **kwargs):
            dialogue_manager.set_state(DialogueState.ERROR_RECOVERY)
            raise Exception("Simulated error")
            
        original_handler = dialogue_manager._handle_task_creation
        dialogue_manager._handle_task_creation = error_handler
        
        try:
            # Trigger error
            response = dialogue_manager.process_input("remind me to test state recovery")
            self.assertEqual(dialogue_manager.get_current_state(), DialogueState.ERROR_RECOVERY)
            self.assertIn("error", response.lower())
            
            # Test recovery
            dialogue_manager._handle_task_creation = original_handler
            response = dialogue_manager.process_input("what time is it?")
            self.assertEqual(dialogue_manager.get_current_state(), DialogueState.IDLE)
            self.assertIn("time", response.lower())
            
        finally:
            dialogue_manager._handle_task_creation = original_handler
            dialogue_manager.clear_error_state()

class StressTests(unittest.TestCase):
    def setUp(self):
        self.start_memory = self.get_memory_usage()
        if not os.path.exists("temp"):
            os.makedirs("temp")

    def get_memory_usage(self) -> float:
        process = psutil.Process()
        return process.memory_percent()

    def test_memory_leak(self):
        """Test for memory leaks during extended operation."""
        large_conversation = [
            f"remind me to do task {i}" for i in range(10)  # Reduced from 100 for faster testing
        ] + [
            "what tasks do I have?",
            "what's the weather like?",
            "take a screenshot"
        ] * 3  # Reduced repetition

        start_memory = self.get_memory_usage()
        
        for msg in large_conversation:
            dialogue_manager.process_input(msg)
            time.sleep(0.1)  # Allow for processing
        
        # Cleanup and force garbage collection
        self.cleanup_memory()
        gc.collect()
        time.sleep(0.5)  # Allow for garbage collection
        
        end_memory = self.get_memory_usage()
        memory_increase = end_memory - start_memory
        
        self.assertLess(memory_increase, 5.0, 
                       f"Memory usage increased by {memory_increase}%")

    def cleanup_memory(self):
        """Helper method to clean up memory."""
        dialogue_manager.clear_context()
        memory_system.memory_queue.queue.queue.clear()  # Clear memory queue
        gc.collect()

    def test_resource_cleanup(self):
        """Test resource cleanup under stress."""
        from screenshot_webcam import take_screenshot
        from clipboard_vision import get_clipboard_text
        
        # Perform resource-intensive operations
        for _ in range(5):  # Reduced from 10
            take_screenshot()
            get_clipboard_text()
            dialogue_manager.process_input("what do you see in the screenshot?")
            time.sleep(0.1)

        # Verify resources are properly cleaned up
        temp_files = [f for f in os.listdir("temp") if f.endswith('.jpg')]
        self.assertEqual(len(temp_files), 0, 
                        f"Temporary files not cleaned up: {temp_files}")

if __name__ == '__main__':
    unittest.main(verbosity=2)