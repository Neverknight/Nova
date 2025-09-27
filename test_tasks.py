import asyncio
from datetime import datetime, timedelta
from task_management import task_scheduler, TaskPriority
from logging_config import logger

async def test_task_scheduler():
    try:
        # Test 1: Create a task due in 15 seconds with a 10-second reminder
        logger.info("\nTest 1: Creating a task due in 15 seconds...")
        task1_id = await task_scheduler.add_task(
            description="Check your email",
            due_date=datetime.now() + timedelta(seconds=15),
            priority=TaskPriority.HIGH,
            remind_before=timedelta(seconds=10)
        )
        logger.info(f"Created task with ID: {task1_id}")

        # Test 2: Create a task due in 30 seconds
        logger.info("\nTest 2: Creating a second task...")
        task2_id = await task_scheduler.add_task(
            description="Review your calendar",
            due_date=datetime.now() + timedelta(seconds=30),
            priority=TaskPriority.MEDIUM,
            remind_before=timedelta(seconds=10)
        )
        logger.info(f"Created task with ID: {task2_id}")

        # Test 3: List pending tasks
        pending_tasks = task_scheduler.get_pending_tasks()
        logger.info("\nTest 3: Current pending tasks:")
        for task in pending_tasks:
            logger.info(f"- {task['description']} (due: {task['due_date']})")

        # Wait for tasks to execute
        logger.info("\nWaiting for task executions (45 seconds)...")
        logger.info("You should hear notifications at:")
        logger.info("- 5 seconds (first task reminder)")
        logger.info("- 15 seconds (first task due)")
        logger.info("- 20 seconds (second task reminder)")
        logger.info("- 30 seconds (second task due)")
        
        # Wait for all notifications
        await asyncio.sleep(45)

        # Show final task status
        logger.info("\nFinal task statuses:")
        for task_id in [task1_id, task2_id]:
            task = task_scheduler.get_task(task_id)
            if task:
                logger.info(f"- {task['description']}: {task['status'].value}")

    except Exception as e:
        logger.error(f"Error in test: {e}")
        raise

if __name__ == "__main__":
    try:
        print("\nStarting task notification test...")
        print("You should hear voice notifications for task reminders and executions.")
        print("Please ensure your system volume is turned on.\n")
        
        # Run the test
        asyncio.run(test_task_scheduler())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        logger.info("Test complete")