import sqlite3
from sqlite3 import Connection, Cursor
from typing import Optional, Dict, Any, Generator, ContextManager
from contextlib import contextmanager
import threading
from queue import Queue, Empty
import time
from logging_config import logger
from monitoring import monitor_performance

class ThreadSafeConnection:
    """Thread-safe wrapper for SQLite connection."""
    def __init__(self, database: str):
        self._database = database
        self._local = threading.local()

    def get_connection(self) -> Connection:
        """Get or create a connection for the current thread."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = self._create_connection()
        return self._local.connection

    def _create_connection(self) -> Connection:
        """Create a new database connection with proper settings."""
        connection = sqlite3.connect(
            self._database,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            timeout=60.0  # Increased timeout for busy database
        )
        # Enable WAL mode for better concurrency
        connection.execute('PRAGMA journal_mode=WAL')
        # Configure for safe concurrent access
        connection.execute('PRAGMA busy_timeout=30000')  # 30 second timeout
        connection.row_factory = sqlite3.Row
        return connection

    def close(self):
        """Close the connection for the current thread if it exists."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection

class DatabaseManager:
    """Thread-safe database manager."""
    def __init__(self, database: str):
        self._database = database
        self._connection_manager = ThreadSafeConnection(database)
        self._lock = threading.RLock()

    @contextmanager
    def get_cursor(self) -> Generator[Cursor, None, None]:
        """Get a database cursor from a thread-local connection."""
        connection = self._connection_manager.get_connection()
        cursor = connection.cursor()
        try:
            yield cursor
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise e

    @monitor_performance("execute_with_retry")
    def execute_with_retry(self, 
                          query: str, 
                          params: tuple = None, 
                          retries: int = 3, 
                          backoff: float = 0.1) -> Any:
        """Execute a query with retry logic for handling concurrent access."""
        last_error = None
        
        for attempt in range(retries):
            try:
                with self.get_cursor() as cursor:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    return cursor.fetchall()
            except sqlite3.OperationalError as e:
                last_error = e
                if "database is locked" in str(e):
                    time.sleep(backoff * (2 ** attempt))
                    continue
                raise
            except Exception as e:
                raise
                
        raise last_error if last_error else Exception("Query failed after retries")

    def executemany_with_retry(self,
                             query: str,
                             params_list: list,
                             retries: int = 3,
                             backoff: float = 0.1) -> None:
        """Execute many queries with retry logic."""
        last_error = None
        
        for attempt in range(retries):
            try:
                with self.get_cursor() as cursor:
                    cursor.executemany(query, params_list)
                return
            except sqlite3.OperationalError as e:
                last_error = e
                if "database is locked" in str(e):
                    time.sleep(backoff * (2 ** attempt))
                    continue
                raise
            except Exception as e:
                raise
                
        raise last_error if last_error else Exception("Query failed after retries")

    def transaction(self):
        """Context manager for handling transactions."""
        @contextmanager
        def transaction_context():
            with self._lock:
                connection = self._connection_manager.get_connection()
                try:
                    yield connection
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

        return transaction_context()

    def close(self) -> None:
        """Close all database connections."""
        self._connection_manager.close()

# Global database manager instance
db_manager = DatabaseManager('memory.db')

# Cleanup handler
import atexit
def cleanup_db():
    """Cleanup function for database connections."""
    try:
        db_manager.close()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

atexit.register(cleanup_db)

# Example usage
if __name__ == "__main__":
    try:
        # Test concurrent access
        def test_concurrent_writes():
            for i in range(5):
                db_manager.execute_with_retry(
                    "INSERT INTO test (value) VALUES (?)",
                    (f"test_{i}",)
                )

        # Create test table
        db_manager.execute_with_retry("""
            CREATE TABLE IF NOT EXISTS test (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """)

        # Start multiple threads
        threads = [
            threading.Thread(target=test_concurrent_writes)
            for _ in range(5)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Check results
        results = db_manager.execute_with_retry("SELECT * FROM test")
        print(f"Test results: {len(results)} rows written")

    except Exception as e:
        print(f"Test error: {e}")
    finally:
        # Clean up test table
        db_manager.execute_with_retry("DROP TABLE IF EXISTS test")