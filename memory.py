from queue import Queue
from threading import Thread, Lock, Event
import sqlite3
import json
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime, timedelta
from collections import deque, defaultdict
from logging_config import logger
from error_recovery import error_handler
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum
from dataclasses import dataclass, field
import uuid
from db_connection import db_manager
import time
from monitoring import monitor_performance
from base_cleanup import base_cleanup_manager

class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    TASK = "task"
    USER_PREFERENCE = "preference"
    CONTEXT = "context"

@dataclass
class Memory:
    content: Dict[str, Any]
    type: MemoryType
    timestamp: datetime
    importance: float = 0.0
    references: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MemoryQueue:
    def __init__(self):
        self.queue = Queue()
        self.stop_event = Event()
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def _process_queue(self):
        while not self.stop_event.is_set():
            try:
                if not self.queue.empty():
                    memory = self.queue.get_nowait()
                    if memory is None:  # Sentinel value
                        break
                    try:
                        memory_system.store_memory(memory)
                    except Exception as e:
                        logger.error(f"Error storing queued memory: {e}")
                    finally:
                        self.queue.task_done()
            except Exception as e:
                logger.error(f"Error in memory queue processing: {e}")
            finally:
                if self.queue.empty():
                    time.sleep(0.1)  # Prevent busy waiting

    def add(self, memory: Memory):
        self.queue.put(memory)

    def stop(self):
        self.stop_event.set()
        self.queue.put(None)  # Add sentinel
        self.worker.join(timeout=5)

class EnhancedMemorySystem:
    def __init__(self, db_path: str = 'memory.db'):
        self.db_path = db_path
        self.max_short_term_memories = 100
        self.max_long_term_memories = 1000
        self.short_term_ttl = timedelta(hours=24)
        self.memory_lock = Lock()
        self.vectorizer = TfidfVectorizer()
        self.memory_vectors = {}
        self.memory_queue = MemoryQueue()
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the SQLite database with enhanced schema."""
        try:
            # Create memories table
            db_manager.execute_with_retry('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL,
                    metadata TEXT,
                    embedding TEXT
                )
            ''')
            
            # Create memory_references table
            db_manager.execute_with_retry('''
                CREATE TABLE IF NOT EXISTS memory_references (
                    memory_id TEXT,
                    referenced_id TEXT,
                    FOREIGN KEY(memory_id) REFERENCES memories(id),
                    FOREIGN KEY(referenced_id) REFERENCES memories(id)
                )
            ''')
            
            # Create indexes
            db_manager.execute_with_retry('''
                CREATE INDEX IF NOT EXISTS idx_memory_type 
                ON memories(memory_type)
            ''')
            
            db_manager.execute_with_retry('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON memories(timestamp)
            ''')
            
            logger.info(f"Memory database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize memory database: {e}")
            raise

    def _generate_embedding(self, content: Dict[str, Any]) -> np.ndarray:
        """Generate embedding vector for memory content."""
        try:
            # Convert content to string representation
            text = ' '.join(str(v) for v in content.values())
            # Generate TF-IDF vector
            vector = self.vectorizer.fit_transform([text])
            return vector.toarray()[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(1)  # Return zero vector on error

    @monitor_performance("store_memory")
    def store_memory(self, memory: Memory) -> str:
        try:
            memory_id = str(uuid.uuid4())
            
            if memory.importance == 0.0:
                memory.importance = self._calculate_importance(memory)
            
            embedding = self._generate_embedding(memory.content)
            
            # Store main memory
            query = '''
                INSERT INTO memories 
                (id, content, memory_type, timestamp, importance, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                memory_id,
                json.dumps(memory.content),
                memory.type.value,
                memory.timestamp.isoformat(),
                memory.importance,
                json.dumps(memory.metadata),
                json.dumps(embedding.tolist())
            )
            
            db_manager.execute_with_retry(query, params)
            
            # Store references
            if memory.references:
                ref_query = '''
                    INSERT INTO memory_references (memory_id, referenced_id)
                    VALUES (?, ?)
                '''
                for ref in memory.references:
                    db_manager.execute_with_retry(ref_query, (memory_id, ref))
            
            with self.memory_lock:
                self.memory_vectors[memory_id] = embedding
            
            self._cleanup_old_memories()
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise

    @monitor_performance("retrieve_memories")
    def retrieve_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on a query string.
        
        Args:
            query (str): The search query
            limit (int): Maximum number of memories to return
            
        Returns:
            List[Dict[str, Any]]: List of relevant memories
        """
        try:
            query_embedding = self._generate_embedding({'text': query})
            
            with self.memory_lock:
                # Get all memory vectors
                memory_ids = list(self.memory_vectors.keys())
                vectors = np.array([self.memory_vectors[mid] for mid in memory_ids])
                
                # Calculate similarities
                similarities = np.inner(vectors, query_embedding)
                top_indices = np.argsort(similarities)[-limit:][::-1]
                
                # Fetch memories
                memories = []
                for idx in top_indices:
                    memory_id = memory_ids[idx]
                    query = "SELECT * FROM memories WHERE id = ?"
                    result = db_manager.execute_with_retry(query, (memory_id,))
                    if result:
                        memory_data = result[0]
                        memories.append({
                            'id': memory_data['id'],
                            'content': json.loads(memory_data['content']),
                            'type': memory_data['memory_type'],
                            'timestamp': memory_data['timestamp'],
                            'importance': memory_data['importance'],
                            'similarity': float(similarities[idx])
                        })
                
                return memories
                
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def _calculate_importance(self, memory: Memory) -> float:
        """Calculate memory importance using multiple factors."""
        importance = 0.0
        try:
            # Base importance by type
            type_weights = {
                MemoryType.LONG_TERM: 0.8,
                MemoryType.SHORT_TERM: 0.4,
                MemoryType.TASK: 0.6,
                MemoryType.USER_PREFERENCE: 0.7,
                MemoryType.CONTEXT: 0.5
            }
            importance += type_weights.get(memory.type, 0.3)
            
            # Add importance based on references
            importance += min(len(memory.references) * 0.1, 0.3)
            
            # Consider metadata
            if memory.metadata:
                if memory.metadata.get('priority') == 'high':
                    importance += 0.2
                if memory.metadata.get('sentiment', 0) > 0.5:
                    importance += 0.1
            
            return min(importance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating importance: {e}")
            return 0.5

    def _cleanup_old_memories(self) -> None:
        """Clean up old memories based on TTL and importance."""
        try:
            current_time = datetime.now()
            
            # Delete old short-term memories
            cutoff_time = current_time - self.short_term_ttl
            delete_query = '''
                DELETE FROM memories 
                WHERE memory_type = ? AND timestamp < ? AND importance < ?
            '''
            db_manager.execute_with_retry(
                delete_query, 
                (MemoryType.SHORT_TERM.value, cutoff_time.isoformat(), 0.7)
            )
            
            # Clean up memory vectors
            with self.memory_lock:
                # Get current memory IDs
                query = "SELECT id FROM memories"
                results = db_manager.execute_with_retry(query)
                valid_ids = {row['id'] for row in results}
                
                # Remove vectors for deleted memories
                self.memory_vectors = {
                    mid: vec for mid, vec in self.memory_vectors.items()
                    if mid in valid_ids
                }
                
        except Exception as e:
            logger.error(f"Error in cleanup_old_memories: {e}")

def remember_interaction(user_input: str, assistant_response: str, context: Dict[str, Any]) -> None:
    try:
        content = {
            "user_input": user_input,
            "assistant_response": assistant_response,
            "context": context
        }
        
        memory = Memory(
            content=content,
            type=MemoryType.SHORT_TERM,
            timestamp=datetime.now(),
            metadata={
                "interaction_type": "conversation",
                "sentiment": context.get("sentiment", 0.0),
                "key_phrases": context.get("key_phrases", [])
            }
        )
        
        # Add to queue instead of direct storage
        memory_system.memory_queue.add(memory)
        logger.debug(f"Queued interaction for memory storage: {user_input[:50]}...")
        
    except Exception as e:
        logger.error(f"Error queueing interaction for memory: {e}")

# Create global instance
memory_system = EnhancedMemorySystem()

# Register cleanup
base_cleanup_manager.add_cleanup_function(lambda: memory_system.memory_queue.stop())

# Expose the get_relevant_memories function as a convenience wrapper
def get_relevant_memories(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Convenience function to retrieve relevant memories."""
    return memory_system.retrieve_memories(query, limit)

__all__ = ['memory_system', 'Memory', 'MemoryType', 'get_relevant_memories', 'remember_interaction']