"""
Extended memory system with file persistence (memory.md).
Builds on SimpleMemory to add query and storage capabilities.
"""

from typing import List, Dict, Any, Tuple, Optional
from .memory import SimpleMemory


class FileMemory(SimpleMemory):
    """
    Extended memory that adds file persistence and query capabilities.
    Inherits from SimpleMemory for compatibility, adds memory.md support.
    """
    
    def __init__(self, memory_file: str = "memory.md"):
        super().__init__()
        self.memory_file = memory_file
        self.file_cache = []  # Recent entries from file
        self._load_file_cache()
    
    def _load_file_cache(self, limit: int = 100):
        """Load recent entries from memory.md into cache."""
        self.file_cache = []
        try:
            with open(self.memory_file, 'r') as f:
                lines = f.readlines()
                # Keep last N entries
                self.file_cache = lines[-limit:] if len(lines) > limit else lines
        except FileNotFoundError:
            pass  # File doesn't exist yet
    
    def store_to_file(self, content: str, episode: str = "", step: int = 0):
        """
        Store content to memory.md file.
        
        Args:
            content: Text to store
            episode: Episode identifier
            step: Step number
        """
        with open(self.memory_file, 'a') as f:
            metadata = f"E:{episode}|S:{step}" if episode else f"S:{step}"
            f.write(f"\n[{metadata}] {content}\n")
        
        # Update cache
        entry = f"[{metadata}] {content}\n"
        self.file_cache.append(entry)
        if len(self.file_cache) > 100:
            self.file_cache.pop(0)
    
    def query(self, query: str, limit: int = 3) -> str:
        """
        Query memory for relevant information.
        Searches both in-memory data and file cache.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Formatted string of matching memories
        """
        results = []
        query_lower = query.lower()
        
        # Search in file cache first (more persistent memories)
        for line in reversed(self.file_cache):
            if query_lower in line.lower():
                results.append(line.strip())
                if len(results) >= limit:
                    break
        
        # If not enough results, search in-memory data
        if len(results) < limit and self._data:
            for env_data in reversed(self._data):
                for record in reversed(env_data):
                    # Search in all fields
                    for value in record.values():
                        if isinstance(value, str) and query_lower in value.lower():
                            results.append(str(record))
                            break
                    if len(results) >= limit:
                        break
                if len(results) >= limit:
                    break
        
        return "\n".join(results) if results else "No relevant memory found"
    
    def store_staged(self, staged_data: Dict[str, Any], episode: str = "", step: int = 0):
        """
        Store data from staged processing.
        
        Args:
            staged_data: Dictionary containing plan, action, reflection, etc.
            episode: Episode identifier
            step: Step number
        """
        # Store important parts to file
        if staged_data.get('plan'):
            self.store_to_file(f"[Plan] {staged_data['plan']}", episode, step)
        
        if staged_data.get('memory_store'):
            self.store_to_file(staged_data['memory_store'], episode, step)
        
        if staged_data.get('reflection'):
            self.store_to_file(f"[Reflection] {staged_data['reflection']}", episode, step)
        
        # Also store in regular memory structure for compatibility
        if self._data is not None:
            record = {
                'text_obs': staged_data.get('plan', ''),
                'action': staged_data.get('action', ''),
                'reflection': staged_data.get('reflection', '')
            }
            # Store for all environments (broadcast)
            broadcast_record = {k: [v] * self.batch_size for k, v in record.items()}
            self.store(broadcast_record)
    
    def clear_file(self):
        """Clear the memory.md file."""
        open(self.memory_file, 'w').close()
        self.file_cache = []
    
    def get_recent_from_file(self, n: int = 10) -> List[str]:
        """Get n most recent entries from file cache."""
        return self.file_cache[-n:] if self.file_cache else []