"""
Brute force vector index for similarity search.
"""

from typing import List, Tuple, Dict, Optional, Union, Any
from uuid import UUID
import numpy as np
from src.domain.models import Vector, Chunk
from src.infrastructure.index import IndexedChunk

class BruteForceIndex:
    """
    A simple brute force index for similarity search.
    
    O(n·d) to load vectors into an array
    O(n·d) per query (exact)
    O(n·d) (just the raw storage)

    Use when:
    - The number of vectors is small (e.g., < 50k)
    - Exact results are required
    - Embeddings dimension is small (e.g., < 512)
    """
    
    def __init__(self):
        # Store chunks with their IDs
        self._chunks: Dict[UUID, IndexedChunk] = {}
        # Cache for normalized vectors to avoid recomputing during search
        self._normalized_vectors: Dict[UUID, np.ndarray] = {}
    
    def add(self, chunk: IndexedChunk) -> None:
        # Store embedding as np.float32 array for memory/speed efficiency
        chunk_embedding_np = np.array(chunk.embedding, dtype=np.float32)
        # Replace the chunk with a copy using np.float32 embedding
        chunk_stored = IndexedChunk(id=chunk.id, embedding=chunk_embedding_np)
        self._chunks[chunk.id] = chunk_stored
        self._normalized_vectors[chunk.id] = self._normalize_vector(chunk_embedding_np)
    
    def add_many(self, chunks: List[IndexedChunk]) -> None:
        for chunk in chunks:
            self.add(chunk) 
    
    def remove(self, chunk_id: UUID) -> bool:
        if chunk_id in self._chunks:
            del self._chunks[chunk_id]
            if chunk_id in self._normalized_vectors:
                del self._normalized_vectors[chunk_id]
            return True
        return False
    
    def update(self, chunk: Chunk) -> bool:
        if chunk.id in self._chunks:
            chunk_embedding_np = np.array(chunk.embedding, dtype=np.float32)
            chunk_stored = IndexedChunk(id=chunk.id, embedding=chunk_embedding_np)
            self._chunks[chunk.id] = chunk_stored
            self._normalized_vectors[chunk.id] = self._normalize_vector(chunk_embedding_np)
            return True
        return False
    
    def search(self, query_vector: Union[List[float], np.ndarray], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for the most similar chunks to the query vector.
        
        Args:
            query_vector: The query vector to search for.
            top_k: The number of results to return.
            
        Returns:
            List of tuples containing (chunk, similarity_score) sorted by similarity score in descending order.
        """
        if not self._chunks:
            return []
        
        # Convert query vector to numpy array and normalize
        query_vector_np = np.array(query_vector)
        normalized_query = self._normalize_vector(query_vector_np)
        
        # Calculate cosine similarity for all vectors
        results = []
        for chunk_id, normalized_vector in self._normalized_vectors.items():
            # Cosine similarity is the dot product of normalized vectors
            similarity = np.dot(normalized_query, normalized_vector)
            results.append((self._chunks[chunk_id], float(similarity)))
        
        # Sort by similarity score in descending order and return top_k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def clear(self) -> None:
        self._chunks.clear()
        self._normalized_vectors.clear()
    
    def count(self) -> int:
        return len(self._chunks)
    
    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)
    
    def get_all_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())