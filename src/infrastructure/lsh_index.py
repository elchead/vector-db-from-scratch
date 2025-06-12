"""
Locality Sensitive Hashing (LSH) Index.
"""

from typing import List, Tuple, Dict, Optional, Union, Any, Set
from uuid import UUID
import numpy as np

from src.domain.models import Chunk # Assuming Chunk is the type returned by search/get_chunk
from src.infrastructure.index import IndexedChunk

# TODO: implementation not yet finished
class LSHIndex:
    """
    Locality Sensitive Hashing (LSH) index for approximate nearest neighbor search.

    LSH works by hashing input items so that similar items map to the same "buckets"
    with high probability. The choice of hash functions is crucial.
    For cosine similarity, random projection-based LSH is common.

    Time Complexity:
    - Building hash tables: O(num_tables * num_hashes_per_table * d) for hyperplane generation.
    - Adding a vector: O(num_tables * num_hashes_per_table * d) to compute hashes and add to buckets.
                       Total for N vectors: O(N * num_tables * num_hashes_per_table * d)
    - Querying: O(num_tables * num_hashes_per_table * d) to compute query hashes.
                  + O(num_candidates * d) for exact similarity calculation on candidates.
                  The number of candidates depends on bucket sizes.

    Space Complexity:
    - Hyperplanes: O(num_tables * num_hashes_per_table * d)
    - Hash tables: O(N * num_tables) in the worst case if each item maps to a unique bucket per table,
                     but typically less due to collisions.
    - Storing chunks: O(N * d) for the original vectors/chunks.

    Use when:
    - Approximate nearest neighbors are acceptable.
    - Query speed is critical for large datasets.
    - High-dimensional data.
    """

    def __init__(self, embedding_dim: int, num_tables: int = 10, num_hashes_per_table: int = 8):
        """
        Initialize the LSHIndex.

        Args:
            embedding_dim: The dimensionality of the vectors to be indexed.
            num_tables: The number of hash tables to use.
            num_hashes_per_table: The number of hash functions (hyperplanes) per table.
        """
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
        if num_tables <= 0:
            raise ValueError("Number of tables must be positive.")
        if num_hashes_per_table <= 0:
            raise ValueError("Number of hashes per table must be positive.")

        self._embedding_dim: int = embedding_dim
        self._num_tables: int = num_tables
        self._num_hashes_per_table: int = num_hashes_per_table

        # Store actual chunks by their ID
        self._chunks: Dict[UUID, IndexedChunk] = {}
        # Cache for normalized vectors to avoid recomputing
        self._normalized_vectors: Dict[UUID, np.ndarray] = {}

        # LSH specific structures
        # List of hash tables. Each table is a dict mapping a hash signature (tuple) to a list of chunk IDs.
        self._hash_tables: List[Dict[Tuple[int, ...], List[UUID]]] = [{} for _ in range(num_tables)]
        # List of hyperplanes. Each element corresponds to a table and contains its hyperplanes.
        # Shape: (num_tables, num_hashes_per_table, embedding_dim)
        self._hyperplanes: np.ndarray = np.random.randn(num_tables, num_hashes_per_table, embedding_dim)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector # Avoid division by zero for zero vectors
        return vector / norm

    def _get_hash_signature(self, vector: np.ndarray, table_index: int) -> Tuple[int, ...]:
        """
        Compute the hash signature for a vector in a specific hash table.
        The vector should be normalized before calling this for cosine LSH.
        """
        # Projections: dot product of vector with hyperplanes
        projections = np.dot(self._hyperplanes[table_index], vector)
        # Hash signature: 1 if projection is non-negative, 0 otherwise
        return tuple((projections >= 0).astype(int))

    def add(self, chunk: IndexedChunk) -> None:
        """Add an IndexedChunk to the index."""
        if len(chunk.embedding) != self._embedding_dim:
            raise ValueError(f"Chunk embedding dimension {len(chunk.embedding)} does not match index dimension {self._embedding_dim}")

        chunk_id = chunk.id
        # Store embedding as np.float32 array for memory/speed efficiency
        chunk_embedding_np = np.array(chunk.embedding, dtype=np.float32)
        normalized_vector = self._normalize_vector(chunk_embedding_np)

        # Replace the chunk with a copy using np.float32 embedding
        chunk_stored = IndexedChunk(id=chunk.id, embedding=chunk_embedding_np)
        self._chunks[chunk_id] = chunk_stored
        self._normalized_vectors[chunk_id] = normalized_vector
        # All embeddings are now np.float32 arrays in storage

        for i in range(self._num_tables):
            signature = self._get_hash_signature(normalized_vector, i)
            if signature not in self._hash_tables[i]:
                self._hash_tables[i][signature] = []
            self._hash_tables[i][signature].append(chunk_id)

    def add_many(self, chunks: List[IndexedChunk]) -> None:
        """Add multiple IndexedChunks to the index."""
        for chunk in chunks:
            self.add(chunk)  # add() already ensures np.float32 embedding storage

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a chunk from the index."""
        if chunk_id not in self._chunks:
            return False

        # Remove from main storage
        del self._chunks[chunk_id]
        normalized_vector_to_remove = self._normalized_vectors.pop(chunk_id, None)

        if normalized_vector_to_remove is None:
            # This case implies inconsistency, but we've removed from _chunks.
            return True 

        # Remove from hash tables
        for i in range(self._num_tables):
            signature = self._get_hash_signature(normalized_vector_to_remove, i)
            if signature in self._hash_tables[i]:
                try:
                    self._hash_tables[i][signature].remove(chunk_id)
                    if not self._hash_tables[i][signature]: # Clean up empty list
                        del self._hash_tables[i][signature]
                except ValueError:
                    # Chunk ID not in this bucket's list, log or handle as appropriate
                    pass 
        return True

    def update(self, chunk: Chunk) -> bool:
        """
        Update a chunk in the index. This is done by removing the old one and adding the new one.
        Note: `chunk` here is `src.domain.models.Chunk` which should have `id` and `embedding`.
        """
        if chunk.id not in self._chunks:
            return False # Chunk to update does not exist
        
        self.remove(chunk.id)
        
        # Store embedding as np.float32 array
        chunk_embedding_np = np.array(chunk.embedding, dtype=np.float32)
        indexed_chunk = IndexedChunk(id=chunk.id, embedding=chunk_embedding_np)
        self.add(indexed_chunk)
        return True

    def search(self, query_vector: Union[List[float], np.ndarray], top_k: int = 5) -> List[Tuple[IndexedChunk, float]]:
        """
        Search for the most similar chunks to the query vector.
        Returns List of Tuples of (IndexedChunk, similarity_score).
        Aligns with BruteForceIndex's effective return type.
        """
        if not self._chunks:
            return []
        if top_k <= 0:
            return []

        query_vector_np = np.array(query_vector)
        if query_vector_np.shape[0] != self._embedding_dim:
            raise ValueError(f"Query vector dimension {query_vector_np.shape[0]} does not match index dimension {self._embedding_dim}")

        normalized_query = self._normalize_vector(query_vector_np)

        candidate_ids: Set[UUID] = set()
        for i in range(self._num_tables):
            signature = self._get_hash_signature(normalized_query, i)
            if signature in self._hash_tables[i]:
                candidate_ids.update(self._hash_tables[i][signature])
        
        if not candidate_ids:
            return []

        results: List[Tuple[IndexedChunk, float]] = []
        for chunk_id in candidate_ids:
            if chunk_id in self._chunks: 
                indexed_chunk = self._chunks[chunk_id]
                normalized_candidate_vector = self._normalized_vectors[chunk_id]
                similarity = np.dot(normalized_query, normalized_candidate_vector)
                results.append((indexed_chunk, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return type is List[Tuple[IndexedChunk, float]]
        return results[:top_k]

    def clear(self) -> None:
        """Clear all chunks from the index."""
        self._chunks.clear()
        self._normalized_vectors.clear()
        self._hash_tables = [{} for _ in range(self._num_tables)]

    def count(self) -> int:
        """Get the number of chunks in the index."""
        return len(self._chunks)

    def get_chunk(self, chunk_id: UUID) -> Optional[IndexedChunk]:
        """Get an indexed chunk by its ID."""
        return self._chunks.get(chunk_id)

    def get_all_chunks(self) -> List[IndexedChunk]:
        """Get all indexed chunks in the index."""
        return list(self._chunks.values())

    def __repr__(self) -> str:
        return (
            f"LSHIndex(embedding_dim={self._embedding_dim}, "
            f"num_tables={self._num_tables}, num_hashes_per_table={self._num_hashes_per_table}, "
            f"chunks_count={self.count()})"
        )

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the index."""
        return {
            "embedding_dim": self._embedding_dim,
            "num_tables": self._num_tables,
            "num_hashes_per_table": self._num_hashes_per_table,
        }

# Example Usage (for testing purposes, typically not in the class file)
if __name__ == '__main__':
    dim = 128
    # For consistent test runs, set a seed for numpy's random operations
    np.random.seed(42)
    lsh_index = LSHIndex(embedding_dim=dim, num_tables=20, num_hashes_per_table=10)

    # Create some dummy IndexedChunk data
    num_items = 1000
    dummy_chunks = []
    for i in range(num_items):
        chunk_uuid = UUID(int=i) 
        embedding = np.random.rand(dim).tolist()
        dummy_chunks.append(IndexedChunk(id=chunk_uuid, embedding=embedding))

    print(f"Adding {len(dummy_chunks)} chunks to the LSH index...")
    lsh_index.add_many(dummy_chunks)
    print(f"Index count: {lsh_index.count()}")

    query_embedding = np.array(dummy_chunks[0].embedding) + np.random.normal(0, 0.1, dim)
    query_embedding_list = query_embedding.tolist()

    print(f"\nSearching for top 5 similar to a query vector...")
    search_results = lsh_index.search(query_vector=query_embedding_list, top_k=5)

    print("Search Results (Chunk ID, Similarity Score):")
    for chunk, score in search_results:
        print(f"  ID: {chunk.id}, Score: {score:.4f}")

    if num_items > 0:
        chunk_to_remove_id = dummy_chunks[0].id
        print(f"\nRemoving chunk: {chunk_to_remove_id}")
        removed = lsh_index.remove(chunk_to_remove_id)
        print(f"Removed: {removed}, Index count: {lsh_index.count()}")
        
        retrieved_chunk = lsh_index.get_chunk(chunk_to_remove_id)
        print(f"Retrieved removed chunk: {retrieved_chunk}")

        print(f"\nSearching again after removal...")
        search_results_after_remove = lsh_index.search(query_vector=query_embedding_list, top_k=5)
        print("Search Results (Chunk ID, Similarity Score):")
        for chunk_obj, score in search_results_after_remove: # Renamed chunk to chunk_obj to avoid conflict
            print(f"  ID: {chunk_obj.id}, Score: {score:.4f}")

    if lsh_index.count() > 0:
        # Get the first chunk from the current state of the index for update test
        first_chunk_in_index = lsh_index.get_all_chunks()[0]
        chunk_to_update_id = first_chunk_in_index.id
        print(f"\nUpdating chunk: {chunk_to_update_id}")
        new_embedding = np.random.rand(dim).tolist()
        
        # Create a mock domain.models.Chunk for the update method
        # Ensure all required fields for Chunk are present
        mock_domain_chunk = Chunk(
            id=chunk_to_update_id, 
            document_id=UUID(int=9999), # Dummy document_id
            text="updated text example", 
            embedding=new_embedding, 
            metadata={'source': 'update_test'}
        )
        updated = lsh_index.update(mock_domain_chunk)
        print(f"Updated: {updated}")

        updated_chunk_retrieved = lsh_index.get_chunk(chunk_to_update_id)
        if updated_chunk_retrieved:
            print(f"Updated chunk embedding (first 5 dims): {updated_chunk_retrieved.embedding[:5]}")
            print(f"Original new embedding (first 5 dims): {new_embedding[:5]}")
            assert np.allclose(updated_chunk_retrieved.embedding, new_embedding), "Embedding not updated correctly!"

    print(f"\nClearing index...")
    lsh_index.clear()
    print(f"Index count after clear: {lsh_index.count()}")

    print("\nLSHIndex basic tests completed.")

    print("\n--- Testing with a small, predictable dataset ---")
    dim_small = 2
    np.random.seed(42) # Reset seed before creating new LSHIndex for predictable hyperplanes
    lsh_small = LSHIndex(embedding_dim=dim_small, num_tables=5, num_hashes_per_table=3)
    
    chunk_a_id = UUID(int=10000)
    chunk_b_id = UUID(int=10001)
    chunk_c_id = UUID(int=10002)

    chunks_small_data = [
        IndexedChunk(id=chunk_a_id, embedding=[1.0, 0.0]),
        IndexedChunk(id=chunk_b_id, embedding=[0.8, 0.2]),
        IndexedChunk(id=chunk_c_id, embedding=[-1.0, 0.1]),
    ]
    lsh_small.add_many(chunks_small_data)

    query_vec_small = [0.9, 0.1]
    print(f"Querying with {query_vec_small}:")
    results_small = lsh_small.search(query_vec_small, top_k=2)
    for chunk_obj, score in results_small: # Renamed chunk to chunk_obj
        print(f"  ID: {chunk_obj.id}, Embedding: {chunk_obj.embedding}, Score: {score:.4f}")

    vec_a_norm = lsh_small._normalize_vector(np.array(chunks_small_data[0].embedding))
    print(f"\nHash signatures for Chunk A ({chunks_small_data[0].embedding}):")
    for i in range(lsh_small._num_tables):
        sig = lsh_small._get_hash_signature(vec_a_norm, i)
        print(f"  Table {i}: {sig} -> Bucket content: {lsh_small._hash_tables[i].get(sig)}")
