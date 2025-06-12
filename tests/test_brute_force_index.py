"""
Tests for the BruteForceIndex implementation.
"""

import sys
import os
import pytest
import numpy as np
from uuid import uuid4

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.domain.models import ChunkMetadata, Vector
from src.infrastructure.brute_force_index import BruteForceIndex
from src.infrastructure.index import IndexedChunk


@pytest.fixture
def index():
    """Create a BruteForceIndex instance for testing."""
    return BruteForceIndex()


@pytest.fixture
def test_chunks():
    """Create test chunks with different embeddings."""
    chunk1 = IndexedChunk(
        id=uuid4(),
        embedding=[0.1, 0.2, 0.3, 0.4]
    )
    
    chunk2 = IndexedChunk(
        id=uuid4(),
        embedding=[0.2, 0.3, 0.4, 0.5]
    )
    
    chunk3 = IndexedChunk(
        id=uuid4(),
        embedding=[0.9, 0.8, 0.7, 0.6]
    )
    
    return chunk1, chunk2, chunk3


@pytest.fixture
def populated_index(index, test_chunks):
    """Create an index populated with test chunks."""
    chunk1, chunk2, chunk3 = test_chunks
    index.add_many([chunk1, chunk2, chunk3])
    return index, test_chunks


def test_add_and_count(index, test_chunks):
    """Test adding chunks and counting them."""
    chunk1, chunk2, chunk3 = test_chunks
    
    # Initially empty
    assert index.count() == 0
    
    # Add chunks one by one
    index.add(chunk1)
    assert index.count() == 1
    
    index.add_many([chunk2, chunk3])
    assert index.count() == 3
    
    # Add another chunk
    chunk4 = IndexedChunk(
        id=uuid4(),
        embedding=[0.5, 0.6, 0.7, 0.8]
    )
    index.add(chunk4)
    
    assert index.count() == 4


def test_remove(populated_index):
    """Test removing a chunk from the index."""
    index, (chunk1, chunk2, chunk3) = populated_index
    
    # Remove chunk1
    result = index.remove(chunk1.id)
    assert result is True
    assert index.count() == 2
    
    # Try to remove a non-existent chunk
    result = index.remove(uuid4())
    assert result is False
    assert index.count() == 2


def test_update(populated_index):
    """Test updating a chunk in the index."""
    index, (chunk1, chunk2, chunk3) = populated_index
    
    # Update chunk1 with new embedding
    updated_chunk = IndexedChunk(
        id=chunk1.id,
        embedding=[0.9, 0.9, 0.9, 0.9]
    )
    
    result = index.update(updated_chunk)
    assert result is True
    
    # Verify the chunk was updated
    retrieved_chunk = index.get_chunk(chunk1.id)
    import numpy as np
    assert np.allclose(retrieved_chunk.embedding, [0.9, 0.9, 0.9, 0.9])
    
    # Try to update a non-existent chunk
    non_existent_chunk = IndexedChunk(
        id=uuid4(),
        embedding=[0.1, 0.1, 0.1, 0.1]
    )
    
    result = index.update(non_existent_chunk)
    assert result is False


def test_search_similarity(populated_index):
    """Test searching for similar chunks."""
    index, (chunk1, chunk2, chunk3) = populated_index
    
    # Search with a vector similar to chunk1 and chunk2
    query_vector = [0.12, 0.22, 0.32, 0.42]
    results = index.search(query_vector, top_k=2)
    
    # Should return chunk1 and chunk2 as they are most similar
    assert len(results) == 2
    
    # Extract chunk IDs from results for easier comparison
    result_ids = [result[0].id for result in results]
    assert chunk1.id in result_ids
    assert chunk2.id in result_ids
    
    # Verify chunk3 is not in the results as it's very different
    assert chunk3.id not in result_ids


def test_search_with_empty_index(index):
    """Test searching with an empty index."""
    # Search should return empty list
    query_vector = [0.1, 0.2, 0.3, 0.4]
    results = index.search(query_vector)
    assert len(results) == 0


def test_clear(populated_index):
    """Test clearing the index."""
    index, _ = populated_index
    
    # Index should have chunks initially
    assert index.count() == 3
    
    # Clear the index
    index.clear()
    assert index.count() == 0


def test_get_all_chunks(populated_index):
    """Test getting all chunks from the index."""
    index, (chunk1, chunk2, chunk3) = populated_index
    
    chunks = index.get_all_chunks()
    assert len(chunks) == 3
    
    # Verify all chunks are in the result
    chunk_ids = [chunk.id for chunk in chunks]
    assert chunk1.id in chunk_ids
    assert chunk2.id in chunk_ids
    assert chunk3.id in chunk_ids


def test_cosine_similarity_ordering(populated_index):
    """Test that search results are ordered by cosine similarity."""
    index, (chunk1, chunk2, chunk3) = populated_index
    
    # Create a query vector that's more similar to chunk2 than chunk1
    query_vector = [0.2, 0.3, 0.4, 0.5]  # Exactly the same as chunk2
    
    results = index.search(query_vector, top_k=3)
    
    # First result should be chunk2 with similarity of 1.0 (exact match)
    assert results[0][0].id == chunk2.id
    assert abs(results[0][1] - 1.0) < 1e-6
    
    # Second result should be chunk1 (similar but not exact)
    assert results[1][0].id == chunk1.id
    
    # Third result should be chunk3 (least similar)
    assert results[2][0].id == chunk3.id
    
    # Verify that similarity scores are in descending order
    assert results[0][1] > results[1][1]
    assert results[1][1] > results[2][1]


def test_get_chunk(populated_index):
    """Test getting a specific chunk by ID."""
    index, (chunk1, chunk2, chunk3) = populated_index
    
    # Get existing chunk
    retrieved_chunk = index.get_chunk(chunk1.id)
    assert retrieved_chunk is not None
    assert retrieved_chunk.id == chunk1.id
    
    # Get non-existent chunk
    non_existent_chunk = index.get_chunk(uuid4())
    assert non_existent_chunk is None
