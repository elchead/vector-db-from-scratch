import sys
import os
import pytest
import numpy as np
from typing import List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the module
from src.infrastructure.embedders import CohereEmbedder


# Helper function for cosine similarity since it was removed from the embedder
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def test_cohere_embeddings():
    """
    Test that embeddings for similar text chunks are closer to each other
    than to embeddings of dissimilar text chunks using the CohereEmbedder wrapper.
    """
    # Skip test if API key is not set
    if not os.environ.get("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY environment variable not set")

    # Initialize our embedder wrapper
    embedder = CohereEmbedder()

    # Define three text chunks (two similar, one different)
    chunk1 = "The quick brown fox jumps over the lazy dog."
    chunk2 = "A fast brown fox leaps over the sleepy canine."  # Similar to chunk1
    chunk3 = "Machine learning models process data to make predictions."  # Different topic

    # Get embeddings for each chunk using our wrapper
    embeddings = embedder.embed([chunk1, chunk2, chunk3])
    
    # Calculate similarities using our cosine similarity function
    similarity_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    similarity_1_3 = cosine_similarity(embeddings[0], embeddings[2])
    similarity_2_3 = cosine_similarity(embeddings[1], embeddings[2])
    
    # Print similarities for debugging
    print(f"Similarity between chunk1 and chunk2: {similarity_1_2:.4f}")
    print(f"Similarity between chunk1 and chunk3: {similarity_1_3:.4f}")
    print(f"Similarity between chunk2 and chunk3: {similarity_2_3:.4f}")
    
    # Assert that similar chunks have higher similarity than dissimilar ones
    assert similarity_1_2 > similarity_1_3
    assert similarity_1_2 > similarity_2_3


def test_cohere_embedder_single_embed():
    """
    Test the embed_single method of the CohereEmbedder wrapper.
    """
    # Skip test if API key is not set
    if not os.environ.get("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY environment variable not set")

    # Initialize our embedder wrapper
    embedder = CohereEmbedder()

    # Define a text chunk
    chunk = "The quick brown fox jumps over the lazy dog."

    # Test embed_single method
    embedding = embedder.embed_single(chunk)
    
    # Assert that the embedding is a list of floats with non-zero length
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    
    # Print embedding dimensions for debugging
    print(f"Embedding dimension: {len(embedding)}")
