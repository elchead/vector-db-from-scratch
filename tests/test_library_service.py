import sys
import os
import pytest
import numpy as np
from uuid import uuid4

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.domain.models import Library, Document, Chunk, LibraryMetadata, ChunkMetadata
from src.infrastructure.repositories import InMemoryRepository
from src.infrastructure.index_manager import IndexManager
from src.infrastructure.embedders import CohereEmbedder
from src.services.library_service import LibraryService


@pytest.mark.skipif(not os.environ.get("COHERE_API_KEY"), reason="COHERE_API_KEY environment variable not set")
def test_library_service_search():
    # Set up dependencies
    idx_mgr = IndexManager()
    embedder = CohereEmbedder()
    service = LibraryService(index_manager=idx_mgr, embedder=embedder)

    library = Library(
        id=uuid4(),
        metadata=LibraryMetadata(name="Test Library", description="A test library"),
    )
    service.create_library(library)

    # Add a document with chunks
    doc_id = uuid4()
    chunks = [
        Chunk(id=uuid4(), text="The quick brown fox jumps over the lazy dog.", metadata=ChunkMetadata()),
        Chunk(id=uuid4(), text="A fast brown fox leaps over the sleepy canine.", metadata=ChunkMetadata()),
        Chunk(id=uuid4(), text="Machine learning models process data to make predictions.", metadata=ChunkMetadata())
    ]
    document = Document(id=doc_id, chunks=chunks)  # DocumentMetadata will be defaulted
    service.add_document(library.id, document)

    # Perform a search using the new interface
    query_text = "A quick brown animal jumps over a lazy dog."
    top_k = 2
    results = service.search(library.id, query_text, top_k=top_k)

    # Assert we get top_k results
    assert len(results) == top_k

    # Optionally, check that the closest chunk is the most similar
    indexed_chunk_uuids = [chunk.id for chunk, _ in results]
    found_chunks = [chunk for chunk in chunks if chunk.id in indexed_chunk_uuids]
    assert any("fox" in chunk.text for chunk in found_chunks)

    # Print results for debugging
    print("Top results:")
    for chunk, score in results:
        print(f"Chunk UUID: {chunk.id}, Score: {score}")
