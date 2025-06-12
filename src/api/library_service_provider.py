from functools import lru_cache
from src.services.library_service import LibraryService
from src.infrastructure.repositories import InMemoryRepository
from src.infrastructure.index_manager import IndexManager
from src.infrastructure.embedders import CohereEmbedder

@lru_cache()
def get_library_service():
    return LibraryService(
        index_manager=IndexManager(),
        embedder=CohereEmbedder(),
    )
