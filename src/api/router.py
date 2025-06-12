from functools import lru_cache
from src.infrastructure.repositories import InMemoryRepository

@lru_cache()
def get_repo():
    return InMemoryRepository()