from typing import Dict, Union
from uuid import UUID

from src.infrastructure.brute_force_index import BruteForceIndex
from src.infrastructure.lsh_index import LSHIndex

class IndexManager:
    """
    Manages per-library vector indexes.
    """
    def __init__(self):
        self._indexes: Dict[UUID, Union[BruteForceIndex, KDTreeIndex]] = {}

    def create_index(self, library_id: UUID, embedding_dim: int, algorithm: str = "brute"):
        """
        Create a new index for the given library.
        """
        if algorithm == "brute":
            idx = BruteForceIndex()
        elif algorithm == "lsh":
            idx = LSHIndex(embedding_dim)
        else:
            raise ValueError(f"Unknown index algorithm: {algorithm}")
        self._indexes[library_id] = idx
        return idx

    def drop_index(self, library_id: UUID):
        """
        Remove an existing index.
        """
        self._indexes.pop(library_id, None)

    def get(self, library_id: UUID):
        """
        Retrieve the index for a library. Raises KeyError if not found.
        """
        return self._indexes[library_id]
