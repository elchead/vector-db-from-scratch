import threading
from uuid import UUID
from typing import List, Optional, Dict, Any

from src.domain.models import Library, Document, Chunk, Vector
from src.infrastructure.repositories import InMemoryRepository
from src.infrastructure.index_manager import IndexManager
from src.infrastructure.embedders import CohereEmbedder
from src.infrastructure.index import IndexedChunk

class LibraryService:
    """
    Service to manage libraries, documents, chunks and keep the index in sync.
    """
    def __init__(self,
                 index_manager: IndexManager,
                 embedder: CohereEmbedder):
        self._repos: Dict[UUID, InMemoryRepository] = {} # TODO: inject factory later (dependency injection)
        self.idx_mgr = index_manager
        self.embedder = embedder
        # Global lock for the service
        self._service_lock = threading.RLock()
        # Per-library locks to allow concurrent operations on different libraries
        self._library_locks: Dict[UUID, threading.RLock] = {}

    def _get_library_lock(self, library_id: UUID) -> threading.RLock:
        """Get or create a lock for a specific library"""
        with self._service_lock:
            if library_id not in self._library_locks:
                self._library_locks[library_id] = threading.RLock()
            return self._library_locks[library_id]

    def create_library(self, library: Library) -> Library:
        with self._service_lock:
            if library.id in self._repos:
                raise ValueError(f"Library {library.id} already exists.")
            repo = InMemoryRepository()
            lib = repo.add_library(library)
            self._repos[library.id] = repo
            self.idx_mgr.create_index(lib.id, self.embedder.output_dimension, "brute") # inject factory
            # Create a lock for this library
            self._library_locks[library.id] = threading.RLock()
            return lib

    def get_library(self, library_id: UUID) -> Optional[Library]:
        repo = self._repos.get(library_id)
        if not repo:
            return None
        
        # Use the library lock for this operation
        with self._get_library_lock(library_id):
            return repo.get_library(library_id)

    def list_libraries(self) -> List[Library]:
        # This operation needs to be atomic across all libraries
        with self._service_lock:
            libs = []
            for repo in self._repos.values():
                libs.extend(repo.list_libraries())
            return libs

    def update_library(self,
                       library_id: UUID,
                       name: Optional[str] = None,
                       description: Optional[str] = None,
                       custom_metadata: Optional[Dict[str, Any]] = None) -> Optional[Library]:
        repo = self._repos.get(library_id)
        if not repo:
            return None
        
        # Use the library lock for this operation
        with self._get_library_lock(library_id):
            return repo.update_library(library_id, name, description, custom_metadata)

    def delete_library(self, library_id: UUID) -> bool:
        if library_id not in self._repos:
            return False
        
        # Use both service lock and library lock to ensure atomicity
        with self._service_lock:
            with self._get_library_lock(library_id):
                repo = self._repos[library_id]
                ok = repo.delete_library(library_id)
                if ok:
                    self.idx_mgr.drop_index(library_id)
                    del self._repos[library_id]
                    del self._library_locks[library_id]
                return ok

    def add_document(self, library_id: UUID, document: Document) -> Optional[Document]:
        repo = self._repos.get(library_id)
        if not repo:
            return None
        
        # Use the library lock for the entire operation
        with self._get_library_lock(library_id):
            # Now we can safely perform both repository and index operations
            doc = repo.add_document(library_id, document)
            if doc:
                idx = self.idx_mgr.get(library_id)
                texts = [chunk.text for chunk in doc.chunks]
                embeddings = self.embedder.embed(texts)
                indexed_chunks = [
                    IndexedChunk(id=chunk.id, embedding=embedding)
                    for chunk, embedding in zip(doc.chunks, embeddings)
                ]
                idx.add_many(indexed_chunks)
            return doc

    def get_document(self, library_id: UUID, document_id: UUID) -> Optional[Document]:
        repo = self._repos.get(library_id)
        if not repo:
            return None
        
        # Use the library lock for this operation
        with self._get_library_lock(library_id):
            return repo.get_document(library_id, document_id)

    def list_documents(self, library_id: UUID) -> Optional[List[Document]]:
        repo = self._repos.get(library_id)
        if not repo:
            return None
        
        # Use the library lock for this operation
        with self._get_library_lock(library_id):
            return repo.list_documents(library_id)

    def delete_document(self, library_id: UUID, document_id: UUID) -> bool:
        repo = self._repos.get(library_id)
        if not repo:
            return False
        
        # Use the library lock for the entire operation
        with self._get_library_lock(library_id):
            # Now we can safely perform both repository and index operations
            doc = repo.get_document(library_id, document_id)
            if not doc:
                return False
                
            ok = repo.delete_document(library_id, document_id)
            if ok:
                idx = self.idx_mgr.get(library_id)
                for chk in doc.chunks:
                    idx.remove(chk.id)
            return ok

    def add_chunk(self, library_id: UUID, document_id: UUID, chunk: Chunk) -> Optional[Chunk]:
        repo = self._repos.get(library_id)
        if not repo:
            return None
        
        # Use the library lock for the entire operation
        with self._get_library_lock(library_id):
            # Now we can safely perform both repository and index operations
            chk = repo.add_chunk_to_document(library_id, document_id, chunk)
            if chk:
                embedding = self.embedder.embed_single(chk.text)
                indexed_chunk = IndexedChunk(id=chk.id, embedding=embedding)
                self.idx_mgr.get(library_id).add(indexed_chunk)
            return chk

    def get_chunk(self, library_id: UUID, document_id: UUID, chunk_id: UUID) -> Optional[Chunk]:
        repo = self._repos.get(library_id)
        if not repo:
            return None
        
        # Use the library lock for this operation
        with self._get_library_lock(library_id):
            return repo.get_chunk(library_id, document_id, chunk_id)

    def update_chunk(self,
                     library_id: UUID,
                     document_id: UUID,
                     chunk_id: UUID,
                     text: Optional[str] = None,
                     embedding: Optional[Vector] = None,
                     source: Optional[str] = None,
                     custom_metadata: Optional[Dict[str, Any]] = None) -> Optional[Chunk]:
        repo = self._repos.get(library_id)
        if not repo:
            return None
        
        # Use the library lock for the entire operation
        with self._get_library_lock(library_id):
            # Now we can safely perform both repository and index operations
            updated = repo.update_chunk(
                library_id,
                document_id,
                chunk_id,
                text=text,
                embedding=embedding,
                source=source,
                custom_chunk_metadata=custom_metadata,
            )
            if updated:
                self.idx_mgr.get(library_id).update(updated)
            return updated

    def delete_chunk(self, library_id: UUID, document_id: UUID, chunk_id: UUID) -> bool:
        repo = self._repos.get(library_id)
        if not repo:
            return False
        
        # Use the library lock for the entire operation
        with self._get_library_lock(library_id):
            # Now we can safely perform both repository and index operations
            ok = repo.delete_chunk(library_id, document_id, chunk_id)
            if ok:
                self.idx_mgr.get(library_id).remove(chunk_id)
            return ok

    def _search_vector(self, library_id: UUID, query_vector: Vector, top_k: int = 5):
        with self._get_library_lock(library_id):
            return self.idx_mgr.get(library_id).search(query_vector, top_k)

    def search(self, library_id: UUID, query_text: str, top_k: int = 5):
        """
        Search for the top_k most similar chunks in the given library to the query_text.
        Returns a list of (chunk, similarity_score) tuples.
        """
        query_vector = self.embedder.embed_single(query_text)
        with self._get_library_lock(library_id):
            return self.idx_mgr.get(library_id).search(query_vector, top_k)
