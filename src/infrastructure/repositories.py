from typing import Dict, List, Optional, Tuple
from uuid import UUID
from datetime import datetime

from src.domain.models import Library, Document, Chunk, Vector

# WARNING: This repository is NOT thread-safe! 
class InMemoryRepository:
    def __init__(self):
        self._libraries: Dict[UUID, Library] = {}

    def _update_timestamp(self, model_instance):
        # Ensure metadata exists before trying to update it
        if hasattr(model_instance, 'metadata') and model_instance.metadata is not None:
            model_instance.metadata.updated_at = datetime.utcnow()
        elif isinstance(model_instance, Chunk): # Chunks have metadata directly
            model_instance.metadata.updated_at = datetime.utcnow()

    # Library CRUD
    def add_library(self, library: Library) -> Library:
        if library.id in self._libraries:
            raise ValueError(f"Library with ID {library.id} already exists.")
        now = datetime.utcnow()
        library.metadata.created_at = library.metadata.created_at or now
        library.metadata.updated_at = library.metadata.updated_at or library.metadata.created_at or now
        self._libraries[library.id] = library
        return library.copy(deep=True)

    def get_library(self, library_id: UUID) -> Optional[Library]:
        library = self._libraries.get(library_id)
        return library.copy(deep=True) if library else None

    def list_libraries(self) -> List[Library]:
        return [lib.copy(deep=True) for lib in self._libraries.values()]

    def update_library(self, library_id: UUID, name: Optional[str] = None, description: Optional[str] = None, custom_metadata: Optional[Dict] = None) -> Optional[Library]:
        library = self._libraries.get(library_id)
        if not library:
            return None
        updated = False
        if name is not None and library.metadata.name != name:
            library.metadata.name = name
            updated = True
        if description is not None and library.metadata.description != description:
            library.metadata.description = description
            updated = True
        if custom_metadata is not None and library.metadata.custom_metadata != custom_metadata:
            library.metadata.custom_metadata.update(custom_metadata) # Merge or replace based on desired behavior, here it updates
            updated = True
        if updated:
            self._update_timestamp(library)
        return library.copy(deep=True)

    def delete_library(self, library_id: UUID) -> bool:
        if library_id in self._libraries:
            del self._libraries[library_id]
            return True
        return False

    # Document helpers
    def _find_document_in_library(self, library: Library, document_id: UUID) -> Optional[Tuple[Document, int]]:
        for idx, doc in enumerate(library.documents):
            if doc.id == document_id:
                return doc, idx
        return None

    # Chunk CRUD (within a document, within a library)
    def add_chunk_to_document(self, library_id: UUID, document_id: UUID, chunk: Chunk) -> Optional[Chunk]:
        library = self._libraries.get(library_id)
        if not library:
            return None

        doc_info = self._find_document_in_library(library, document_id)
        if not doc_info:
            return None
        document, _ = doc_info

        if any(c.id == chunk.id for c in document.chunks):
            raise ValueError(f"Chunk with ID {chunk.id} already exists in document {document_id}.")

        now = datetime.utcnow()
        chunk.metadata.created_at = chunk.metadata.created_at or now
        chunk.metadata.updated_at = chunk.metadata.updated_at or chunk.metadata.created_at or now

        document.chunks.append(chunk)
        self._update_timestamp(document)
        self._update_timestamp(library)
        return chunk.copy(deep=True)

    def get_chunk(self, library_id: UUID, document_id: UUID, chunk_id: UUID) -> Optional[Chunk]:
        library = self._libraries.get(library_id)
        if not library:
            return None
        doc_info = self._find_document_in_library(library, document_id)
        if not doc_info:
            return None
        document, _ = doc_info
        for chk in document.chunks:
            if chk.id == chunk_id:
                return chk.copy(deep=True)
        return None

    def update_chunk(self, library_id: UUID, document_id: UUID, chunk_id: UUID, 
                     text: Optional[str] = None, embedding: Optional[Vector] = None, 
                     source: Optional[str] = None, custom_chunk_metadata: Optional[Dict] = None) -> Optional[Chunk]:
        library = self._libraries.get(library_id)
        if not library:
            return None
        doc_info = self._find_document_in_library(library, document_id)
        if not doc_info:
            return None
        document, _ = doc_info

        chunk_to_update = None
        for chk in document.chunks:
            if chk.id == chunk_id:
                chunk_to_update = chk
                break

        if not chunk_to_update:
            return None

        updated = False
        if text is not None and chunk_to_update.text != text:
            chunk_to_update.text = text
            updated = True
        if embedding is not None and chunk_to_update.embedding != embedding:
            chunk_to_update.embedding = embedding
            updated = True
        if source is not None and chunk_to_update.metadata.source != source:
            chunk_to_update.metadata.source = source
            updated = True
        if custom_chunk_metadata is not None and chunk_to_update.metadata.custom_metadata != custom_chunk_metadata:
            chunk_to_update.metadata.custom_metadata.update(custom_chunk_metadata)
            updated = True

        if updated:
            self._update_timestamp(chunk_to_update)
            self._update_timestamp(document)
            self._update_timestamp(library)
        return chunk_to_update.copy(deep=True)

    def delete_chunk(self, library_id: UUID, document_id: UUID, chunk_id: UUID) -> bool:
        library = self._libraries.get(library_id)
        if not library:
            return False
        doc_info = self._find_document_in_library(library, document_id)
        if not doc_info:
            return False
        document, _ = doc_info

        initial_len = len(document.chunks)
        document.chunks = [chk for chk in document.chunks if chk.id != chunk_id]

        if len(document.chunks) < initial_len:
            self._update_timestamp(document)
            self._update_timestamp(library)
            return True
        return False

    # Document specific CRUD
    def add_document(self, library_id: UUID, document: Document) -> Optional[Document]:
        library = self._libraries.get(library_id)
        if not library:
            return None

        if any(d.id == document.id for d in library.documents):
            raise ValueError(f"Document with ID {document.id} already exists in library {library_id}.")

        now = datetime.utcnow()
        document.metadata.created_at = document.metadata.created_at or now
        document.metadata.updated_at = document.metadata.updated_at or document.metadata.created_at or now
        for chunk in document.chunks:
            chunk.metadata.created_at = chunk.metadata.created_at or now
            chunk.metadata.updated_at = chunk.metadata.updated_at or chunk.metadata.created_at or now

        library.documents.append(document)
        self._update_timestamp(library)
        return document.copy(deep=True)

    def get_document(self, library_id: UUID, document_id: UUID) -> Optional[Document]:
        library = self._libraries.get(library_id)
        if not library:
            return None
        doc_info = self._find_document_in_library(library, document_id)
        return doc_info[0].copy(deep=True) if doc_info else None

    def list_documents(self, library_id: UUID) -> Optional[List[Document]]:
        library = self._libraries.get(library_id)
        if not library:
            return None
        return [doc.copy(deep=True) for doc in library.documents]

    def delete_document(self, library_id: UUID, document_id: UUID) -> bool:
        library = self._libraries.get(library_id)
        if not library:
            return False

        initial_len = len(library.documents)
        library.documents = [doc for doc in library.documents if doc.id != document_id]

        if len(library.documents) < initial_len:
            self._update_timestamp(library)
            return True
        return False

    def clear_all_data(self):
        self._libraries.clear()
