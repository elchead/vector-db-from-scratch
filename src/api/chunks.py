from fastapi import APIRouter, HTTPException, status, Depends, Body
from uuid import UUID, uuid4
from typing import List

from src.domain.models import Chunk, ChunkMetadata
from src.api.library_service_provider import get_library_service
from src.services.library_service import LibraryService

router = APIRouter(prefix="/libraries/{library_id}/documents/{document_id}/chunks", tags=["chunks"])

@router.get("/", response_model=List[Chunk])
def list_chunks(
    library_id: UUID,
    document_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    doc = service.get_document(library_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Library or document not found")
    return doc.chunks

@router.get("/{chunk_id}", response_model=Chunk)
def get_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    chunk = service.get_chunk(library_id, document_id, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk

@router.put("/{chunk_id}", response_model=Chunk)
def update_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    text: str = None,
    embedding: List[float] = None,
    source: str = None,
    custom_chunk_metadata: dict = None,
    service: LibraryService = Depends(get_library_service),
):
    updated = service.update_chunk(
        library_id, document_id, chunk_id, text=text, embedding=embedding, source=source, custom_chunk_metadata=custom_chunk_metadata
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return updated

@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chunk(
    library_id: UUID,
    document_id: UUID,
    chunk_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    deleted = service.delete_chunk(library_id, document_id, chunk_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chunk not found")
