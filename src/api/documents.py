from fastapi import APIRouter, HTTPException, status, Depends
from uuid import UUID, uuid4
from typing import List, Optional

from pydantic import BaseModel, Field

from src.domain.models import Document, DocumentMetadata, Chunk, ChunkMetadata
from src.api.library_service_provider import get_library_service
from src.services.library_service import LibraryService

router = APIRouter(prefix="/libraries/{library_id}/documents", tags=["documents"])

# Request schema for embedding chunk creation
class ChunkCreate(BaseModel):
    text: str
    metadata: Optional[ChunkMetadata] = None

# Request schema for creating a document, optionally with initial chunks
class DocumentCreate(BaseModel):
    chunks: List[ChunkCreate] = Field(default_factory=list)
    metadata: DocumentMetadata

@router.post("/", response_model=Document, status_code=status.HTTP_201_CREATED)
def add_document(
    library_id: UUID,
    payload: DocumentCreate,
    service: LibraryService = Depends(get_library_service),
):
    from src.infrastructure.embedders import CohereEmbedder
    embedder = CohereEmbedder()
    chunks: List[Chunk] = []
    for chunk_req in payload.chunks:
        metadata = chunk_req.metadata or ChunkMetadata()
        chunk_model = Chunk(
            id=uuid4(),
            text=chunk_req.text,
            metadata=metadata
        )
        chunks.append(chunk_model)

    document = Document(
        id=uuid4(),
        chunks=chunks,
        metadata=payload.metadata,
    )

    created = service.add_document(library_id, document)
    if not created:
        raise HTTPException(status_code=404, detail="Library not found")
    return created

@router.get("/", response_model=List[Document])
def list_documents(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    return service.list_documents(library_id)

@router.get("/{document_id}", response_model=Document)
def get_document(
    library_id: UUID,
    document_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    doc = service.get_document(library_id, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    library_id: UUID,
    document_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    success = service.delete_document(library_id, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")