from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List
from uuid import UUID

from src.services.library_service import LibraryService
from src.api.library_service_provider import get_library_service
from src.domain.models import Chunk

class SearchRequest(BaseModel):
    query: str
    top_k: int



router = APIRouter(prefix="/libraries/{library_id}/search", tags=["search"])

class SearchResponse(BaseModel):
    results: List[Chunk]

@router.post("/", response_model=SearchResponse)
def search(
    library_id: UUID,
    req: SearchRequest,
    service: LibraryService = Depends(get_library_service),
):
    try:
        hits = service.search(library_id, req.query, req.top_k)
    except KeyError:
        raise HTTPException(status_code=404, detail="Library not found")
    # hits is List[Tuple[Chunk, float]]; build SearchResult objects
    # hits contains (IndexedChunk, score), need to retrieve the full Chunk for each
    library = service.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    chunk_map = {}
    for doc in library.documents:
        for chk in doc.chunks:
            chunk_map[chk.id] = chk
    # Return a list of Chunk objects (matching test expectation and response_model)
    results = []
    for indexed_chunk, score in hits:
        chunk = chunk_map.get(indexed_chunk.id)
        if chunk:
            results.append(chunk)
    return SearchResponse(results=results)
