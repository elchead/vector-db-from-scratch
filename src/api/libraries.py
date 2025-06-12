from fastapi import APIRouter, HTTPException, status, Depends
from uuid import UUID, uuid4
from typing import List

from src.domain.models import Library, LibraryMetadata
from src.api.library_service_provider import get_library_service
from src.services.library_service import LibraryService
from fastapi import Body

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.post("/", response_model=Library, status_code=status.HTTP_201_CREATED)
def create_library(
    metadata: LibraryMetadata = Body(...),
    service: LibraryService = Depends(get_library_service),
):
    payload = Library(id=uuid4(), metadata=metadata)
    return service.create_library(payload)

@router.get("/", response_model=List[Library])
def list_libraries(service: LibraryService = Depends(get_library_service)):
    return service.list_libraries()

@router.get("/{library_id}", response_model=Library)
def get_library(library_id: UUID, service: LibraryService = Depends(get_library_service)):
    lib = service.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=404, detail="Library not found")
    return lib

@router.put("/{library_id}", response_model=Library)
def update_library(
    library_id: UUID,
    name: str = None,
    description: str = None,
    service: LibraryService = Depends(get_library_service),
):
    lib = service.update_library(library_id, name=name, description=description)
    if not lib:
        raise HTTPException(status_code=404, detail="Library not found")
    return lib

@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_library(library_id: UUID, service: LibraryService = Depends(get_library_service)):
    success = service.delete_library(library_id)
    if not success:
        raise HTTPException(status_code=404, detail="Library not found")