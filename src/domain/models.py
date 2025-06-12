from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# Type alias for a vector embedding
Vector = List[float]

class BaseMetadata(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator('updated_at', mode='before')
    @classmethod
    def default_updated_at(cls, v, info):
        """Set updated_at to created_at if not explicitly provided during instantiation."""
        if hasattr(info, 'data') and 'created_at' in info.data and v is info.data.get('updated_at'):
            return info.data['created_at']
        return v

class ChunkMetadata(BaseMetadata):
    source: Optional[str] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict) # Ensure it's always a dict

class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)

class DocumentMetadata(BaseMetadata):
    title: Optional[str] = None
    author: Optional[str] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

class LibraryMetadata(BaseMetadata):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

class Library(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    documents: List[Document] = Field(default_factory=list)
    metadata: LibraryMetadata
