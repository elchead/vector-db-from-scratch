from pydantic import BaseModel, Field
from uuid import UUID
from typing import List

class IndexedChunk(BaseModel):
    model_config = {"frozen": True}
    id: UUID = Field(...)
    embedding: List[float] = Field(...)
