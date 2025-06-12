from fastapi import FastAPI
from src.api.libraries import router as libraries_router
from src.api.documents import router as documents_router
from src.api.chunks import router as chunks_router
from src.api.search import router as search_router

app = FastAPI()
app.include_router(libraries_router)
app.include_router(documents_router)
app.include_router(chunks_router)
app.include_router(search_router)