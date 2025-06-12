"""
Embedder wrappers for various embedding providers.
This module provides abstraction over embedding APIs.
"""

import os
from typing import List, Optional, Union
import cohere
import numpy as np


class CohereEmbedder:
    """
    Wrapper for Cohere embedding API that abstracts implementation details.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        input_type: Optional[str] = None,
        embedding_types: Optional[List[str]] = None,
        output_dimension: Optional[int] = 256 # use lower default for performance
    ):
        """
        Initialize the Cohere embedder.

        Args:
            api_key: Cohere API key. If None, will try to get from COHERE_API_KEY env var.
            model: Embedding model to use. If None, use client default.
            input_type: Type of input for embedding. If None, use client default.
            embedding_types: Types of embeddings to return. If None, use client default.
            output_dimension: Dimension of the output embeddings. If None, use client default.
        """
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key must be provided or set as COHERE_API_KEY environment variable")

        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = "embed-v4.0"
        self.input_type = input_type
        self.embedding_types = embedding_types
        self.output_dimension = output_dimension # 256, 512, 1024, or 1536
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for one or more texts.
        
        Args:
            texts: Single text or list of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        # Convert single text to list for consistent handling
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type,
            embedding_types=self.embedding_types,
            output_dimension=self.output_dimension
        )
        
        # The V2 API returns embeddings in a different structure
        # Access the float embeddings using the attribute syntax
        return response.embeddings.float
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        embeddings = self.embed([text])
        return embeddings[0]
