"""
Semantic text embeddings using sentence-transformers.
"""
import logging
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating semantic embeddings from text."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Use CPU if CUDA is not available
            if not torch.cuda.is_available():
                self.model = self.model.to('cpu')
                
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # Clean and prepare texts
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=True  # L2 normalize for better cosine similarity
            )
            
            logger.debug(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            NumPy array embedding with shape (embedding_dim,)
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def batch_cosine_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (N, D)
            embeddings2: Second set of embeddings (M, D)
            
        Returns:
            Similarity matrix of shape (N, M)
        """
        if embeddings1.size == 0 or embeddings2.size == 0:
            return np.array([[]])
        
        # Normalize embeddings
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Compute cosine similarity matrix
        return np.dot(embeddings1_norm, embeddings2_norm.T)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and prepare text for embedding.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate very long texts (sentence-transformers has token limits)
        max_chars = 5000  # Approximate limit to stay under token limits
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.debug("Text truncated for embedding")
        
        return text
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.model is None:
            return 0
        return self.model.get_sentence_embedding_dimension()


# Global instance for reuse
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convenience function to embed texts using the global service.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        NumPy array of embeddings
    """
    service = get_embedding_service()
    return service.embed_texts(texts)


def embed_single(text: str) -> np.ndarray:
    """
    Convenience function to embed a single text using the global service.
    
    Args:
        text: Text string to embed
        
    Returns:
        NumPy array embedding
    """
    service = get_embedding_service()
    return service.embed_single(text)
