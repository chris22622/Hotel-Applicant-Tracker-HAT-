"""Embedding service using sentence-transformers."""
import numpy as np
from sentence_transformers import SentenceTransformer


# Global model instance
_model = None


def get_model() -> SentenceTransformer:
    """Get sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    """Generate embedding for text."""
    if not text or not text.strip():
        text = "empty resume"
    
    model = get_model()
    # Generate embedding and normalize
    vector = model.encode([text], normalize_embeddings=True)[0].astype(np.float32)
    return vector


def embed_role(role_title: str, role_description: str, must_have: list, nice_to_have: list) -> np.ndarray:
    """Generate embedding for job role."""
    # Combine role information
    role_text = f"{role_title}\n{role_description}"
    
    if must_have:
        role_text += "\nRequired skills: " + ", ".join(must_have)
    
    if nice_to_have:
        role_text += "\nPreferred skills: " + ", ".join(nice_to_have)
    
    return embed(role_text)
