"""
Storage management for ML models and FAISS indices.
"""
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)

class ModelStore:
    """Storage manager for ML models and vector indices."""
    
    def __init__(self, base_path: str = "./var/models"):
        """
        Initialize model store.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def get_role_path(self, role_id: str) -> Path:
        """Get storage path for a specific role."""
        role_path = self.base_path / role_id
        role_path.mkdir(parents=True, exist_ok=True)
        return role_path
    
    def save_model(self, role_id: str, model: Any, model_type: str = "ranker") -> str:
        """
        Save a trained model.
        
        Args:
            role_id: Role identifier
            model: Trained model object
            model_type: Type of model (ranker, classifier, etc.)
            
        Returns:
            Path to saved model file
        """
        role_path = self.get_role_path(role_id)
        model_file = role_path / f"{model_type}_model.joblib"
        
        try:
            joblib.dump(model, model_file)
            logger.info(f"Model saved: {model_file}")
            return str(model_file)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, role_id: str, model_type: str = "ranker") -> Any:
        """
        Load a trained model.
        
        Args:
            role_id: Role identifier
            model_type: Type of model
            
        Returns:
            Loaded model object
        """
        role_path = self.get_role_path(role_id)
        model_file = role_path / f"{model_type}_model.joblib"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        try:
            model = joblib.load(model_file)
            logger.info(f"Model loaded: {model_file}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def save_faiss_index(self, role_id: str, index, embedding_ids: list) -> str:
        """
        Save FAISS index and associated metadata.
        
        Args:
            role_id: Role identifier
            index: FAISS index object
            embedding_ids: List of IDs corresponding to index entries
            
        Returns:
            Path to saved index
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
        
        role_path = self.get_role_path(role_id)
        index_file = role_path / "embeddings.index"
        metadata_file = role_path / "embeddings_metadata.json"
        
        try:
            # Save FAISS index
            faiss.write_index(index, str(index_file))
            
            # Save metadata
            metadata = {
                'embedding_ids': embedding_ids,
                'index_size': index.ntotal,
                'dimension': index.d
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"FAISS index saved: {index_file}")
            return str(index_file)
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    def load_faiss_index(self, role_id: str) -> tuple:
        """
        Load FAISS index and metadata.
        
        Args:
            role_id: Role identifier
            
        Returns:
            Tuple of (index, embedding_ids)
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
        
        role_path = self.get_role_path(role_id)
        index_file = role_path / "embeddings.index"
        metadata_file = role_path / "embeddings_metadata.json"
        
        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"FAISS index not found for role: {role_id}")
        
        try:
            # Load FAISS index
            index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            embedding_ids = metadata['embedding_ids']
            
            logger.info(f"FAISS index loaded: {index_file}")
            return index, embedding_ids
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    def save_embeddings(self, role_id: str, embeddings: np.ndarray, 
                       embedding_ids: list, metadata: Optional[Dict] = None) -> str:
        """
        Save embeddings as numpy array with metadata.
        
        Args:
            role_id: Role identifier
            embeddings: Embedding matrix
            embedding_ids: List of IDs
            metadata: Optional metadata
            
        Returns:
            Path to saved embeddings
        """
        role_path = self.get_role_path(role_id)
        embeddings_file = role_path / "embeddings.npy"
        metadata_file = role_path / "embeddings_info.json"
        
        try:
            # Save embeddings
            np.save(embeddings_file, embeddings)
            
            # Save metadata
            info = {
                'embedding_ids': embedding_ids,
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype)
            }
            if metadata:
                info.update(metadata)
            
            with open(metadata_file, 'w') as f:
                json.dump(info, f)
            
            logger.info(f"Embeddings saved: {embeddings_file}")
            return str(embeddings_file)
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise
    
    def load_embeddings(self, role_id: str) -> tuple:
        """
        Load embeddings and metadata.
        
        Args:
            role_id: Role identifier
            
        Returns:
            Tuple of (embeddings, embedding_ids, metadata)
        """
        role_path = self.get_role_path(role_id)
        embeddings_file = role_path / "embeddings.npy"
        metadata_file = role_path / "embeddings_info.json"
        
        if not embeddings_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Embeddings not found for role: {role_id}")
        
        try:
            # Load embeddings
            embeddings = np.load(embeddings_file)
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                info = json.load(f)
            
            embedding_ids = info['embedding_ids']
            metadata = {k: v for k, v in info.items() if k not in ['embedding_ids', 'shape', 'dtype']}
            
            logger.info(f"Embeddings loaded: {embeddings_file}")
            return embeddings, embedding_ids, metadata
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def save_metrics(self, role_id: str, metrics: Dict[str, Any]) -> str:
        """
        Save training metrics.
        
        Args:
            role_id: Role identifier
            metrics: Metrics dictionary
            
        Returns:
            Path to saved metrics
        """
        role_path = self.get_role_path(role_id)
        metrics_file = role_path / "metrics.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Metrics saved: {metrics_file}")
            return str(metrics_file)
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            raise
    
    def load_metrics(self, role_id: str) -> Dict[str, Any]:
        """
        Load training metrics.
        
        Args:
            role_id: Role identifier
            
        Returns:
            Metrics dictionary
        """
        role_path = self.get_role_path(role_id)
        metrics_file = role_path / "metrics.json"
        
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics not found for role: {role_id}")
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            logger.info(f"Metrics loaded: {metrics_file}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            raise
    
    def model_exists(self, role_id: str, model_type: str = "ranker") -> bool:
        """Check if a model exists for the given role."""
        role_path = self.get_role_path(role_id)
        model_file = role_path / f"{model_type}_model.joblib"
        return model_file.exists()
    
    def index_exists(self, role_id: str) -> bool:
        """Check if a FAISS index exists for the given role."""
        if not FAISS_AVAILABLE:
            return False
        
        role_path = self.get_role_path(role_id)
        index_file = role_path / "embeddings.index"
        metadata_file = role_path / "embeddings_metadata.json"
        return index_file.exists() and metadata_file.exists()
    
    def embeddings_exist(self, role_id: str) -> bool:
        """Check if embeddings exist for the given role."""
        role_path = self.get_role_path(role_id)
        embeddings_file = role_path / "embeddings.npy"
        metadata_file = role_path / "embeddings_info.json"
        return embeddings_file.exists() and metadata_file.exists()
    
    def delete_role_data(self, role_id: str) -> None:
        """Delete all data for a specific role."""
        role_path = self.get_role_path(role_id)
        
        if role_path.exists():
            import shutil
            shutil.rmtree(role_path)
            logger.info(f"Deleted all data for role: {role_id}")
    
    def list_roles(self) -> list:
        """List all roles with stored models."""
        if not self.base_path.exists():
            return []
        
        roles = []
        for path in self.base_path.iterdir():
            if path.is_dir():
                roles.append(path.name)
        
        return sorted(roles)
    
    def get_role_info(self, role_id: str) -> Dict[str, Any]:
        """Get information about stored data for a role."""
        info = {
            'role_id': role_id,
            'model_exists': self.model_exists(role_id),
            'index_exists': self.index_exists(),
            'embeddings_exist': self.embeddings_exist(role_id)
        }
        
        # Add metrics if available
        try:
            metrics = self.load_metrics(role_id)
            info['metrics'] = metrics
        except FileNotFoundError:
            info['metrics'] = None
        
        return info


class VectorStore:
    """Simplified vector store for embeddings without FAISS dependency."""
    
    def __init__(self, role_id: str, store: ModelStore):
        self.role_id = role_id
        self.store = store
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_ids: Optional[list] = None
        self.metadata: Optional[Dict] = None
    
    def add_embeddings(self, embeddings: np.ndarray, ids: list) -> None:
        """Add embeddings to the store."""
        if self.embeddings is None:
            self.embeddings = embeddings
            self.embedding_ids = ids
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.embedding_ids.extend(ids)
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10) -> list:
        """Search for similar embeddings using cosine similarity."""
        if self.embeddings is None:
            return []
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results with similarities
        results = []
        for idx in top_indices:
            results.append({
                'id': self.embedding_ids[idx],
                'similarity': float(similarities[idx]),
                'embedding': self.embeddings[idx]
            })
        
        return results
    
    def save(self) -> None:
        """Save embeddings to disk."""
        if self.embeddings is not None and self.embedding_ids is not None:
            self.store.save_embeddings(self.role_id, self.embeddings, self.embedding_ids)
    
    def load(self) -> None:
        """Load embeddings from disk."""
        try:
            self.embeddings, self.embedding_ids, self.metadata = self.store.load_embeddings(self.role_id)
        except FileNotFoundError:
            self.embeddings = None
            self.embedding_ids = None
            self.metadata = None


# Global store instance
_model_store: Optional[ModelStore] = None


def get_model_store() -> ModelStore:
    """Get the global model store instance."""
    global _model_store
    if _model_store is None:
        _model_store = ModelStore()
    return _model_store
