"""
Embedding service for the RAG system.
Generates vector representations for text chunks using various models.
"""
import os
import pickle
from typing import List, Optional, Dict, Any
import numpy as np
import logging

from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.models.schemas import Chunk, Embedding

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing text embeddings."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        
        # Load the embedding model
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            # Fallback to a simpler model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Using fallback embedding model: all-MiniLM-L6-v2")
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[Embedding]:
        """Generate embeddings for a list of chunks."""
        if not chunks:
            return []
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        try:
            # Generate embeddings
            vectors = self.model.encode(texts, convert_to_tensor=False)
            
            # Convert to list format and normalize
            embeddings = []
            for i, chunk in enumerate(chunks):
                vector = vectors[i].tolist()
                vector = self._normalize_vector(vector)
                
                embedding = Embedding(
                    chunk_id=chunk.id,
                    vector=vector,
                    model_name=self.model_name
                )
                embeddings.append(embedding)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            vector = self.model.encode([text], convert_to_tensor=False)[0]
            return self._normalize_vector(vector.tolist())
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise
    
    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings in batches for better memory management."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                batch_vectors = self.model.encode(batch_texts, convert_to_tensor=False)
                
                for vector in batch_vectors:
                    normalized_vector = self._normalize_vector(vector.tolist())
                    all_embeddings.append(normalized_vector)
                
                logger.debug(f"Processed batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add zero vectors for failed batches
                for _ in batch_texts:
                    all_embeddings.append([0.0] * self.dimension)
        
        return all_embeddings
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize vector to unit length (L2 normalization)."""
        vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            v1 = np.array(vec1, dtype=np.float32)
            v2 = np.array(vec2, dtype=np.float32)
            
            # Since vectors are normalized, cosine similarity is just dot product
            similarity = np.dot(v1, v2)
            
            # Clamp to [-1, 1] range to handle floating point errors
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def compute_batch_similarity(self, query_vector: List[float], 
                                candidate_vectors: List[List[float]]) -> List[float]:
        """Compute similarities between query vector and multiple candidate vectors."""
        try:
            query_vec = np.array(query_vector, dtype=np.float32)
            candidate_matrix = np.array(candidate_vectors, dtype=np.float32)
            
            # Compute dot products (cosine similarity for normalized vectors)
            similarities = np.dot(candidate_matrix, query_vec)
            
            # Clamp to [-1, 1] range
            similarities = np.clip(similarities, -1.0, 1.0)
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error computing batch similarity: {e}")
            return [0.0] * len(candidate_vectors)
    
    def save_embeddings(self, embeddings: List[Embedding], file_path: str):
        """Save embeddings to a file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert to serializable format
            serializable_embeddings = []
            for emb in embeddings:
                serializable_embeddings.append({
                    'chunk_id': emb.chunk_id,
                    'vector': emb.vector,
                    'model_name': emb.model_name,
                    'created_at': emb.created_at.isoformat()
                })
            
            with open(file_path, 'wb') as f:
                pickle.dump(serializable_embeddings, f)
            
            logger.info(f"Saved {len(embeddings)} embeddings to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    def load_embeddings(self, file_path: str) -> List[Embedding]:
        """Load embeddings from a file."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Embeddings file not found: {file_path}")
                return []
            
            with open(file_path, 'rb') as f:
                serialized_data = pickle.load(f)
            
            # Convert back to Embedding objects
            embeddings = []
            for data in serialized_data:
                from datetime import datetime
                embedding = Embedding(
                    chunk_id=data['chunk_id'],
                    vector=data['vector'],
                    model_name=data['model_name'],
                    created_at=datetime.fromisoformat(data['created_at'])
                )
                embeddings.append(embedding)
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {file_path}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
        }
    
    def update_model(self, new_model_name: str):
        """Update the embedding model."""
        try:
            old_model = self.model_name
            self.model = SentenceTransformer(new_model_name)
            self.model_name = new_model_name
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Updated embedding model from {old_model} to {new_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to update embedding model: {e}")
            raise
