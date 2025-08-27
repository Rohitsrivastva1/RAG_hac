"""
Vector store service using FAISS for efficient similarity search.
"""
import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from app.core.config import settings
from app.models.schemas import Chunk, Embedding, SearchResult
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, index_path: str = None):
        """Initialize the FAISS vector store."""
        self.index_path = index_path or settings.faiss_index_path
        self.dimension = settings.embedding_dimension
        self.index = None
        self.chunk_map = {}  # chunk_id -> Chunk mapping
        self.embedding_map = {}  # chunk_id -> embedding vector mapping
        
        # Initialize or load the index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load the FAISS index."""
        try:
            if os.path.exists(self.index_path):
                self._load_index()
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            else:
                self._create_index()
                logger.info(f"Created new FAISS index at {self.index_path}")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            self._create_index()
    
    def _create_index(self):
        """Create a new FAISS index."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # For small datasets and development, use simple L2 index
            # This avoids clustering issues with small numbers of vectors
            self.index = faiss.IndexFlatL2(self.dimension)
            
            logger.info(f"Created FAISS index with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            # Fallback to simple index
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def _load_index(self):
        """Load existing FAISS index and chunk mappings."""
        try:
            # Load the FAISS index
            index_file = f"{self.index_path}.faiss"
            if os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
            
            # Load chunk mappings
            chunk_map_file = f"{self.index_path}_chunks.pkl"
            if os.path.exists(chunk_map_file):
                with open(chunk_map_file, 'rb') as f:
                    self.chunk_map = pickle.load(f)
            
            # Load embedding mappings
            embedding_map_file = f"{self.index_path}_embeddings.pkl"
            if os.path.exists(embedding_map_file):
                with open(embedding_map_file, 'rb') as f:
                    self.embedding_map = pickle.load(f)
            
            # Restore vectors to the FAISS index to fix ntotal count
            if self.chunk_map and self.embedding_map:
                vectors = []
                for chunk_id, embedding_vector in self.embedding_map.items():
                    if chunk_id in self.chunk_map:
                        vectors.append(embedding_vector)
                
                if vectors:
                    vectors_array = np.array(vectors, dtype=np.float32)
                    # Clear the index first to avoid duplicates
                    self.index.reset()
                    # Add all vectors back
                    self.index.add(vectors_array)
                    logger.info(f"Restored {len(vectors)} vectors to FAISS index")
            
            logger.info(f"Loaded {len(self.chunk_map)} chunks and {len(self.embedding_map)} embeddings")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self._create_index()
    
    def add_chunks(self, chunks: List[Chunk], embeddings: List[Embedding]):
        """Add chunks and their embeddings to the vector store."""
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided")
            return
        
        try:
            # Prepare vectors for FAISS
            vectors = []
            valid_chunks = []
            valid_embeddings = []
            
            for chunk, embedding in zip(chunks, embeddings):
                if chunk.id == embedding.chunk_id and embedding.vector:
                    vectors.append(embedding.vector)
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
            
            if not vectors:
                logger.warning("No valid vectors to add")
                return
            
            # Convert to numpy array
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Add to FAISS index
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                # Train the index if needed (for IVF indices)
                self.index.train(vectors_array)
            
            self.index.add(vectors_array)
            
            # Update mappings
            for chunk, embedding in zip(valid_chunks, valid_embeddings):
                self.chunk_map[chunk.id] = chunk
                self.embedding_map[chunk.id] = embedding.vector
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
            
            # Save the updated index
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def search(self, query_vector: List[float], top_k: int = 5, 
               filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search for similar chunks using vector similarity."""
        try:
            if not self.index or self.index.ntotal == 0:
                logger.warning("Vector store is empty")
                return []
            
            # Convert query vector to numpy array
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Search the index
            similarities, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
            
            # Convert results to SearchResult objects
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                # Get chunk ID from the index mapping
                chunk_id = list(self.chunk_map.keys())[idx]
                chunk = self.chunk_map[chunk_id]
                
                # Apply filters if specified
                if filters and not self._apply_filters(chunk, filters):
                    continue
                
                # Convert similarity score (FAISS returns L2 distance, convert to similarity)
                similarity_score = self._l2_to_similarity(similarity)
                
                result = SearchResult(
                    chunk=chunk,
                    similarity_score=similarity_score,
                    rank=i + 1,
                    metadata={'l2_distance': float(similarity)}
                )
                results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def hybrid_search(self, query_vector: List[float], query_text: str, 
                     top_k: int = 5, filters: Dict[str, Any] = None,
                     vector_weight: float = 0.7) -> List[SearchResult]:
        """Hybrid search combining vector similarity and text matching."""
        try:
            # Vector search
            vector_results = self.search(query_vector, top_k * 2, filters)
            
            # Simple text matching (BM25-like scoring)
            text_results = self._text_search(query_text, top_k * 2, filters)
            
            # Combine and rerank results
            combined_results = self._combine_results(
                vector_results, text_results, vector_weight
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.search(query_vector, top_k, filters)
    
    def _text_search(self, query_text: str, top_k: int, 
                     filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Simple text-based search using keyword matching."""
        query_terms = query_text.lower().split()
        results = []
        
        for chunk_id, chunk in self.chunk_map.items():
            # Apply filters
            if filters and not self._apply_filters(chunk, filters):
                continue
            
            # Calculate text similarity score
            chunk_text = chunk.content.lower()
            term_matches = sum(1 for term in query_terms if term in chunk_text)
            
            if term_matches > 0:
                # Simple TF-like scoring
                score = term_matches / len(query_terms)
                
                result = SearchResult(
                    chunk=chunk,
                    similarity_score=score,
                    rank=0,  # Will be updated after combination
                    metadata={'text_matches': term_matches}
                )
                results.append(result)
        
        # Sort by text similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def _combine_results(self, vector_results: List[SearchResult], 
                        text_results: List[SearchResult], 
                        vector_weight: float) -> List[SearchResult]:
        """Combine and rerank vector and text search results."""
        # Create a mapping of chunk_id to results
        combined_map = {}
        
        # Add vector results
        for result in vector_results:
            combined_map[result.chunk.id] = {
                'chunk': result.chunk,
                'vector_score': result.similarity_score,
                'text_score': 0.0,
                'metadata': result.metadata
            }
        
        # Add text results
        for result in text_results:
            if result.chunk.id in combined_map:
                combined_map[result.chunk.id]['text_score'] = result.similarity_score
            else:
                combined_map[result.chunk.id] = {
                    'chunk': result.chunk,
                    'vector_score': 0.0,
                    'text_score': result.similarity_score,
                    'metadata': result.metadata
                }
        
        # Combine scores and create final results
        combined_results = []
        for chunk_id, data in combined_map.items():
            combined_score = (vector_weight * data['vector_score'] + 
                           (1 - vector_weight) * data['text_score'])
            
            result = SearchResult(
                chunk=data['chunk'],
                similarity_score=combined_score,
                rank=0,  # Will be updated after sorting
                metadata={
                    'vector_score': data['vector_score'],
                    'text_score': data['text_score'],
                    'combined_score': combined_score,
                    **data['metadata']
                }
            )
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results
    
    def _apply_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to a chunk."""
        for key, value in filters.items():
            if key == 'document_id' and chunk.document_id != value:
                return False
            elif key == 'chunk_type' and chunk.chunk_type != value:
                return False
            elif key == 'language' and chunk.metadata.get('language') != value:
                return False
            # Add more filter types as needed
        
        return True
    
    def _l2_to_similarity(self, l2_distance: float) -> float:
        """Convert L2 distance to similarity score (0-1)."""
        # Convert L2 distance to similarity using exponential decay
        # This is a simple heuristic - you might want to tune this
        similarity = np.exp(-l2_distance)
        return float(similarity)
    
    def _save_index(self):
        """Save the FAISS index and mappings."""
        try:
            # Save FAISS index
            index_file = f"{self.index_path}.faiss"
            faiss.write_index(self.index, index_file)
            
            # Save chunk mappings
            chunk_map_file = f"{self.index_path}_chunks.pkl"
            with open(chunk_map_file, 'wb') as f:
                pickle.dump(self.chunk_map, f)
            
            # Save embedding mappings
            embedding_map_file = f"{self.index_path}_embeddings.pkl"
            with open(embedding_map_file, 'wb') as f:
                pickle.dump(self.embedding_map, f)
            
            logger.info(f"Saved FAISS index and mappings to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.index:
            return {'status': 'not_initialized'}
        
        return {
            'total_vectors': int(self.index.ntotal),
            'dimension': int(self.index.d),
            'chunk_count': len(self.chunk_map),
            'embedding_count': len(self.embedding_map),
            'index_type': type(self.index).__name__,
            'is_trained': getattr(self.index, 'is_trained', True)
        }
    
    def clear(self):
        """Clear all data from the vector store."""
        try:
            self.index.reset()
            self.chunk_map.clear()
            self.embedding_map.clear()
            
            # Remove saved files
            for ext in ['.faiss', '_chunks.pkl', '_embeddings.pkl']:
                file_path = f"{self.index_path}{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            logger.info("Cleared vector store")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    def delete_chunk(self, chunk_id: str):
        """Delete a specific chunk from the vector store."""
        try:
            if chunk_id in self.chunk_map:
                # Note: FAISS doesn't support efficient deletion
                # This is a simplified implementation
                del self.chunk_map[chunk_id]
                if chunk_id in self.embedding_map:
                    del self.embedding_map[chunk_id]
                
                logger.info(f"Deleted chunk {chunk_id}")
                
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
