"""
Main RAG service that orchestrates the entire pipeline.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.core.config import settings
from app.models.schemas import (
    Document, Chunk, SearchQuery, SearchResult, RAGResponse, 
    DocumentUpload, SystemHealth
)
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import FAISSVectorStore
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGService:
    """Main service that orchestrates the RAG pipeline."""
    
    def __init__(self):
        """Initialize the RAG service."""
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = FAISSVectorStore()
        self.llm_service = LLMService()
        
        # In-memory storage for documents and chunks (in production, use a database)
        self.documents = {}  # document_id -> Document
        self.chunks = {}     # chunk_id -> Chunk
        self.start_time = time.time()
        
        logger.info("RAG service initialized")
    
    def ingest_document(self, file_path: str, title: str, 
                       metadata: Dict[str, Any] = None) -> Document:
        """Ingest a document into the RAG system."""
        try:
            logger.info(f"Starting ingestion of document: {title}")
            
            # 1. Process document
            print(f"Processing document: {file_path}")
            document = self.document_processor.process_document(file_path, title, metadata)
            print(f"Document created with ID: {document.id}")
            
            # 2. Extract text content
            print(f"Extracting text from document...")
            text_content = self.document_processor.extract_text(document)
            print(f"Text extracted, length: {len(text_content)}")
            
            # 3. Chunk the text
            chunks = self.document_processor.chunk_text(text_content, document.id)
            print(f"Created {len(chunks)} chunks")
            logger.info(f"Text content length: {len(text_content)}")
            logger.info(f"Created {len(chunks)} chunks")
            
            # 4. Generate embeddings
            embeddings = self.embedding_service.generate_embeddings(chunks)
            
            # 5. Store in vector store
            self.vector_store.add_chunks(chunks, embeddings)
            
            # 6. Update in-memory storage
            self.documents[document.id] = document
            for chunk in chunks:
                self.chunks[chunk.id] = chunk
            
            # 7. Update document status
            document.status = "completed"
            
            logger.info(f"Successfully ingested document {document.id} with {len(chunks)} chunks")
            return document
            
        except Exception as e:
            logger.error(f"Error ingesting document {title}: {e}")
            if 'document' in locals():
                document.status = "failed"
            raise
    
    def search(self, query: str, top_k: int = 5, 
               filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search for relevant chunks using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_single_embedding(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding, 
                top_k=top_k, 
                filters=filters
            )
            
            logger.info(f"Search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 5, 
                     filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Perform hybrid search combining vector and text similarity."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_single_embedding(query)
            
            # Perform hybrid search
            results = self.vector_store.hybrid_search(
                query_embedding, 
                query, 
                top_k=top_k, 
                filters=filters
            )
            
            logger.info(f"Hybrid search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.search(query, top_k, filters)
    
    def query(self, query: str, top_k: int = 5, 
              filters: Dict[str, Any] = None, 
              use_hybrid: bool = True) -> RAGResponse:
        """Main query method that retrieves and generates responses."""
        start_time = time.time()
        
        try:
            # 1. Search for relevant chunks
            if use_hybrid:
                search_results = self.hybrid_search(query, top_k, filters)
            else:
                search_results = self.search(query, top_k, filters)
            
            # 2. Prepare context from search results
            context_text = self._prepare_context(search_results)
            
            # 3. Generate response using LLM
            answer = self.llm_service.generate_response(query, context_text)
            
            # 4. Create RAG response
            response = RAGResponse(
                answer=answer,
                sources=search_results,
                confidence_score=self._calculate_confidence(search_results),
                processing_time=time.time() - start_time,
                query=query
            )
            
            logger.info(f"Generated response for query '{query}' in {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            # Return error response
            return RAGResponse(
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                query=query
            )
    
    def batch_ingest(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Ingest multiple documents in batch."""
        results = []
        
        for doc_info in documents:
            try:
                document = self.ingest_document(
                    file_path=doc_info['file_path'],
                    title=doc_info['title'],
                    metadata=doc_info.get('metadata', {})
                )
                results.append(document)
                
            except Exception as e:
                logger.error(f"Failed to ingest document {doc_info.get('title', 'Unknown')}: {e}")
                # Continue with other documents
        
        logger.info(f"Batch ingestion completed: {len(results)}/{len(documents)} documents successful")
        return results
    
    def update_document(self, document_id: str, new_file_path: str) -> Document:
        """Update an existing document with new content."""
        try:
            if document_id not in self.documents:
                raise ValueError(f"Document {document_id} not found")
            
            old_document = self.documents[document_id]
            
            # Check if content has changed
            new_checksum = self.document_processor._calculate_checksum(Path(new_file_path))
            if new_checksum == old_document.checksum:
                logger.info(f"Document {document_id} unchanged, no update needed")
                return old_document
            
            # Remove old chunks from vector store
            old_chunks = [chunk for chunk in self.chunks.values() 
                         if chunk.document_id == document_id]
            
            for chunk in old_chunks:
                self.vector_store.delete_chunk(chunk.id)
                del self.chunks[chunk.id]
            
            # Ingest new version
            new_document = self.ingest_document(
                new_file_path, 
                old_document.title, 
                old_document.metadata
            )
            
            # Update document ID mapping
            del self.documents[document_id]
            self.documents[new_document.id] = new_document
            
            logger.info(f"Successfully updated document {document_id}")
            return new_document
            
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            if document_id not in self.documents:
                return False
            
            # Remove chunks from vector store
            chunks_to_delete = [chunk for chunk in self.chunks.values() 
                              if chunk.document_id == document_id]
            
            for chunk in chunks_to_delete:
                self.vector_store.delete_chunk(chunk.id)
                del self.chunks[chunk.id]
            
            # Remove document
            del self.documents[document_id]
            
            logger.info(f"Successfully deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """Prepare context text from search results."""
        if not search_results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            chunk = result.chunk
            context_parts.append(f"Source {i} (Score: {result.similarity_score:.3f}):")
            context_parts.append(f"Content: {chunk.content}")
            context_parts.append(f"Document: {chunk.document_id}")
            if chunk.heading_path:
                context_parts.append(f"Section: {' > '.join(chunk.heading_path)}")
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """Calculate confidence score based on search results."""
        if not search_results:
            return 0.0
        
        # Base confidence on search result scores
        avg_search_score = sum(r.similarity_score for r in search_results) / len(search_results)
        
        # Boost confidence if we have multiple good sources
        source_boost = min(0.2, len([r for r in search_results if r.similarity_score > 0.7]) * 0.05)
        
        confidence = avg_search_score + source_boost
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, confidence))
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        if document_id not in self.documents:
            return None
        
        document = self.documents[document_id]
        chunk_count = len([c for c in self.chunks.values() if c.document_id == document_id])
        
        return {
            'id': document.id,
            'title': document.title,
            'format': document.format.value,
            'status': document.status.value,
            'chunk_count': chunk_count,
            'created_at': document.created_at.isoformat(),
            'updated_at': document.updated_at.isoformat(),
            'metadata': document.metadata
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the system."""
        documents_info = []
        
        for document in self.documents.values():
            chunk_count = len([c for c in self.chunks.values() if c.document_id == document.id])
            
            documents_info.append({
                'id': document.id,
                'title': document.title,
                'format': document.format.value,
                'status': document.status.value,
                'chunk_count': chunk_count,
                'created_at': document.created_at.isoformat(),
                'updated_at': document.updated_at.isoformat()
            })
        
        return documents_info
    
    def get_system_health(self) -> SystemHealth:
        """Get system health and statistics."""
        vector_store_stats = self.vector_store.get_stats()
        
        return SystemHealth(
            status="healthy" if vector_store_stats.get('status') != 'not_initialized' else "initializing",
            version=settings.app_version,
            uptime=time.time() - self.start_time,
            document_count=len(self.documents),
            chunk_count=len(self.chunks),
            vector_store_status=vector_store_stats.get('status', 'unknown'),
            database_status="in_memory"  # Since we're using in-memory storage for MVP
        )
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search and retrieval statistics."""
        vector_store_stats = self.vector_store.get_stats()
        
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'vector_store_stats': vector_store_stats,
            'embedding_model': self.embedding_service.get_model_info(),
            'llm_provider': self.llm_service.get_provider_info()
        }
    
    def clear_system(self):
        """Clear all data from the system."""
        try:
            # Clear vector store
            self.vector_store.clear()
            
            # Clear in-memory storage
            self.documents.clear()
            self.chunks.clear()
            
            logger.info("System cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing system: {e}")
            raise
    
    def export_data(self, export_path: str):
        """Export system data for backup or migration."""
        try:
            import json
            from datetime import datetime
            
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'system_version': settings.app_version,
                'documents': [doc.dict() for doc in self.documents.values()],
                'chunks': [chunk.dict() for chunk in self.chunks.values()],
                'vector_store_stats': self.vector_store.get_stats()
            }
            
            # Ensure export directory exists
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"System data exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise
