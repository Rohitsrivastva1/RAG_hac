"""
🎓 RAG Service - The Smart Document Assistant! 📚

This is the MAIN BRAIN of our RAG system. Think of it as a super-smart librarian who can:
1. 📖 Read any document you give it (PDF, Word, etc.)
2. 🧠 Understand what's in the document
3. 🔍 Find relevant information when you ask questions
4. 💬 Give you intelligent answers based on the documents
5. 📍 Show you exactly where the information came from

HOW IT WORKS:
- Document Ingestion: Takes your documents and processes them
- Text Extraction: Gets the actual text content
- Chunking: Breaks long documents into smaller, manageable pieces
- Embedding: Converts text into numbers (vectors) that computers can understand
- Storage: Saves everything in a searchable format
- Query Processing: Answers your questions using the stored information

This service coordinates all the other services to make everything work together!
"""
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import all the tools we need
from app.core.config import settings
from app.models.schemas import (
    Document, Chunk, SearchQuery, SearchResult, RAGResponse, 
    DocumentUpload, SystemHealth
)
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import FAISSVectorStore
from app.services.llm_service import LLMService

# Set up logging to track what's happening
logger = logging.getLogger(__name__)


class RAGService:
    """
    🚀 Main RAG Service - The Orchestrator!
    
    This class is like a conductor in an orchestra - it doesn't play the instruments
    itself, but it coordinates all the other services to create beautiful music!
    
    What it does:
    - Manages document processing workflow
    - Coordinates between different services
    - Handles user queries and responses
    - Keeps track of all documents and chunks
    """
    
    def __init__(self):
        """
        🏗️ Initialize the RAG service - Set up all our tools!
        
        This is like setting up a new workspace with all the tools you need:
        - Document Processor: Reads and breaks down documents
        - Embedding Service: Converts text to numbers (vectors)
        - Vector Store: Stores and searches these numbers efficiently
        - LLM Service: Generates human-like responses using AI
        """
        # Initialize all the services we need
        self.document_processor = DocumentProcessor()  # Handles PDF, DOCX, HTML, etc.
        self.embedding_service = EmbeddingService()    # Converts text to vectors
        self.vector_store = FAISSVectorStore()         # Stores and searches vectors
        self.llm_service = LLMService()                # Generates AI responses
        
        # Temporary storage (in production, you'd use a real database)
        # Think of these as filing cabinets where we store information
        self.documents = {}  # document_id -> Document (like a file folder)
        self.chunks = {}     # chunk_id -> Chunk (like individual pages)
        self.start_time = time.time()  # Track when we started
        
        logger.info("🎉 RAG service initialized - Ready to help with your documents!")
    
    def ingest_document(self, file_path: str, title: str, 
                       metadata: Dict[str, Any] = None) -> Document:
        """
        📥 Ingest a document into the RAG system - This is where the magic happens!
        
        Think of this like teaching a computer to read and understand a book:
        1. 📖 We give it a document (PDF, Word, etc.)
        2. 🔍 It reads and extracts all the text
        3. ✂️ It breaks the text into smaller, manageable pieces (chunks)
        4. 🔢 It converts each piece into numbers (embeddings)
        5. 💾 It stores everything so it can find information later
        
        Args:
            file_path: Where the document is located on your computer
            title: What to call this document
            metadata: Extra information about the document (optional)
        
        Returns:
            Document: The processed document with all its information
        """
        try:
            logger.info(f"🚀 Starting ingestion of document: {title}")
            
            # STEP 1: Process the document 📋
            # This creates a Document object with basic information
            print(f"📄 Processing document: {file_path}")
            document = self.document_processor.process_document(file_path, title, metadata)
            print(f"✅ Document created with ID: {document.id}")
            
            # STEP 2: Extract text content 📝
            # Get all the actual text from the document (like copying text from a PDF)
            print(f"📖 Extracting text from document...")
            text_content = self.document_processor.extract_text(document)
            print(f"📊 Text extracted, length: {len(text_content)} characters")
            logger.info(f"Text content length: {len(text_content)}")
            
            # STEP 3: Chunk the text ✂️
            # Break long text into smaller pieces (like chapters in a book)
            # This makes it easier to find specific information later
            chunks = self.document_processor.chunk_text(text_content, document.id)
            print(f"🔪 Created {len(chunks)} chunks (smaller pieces)")
            logger.info(f"Created {len(chunks)} chunks")
            
            # STEP 4: Generate embeddings 🔢
            # Convert each text chunk into numbers (vectors)
            # Think of this like giving each piece a unique "fingerprint"
            print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_service.generate_embeddings(chunks)
            print(f"✅ Generated {len(embeddings)} embeddings")
            
            # STEP 5: Store in vector store 💾
            # Save all the chunks and their embeddings for later searching
            print(f"💾 Storing chunks in vector store...")
            self.vector_store.add_chunks(chunks, embeddings)
            print(f"✅ Stored {len(chunks)} chunks in vector store")
            
            # STEP 6: Update in-memory storage 🧠
            # Keep track of everything in our temporary memory
            # (In a real system, this would go in a database)
            self.documents[document.id] = document  # Store the main document
            for chunk in chunks:
                self.chunks[chunk.id] = chunk      # Store each chunk
            print(f"🧠 Updated memory storage with document and {len(chunks)} chunks")
            
            # STEP 7: Mark as completed ✅
            # Update the document status to show it's been processed
            document.status = "completed"
            
            logger.info(f"🎉 Successfully ingested document {document.id} with {len(chunks)} chunks")
            print(f"🎉 Document '{title}' successfully processed and ready for questions!")
            return document
            
        except Exception as e:
            # If something goes wrong, log the error and mark the document as failed
            logger.error(f"❌ Error ingesting document {title}: {e}")
            if 'document' in locals():
                document.status = "failed"
            raise  # Re-raise the error so the calling code knows something went wrong
    
    def search(self, query: str, top_k: int = 5, 
               filters: Dict[str, Any] = None) -> List[SearchResult]:
        """
        🔍 Search for relevant information in your documents!
        
        This is like asking a librarian to find books about a specific topic:
        1. 📝 You ask a question (query)
        2. 🔢 We convert your question into numbers (embeddings)
        3. 🔍 We search through all stored documents to find similar content
        4. 📊 We rank the results by how relevant they are
        5. 📋 We return the most relevant pieces of information
        
        Args:
            query: Your question or what you're looking for
            top_k: How many results you want (default: 5)
            filters: Optional filters to narrow down the search
        
        Returns:
            List of search results, ranked by relevance
        """
        try:
            print(f"🔍 Searching for: '{query}'")
            
            # STEP 1: Convert your question to numbers 🔢
            # This is like translating your question into a language the computer understands
            print(f"🔢 Converting query to embedding...")
            query_embedding = self.embedding_service.generate_single_embedding(query)
            print(f"✅ Query converted to embedding")
            
            # STEP 2: Search through all stored documents 🔍
            # Find the most similar pieces of information
            print(f"🔍 Searching vector store for similar content...")
            results = self.vector_store.search(
                query_embedding,  # Your question as numbers
                top_k=top_k,      # How many results you want
                filters=filters    # Any filters to narrow down results
            )
            
            print(f"📊 Found {len(results)} relevant results")
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
        """
        💬 Answer your questions using the documents you've uploaded!
        
        This is the MAIN method that makes the RAG system work:
        1. 🔍 You ask a question
        2. 📚 We search through your documents to find relevant information
        3. 🧠 We use AI to generate a smart answer based on that information
        4. 📍 We show you exactly where the answer came from
        5. ⏱️ We track how long it took to answer
        
        Think of it like having a super-smart research assistant who:
        - Knows everything in your documents
        - Can answer any question about them
        - Always cites their sources
        - Works really fast!
        
        Args:
            query: Your question (e.g., "What is machine learning?")
            top_k: How many document pieces to consider (default: 5)
            filters: Optional filters to narrow down results
            use_hybrid: Whether to use hybrid search (recommended: True)
        
        Returns:
            RAGResponse: Your answer with sources and confidence score
        """
        start_time = time.time()  # Start the timer
        
        try:
            print(f"🤔 Processing question: '{query}'")
            
            # STEP 1: Search for relevant information 🔍
            # Look through all your documents to find pieces that might answer your question
            print(f"🔍 Searching for relevant information...")
            if use_hybrid:
                # Hybrid search combines vector search + keyword search for better results
                print(f"🔀 Using hybrid search (vector + keyword)...")
                search_results = self.hybrid_search(query, top_k, filters)
            else:
                # Vector search only - still good but hybrid is usually better
                print(f"🔢 Using vector search only...")
                search_results = self.search(query, top_k, filters)
            
            print(f"📚 Found {len(search_results)} relevant pieces of information")
            
            # STEP 2: Prepare the context 📝
            # Take all the relevant information and format it nicely for the AI
            print(f"📝 Preparing context from search results...")
            context_text = self._prepare_context(search_results)
            print(f"✅ Context prepared")
            
            # STEP 3: Generate the answer using AI 🧠
            # Use a large language model to create a smart, coherent answer
            print(f"🧠 Generating AI response...")
            answer = self.llm_service.generate_response(query, context_text)
            print(f"✅ AI response generated")
            
            # STEP 4: Create the final response 📋
            # Package everything together with metadata
            print(f"📋 Creating final response...")
            response = RAGResponse(
                answer=answer,                    # The AI-generated answer
                sources=search_results,           # Where the information came from
                confidence_score=self._calculate_confidence(search_results),  # How confident we are
                processing_time=time.time() - start_time,  # How long it took
                query=query                       # Your original question
            )
            
            # Log success and return
            total_time = time.time() - start_time
            print(f"⏱️ Total processing time: {total_time:.2f} seconds")
            print(f"🎉 Question answered successfully!")
            logger.info(f"Generated response for query '{query}' in {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            # If something goes wrong, log the error and return a helpful error message
            print(f"❌ Error processing query: {e}")
            logger.error(f"Error processing query '{query}': {e}")
            
            # Return an error response that's still useful to the user
            return RAGResponse(
                answer=f"I encountered an error while processing your query: {str(e)}. Sources:",
                sources=[],  # No sources since we had an error
                confidence_score=0.0,  # Low confidence since something went wrong
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
