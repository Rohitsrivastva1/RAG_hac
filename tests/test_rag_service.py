import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from app.services.rag_service import RAGService
from app.models.schemas import Document, Chunk, SearchQuery, RAGResponse


class TestRAGService:
    @pytest.fixture
    def rag_service(self):
        return RAGService()
    
    @pytest.fixture
    def sample_document(self):
        return Document(
            id="doc1",
            title="Test Document",
            content="# Introduction\nThis is a test document about machine learning.",
            format="markdown",
            source="test.md",
            checksum="abc123",
            status="processed",
            metadata={"topic": "ml"}
        )

    def test_initialization(self, rag_service):
        """Test RAG service initialization"""
        assert rag_service.document_processor is not None
        assert rag_service.embedding_service is not None
        assert rag_service.vector_store is not None
        assert rag_service.llm_service is not None
        assert rag_service.documents == {}
        assert rag_service.chunks == {}

    def test_ingest_document(self, rag_service, sample_document):
        """Test document ingestion"""
        result = rag_service.ingest_document(sample_document)
        
        assert result.success is True
        assert result.document_id == "doc1"
        assert result.chunks_created > 0
        
        # Check that document and chunks are stored
        assert "doc1" in rag_service.documents
        assert len(rag_service.chunks) > 0

    def test_search_basic(self, rag_service, sample_document):
        """Test basic search functionality"""
        # First ingest a document
        rag_service.ingest_document(sample_document)
        
        # Search for content
        query = "machine learning"
        results = rag_service.search(query, top_k=5)
        
        assert len(results) > 0
        assert results[0].chunk_id is not None
        assert results[0].similarity > 0

    def test_hybrid_search(self, rag_service, sample_document):
        """Test hybrid search functionality"""
        # First ingest a document
        rag_service.ingest_document(sample_document)
        
        # Search with hybrid approach
        query = "machine learning"
        results = rag_service.hybrid_search(query, top_k=5)
        
        assert len(results) > 0
        assert results[0].chunk_id is not None

    def test_query_generation(self, rag_service, sample_document):
        """Test RAG query generation"""
        # First ingest a document
        rag_service.ingest_document(sample_document)
        
        # Query the system
        query = "What is machine learning?"
        response = rag_service.query(query)
        
        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert response.confidence > 0
        assert len(response.sources) > 0

    def test_batch_ingest(self, rag_service):
        """Test batch document ingestion"""
        documents = [
            Document(
                id=f"doc{i}",
                title=f"Document {i}",
                content=f"Content for document {i}",
                format="text",
                source=f"test{i}.txt",
                checksum=f"hash{i}",
                status="processed",
                metadata={"batch": "test"}
            )
            for i in range(3)
        ]
        
        results = rag_service.batch_ingest(documents)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert len(rag_service.documents) == 3

    def test_update_document(self, rag_service, sample_document):
        """Test document update functionality"""
        # First ingest the document
        rag_service.ingest_document(sample_document)
        
        # Update the document
        updated_doc = sample_document.copy()
        updated_doc.content = "# Updated Introduction\nThis is an updated test document."
        updated_doc.checksum = "def456"
        
        result = rag_service.update_document("doc1", updated_doc)
        
        assert result.success is True
        assert rag_service.documents["doc1"].content == updated_doc.content

    def test_delete_document(self, rag_service, sample_document):
        """Test document deletion"""
        # First ingest the document
        rag_service.ingest_document(sample_document)
        
        # Verify document exists
        assert "doc1" in rag_service.documents
        assert len(rag_service.chunks) > 0
        
        # Delete the document
        result = rag_service.delete_document("doc1")
        
        assert result.success is True
        assert "doc1" not in rag_service.documents
        assert len(rag_service.chunks) == 0

    def test_get_document_info(self, rag_service, sample_document):
        """Test document information retrieval"""
        # First ingest the document
        rag_service.ingest_document(sample_document)
        
        # Get document info
        info = rag_service.get_document_info("doc1")
        
        assert info is not None
        assert info.id == "doc1"
        assert info.title == "Test Document"

    def test_list_documents(self, rag_service, sample_document):
        """Test document listing"""
        # First ingest the document
        rag_service.ingest_document(sample_document)
        
        # List documents
        documents = rag_service.list_documents()
        
        assert len(documents) == 1
        assert documents[0].id == "doc1"

    def test_get_system_health(self, rag_service):
        """Test system health reporting"""
        health = rag_service.get_system_health()
        
        assert "status" in health
        assert "documents_count" in health
        assert "chunks_count" in health
        assert "vector_store_status" in health
        assert "embedding_service_status" in health

    def test_get_search_statistics(self, rag_service, sample_document):
        """Test search statistics"""
        # First ingest a document
        rag_service.ingest_document(sample_document)
        
        # Perform some searches
        rag_service.search("machine learning", top_k=5)
        rag_service.search("test", top_k=5)
        
        # Get statistics
        stats = rag_service.get_search_statistics()
        
        assert "total_searches" in stats
        assert "total_queries" in stats
        assert stats["total_searches"] >= 2

    def test_clear_system(self, rag_service, sample_document):
        """Test system clearing"""
        # First ingest a document
        rag_service.ingest_document(sample_document)
        
        # Verify data exists
        assert len(rag_service.documents) > 0
        assert len(rag_service.chunks) > 0
        
        # Clear the system
        result = rag_service.clear_system()
        
        assert result.success is True
        assert len(rag_service.documents) == 0
        assert len(rag_service.chunks) == 0

    def test_export_data(self, rag_service, sample_document):
        """Test data export functionality"""
        # First ingest a document
        rag_service.ingest_document(sample_document)
        
        # Export data
        export_data = rag_service.export_data()
        
        assert "documents" in export_data
        assert "chunks" in export_data
        assert len(export_data["documents"]) == 1
        assert len(export_data["chunks"]) > 0

    def test_empty_search(self, rag_service):
        """Test search behavior with no documents"""
        results = rag_service.search("test query", top_k=5)
        assert len(results) == 0

    def test_empty_query(self, rag_service):
        """Test query behavior with no documents"""
        response = rag_service.query("test question")
        assert response.answer is not None
        assert "no relevant documents" in response.answer.lower()

    def test_invalid_document_id(self, rag_service):
        """Test handling of invalid document IDs"""
        # Try to get info for non-existent document
        info = rag_service.get_document_info("nonexistent")
        assert info is None
        
        # Try to delete non-existent document
        result = rag_service.delete_document("nonexistent")
        assert result.success is False

    def test_document_processing_error(self, rag_service):
        """Test handling of document processing errors"""
        # Create a document with invalid content
        invalid_doc = Document(
            id="invalid",
            title="Invalid Document",
            content="",  # Empty content should cause error
            format="text",
            source="invalid.txt",
            checksum="hash",
            status="pending",
            metadata={}
        )
        
        result = rag_service.ingest_document(invalid_doc)
        assert result.success is False

    def test_search_with_filters(self, rag_service, sample_document):
        """Test search with metadata filters"""
        # First ingest the document
        rag_service.ingest_document(sample_document)
        
        # Search with filters
        results = rag_service.search(
            "machine learning", 
            top_k=5, 
            filters={"topic": "ml"}
        )
        
        assert len(results) > 0

    def test_chunk_metadata_preservation(self, rag_service, sample_document):
        """Test that chunk metadata is preserved during processing"""
        # First ingest the document
        rag_service.ingest_document(sample_document)
        
        # Check that chunks have proper metadata
        for chunk in rag_service.chunks.values():
            assert chunk.document_id == "doc1"
            assert chunk.metadata is not None
            assert "topic" in chunk.metadata

