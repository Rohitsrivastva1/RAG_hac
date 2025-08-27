pythimport pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from app.services.vector_store import FAISSVectorStore
from app.models.schemas import Chunk, Embedding


class TestFAISSVectorStore:
    @pytest.fixture
    def vector_store(self):
        return FAISSVectorStore()
    
    @pytest.fixture
    def sample_chunks(self):
        return [
            Chunk(
                id="chunk1",
                text="This is the first chunk about machine learning.",
                chunk_type="semantic",
                document_id="doc1",
                chunk_index=0,
                metadata={"section": "intro"}
            ),
            Chunk(
                id="chunk2", 
                text="This is the second chunk about neural networks.",
                chunk_type="semantic",
                document_id="doc1",
                chunk_index=1,
                metadata={"section": "methods"}
            ),
            Chunk(
                id="chunk3",
                text="This is the third chunk about deep learning applications.",
                chunk_type="semantic", 
                document_id="doc1",
                chunk_index=2,
                metadata={"section": "applications"}
            )
        ]
    
    @pytest.fixture
    def sample_embeddings(self):
        # Create orthogonal embeddings for testing
        embeddings = []
        for i in range(3):
            emb = np.zeros(384, dtype=np.float32)
            emb[i] = 1.0
            embeddings.append(emb)
        return embeddings

    def test_initialization(self, vector_store):
        """Test vector store initialization"""
        assert vector_store.index is not None
        assert vector_store.dimension == 384
        assert vector_store.chunk_map == {}
        assert vector_store.embedding_map == {}

    def test_add_chunks(self, vector_store, sample_chunks, sample_embeddings):
        """Test adding chunks and embeddings to the store"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        assert len(vector_store.chunk_map) == 3
        assert len(vector_store.embedding_map) == 3
        
        # Check that chunks are stored correctly
        for chunk in sample_chunks:
            assert chunk.id in vector_store.chunk_map
            assert chunk.id in vector_store.embedding_map

    def test_search_basic(self, vector_store, sample_chunks, sample_embeddings):
        """Test basic vector search functionality"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        # Search with first embedding
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=2)
        
        assert len(results) == 2
        assert results[0].chunk_id == "chunk1"  # Should be most similar to itself
        assert results[0].similarity > results[1].similarity

    def test_search_top_k(self, vector_store, sample_chunks, sample_embeddings):
        """Test search with different top_k values"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        
        # Test with top_k=1
        results = vector_store.search(query_embedding, top_k=1)
        assert len(results) == 1
        
        # Test with top_k=3
        results = vector_store.search(query_embedding, top_k=3)
        assert len(results) == 3
        
        # Test with top_k larger than available chunks
        results = vector_store.search(query_embedding, top_k=10)
        assert len(results) == 3

    def test_search_similarity_threshold(self, vector_store, sample_chunks, sample_embeddings):
        """Test search with similarity threshold filtering"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        
        # High threshold should return fewer results
        results_high = vector_store.search(query_embedding, top_k=3, similarity_threshold=0.9)
        results_low = vector_store.search(query_embedding, top_k=3, similarity_threshold=0.1)
        
        assert len(results_high) <= len(results_low)

    def test_hybrid_search(self, vector_store, sample_chunks, sample_embeddings):
        """Test hybrid search combining vector and text similarity"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        query = "machine learning"
        query_embedding = sample_embeddings[0]
        
        results = vector_store.hybrid_search(query, query_embedding, top_k=3)
        
        assert len(results) > 0
        # Should find chunks containing "machine learning"
        assert any("machine learning" in result.chunk.text.lower() for result in results)

    def test_text_search(self, vector_store, sample_chunks, sample_embeddings):
        """Test text-only search functionality"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        query = "neural networks"
        results = vector_store._text_search(query, top_k=2)
        
        assert len(results) > 0
        # Should find chunk containing "neural networks"
        assert any("neural networks" in result.chunk.text.lower() for result in results)

    def test_combine_results(self, vector_store):
        """Test result combination logic"""
        vector_results = [
            Mock(chunk_id="chunk1", similarity=0.9),
            Mock(chunk_id="chunk2", similarity=0.7)
        ]
        text_results = [
            Mock(chunk_id="chunk2", similarity=0.8),
            Mock(chunk_id="chunk3", similarity=0.6)
        ]
        
        combined = vector_store._combine_results(vector_results, text_results, top_k=3)
        
        assert len(combined) > 0
        # chunk2 should appear in results due to both vector and text similarity

    def test_apply_filters(self, vector_store, sample_chunks, sample_embeddings):
        """Test metadata filtering"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        # Filter by section
        results = vector_store.search(
            sample_embeddings[0], 
            top_k=3, 
            filters={"section": "intro"}
        )
        
        assert len(results) > 0
        # All results should have section="intro"
        for result in results:
            chunk = vector_store.chunk_map[result.chunk_id]
            assert chunk.metadata.get("section") == "intro"

    def test_l2_to_similarity_conversion(self, vector_store):
        """Test L2 distance to similarity score conversion"""
        # Test with zero distance (should be maximum similarity)
        similarity = vector_store._l2_to_similarity(0.0)
        assert np.isclose(similarity, 1.0, atol=1e-6)
        
        # Test with large distance (should be low similarity)
        similarity = vector_store._l2_to_similarity(10.0)
        assert similarity < 0.1
        
        # Test that similarity decreases with distance
        sim1 = vector_store._l2_to_similarity(1.0)
        sim2 = vector_store._l2_to_similarity(2.0)
        assert sim1 > sim2

    def test_save_and_load_index(self, vector_store, sample_chunks, sample_embeddings):
        """Test index persistence"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save index
            vector_store._save_index(temp_path)
            assert os.path.exists(temp_path)
            
            # Create new store and load index
            new_store = FAISSVectorStore()
            new_store._load_index(temp_path)
            
            # Check that data is preserved
            assert len(new_store.chunk_map) == len(vector_store.chunk_map)
            assert len(new_store.embedding_map) == len(vector_store.embedding_map)
            
        finally:
            os.unlink(temp_path)

    def test_get_stats(self, vector_store, sample_chunks, sample_embeddings):
        """Test statistics retrieval"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        stats = vector_store.get_stats()
        
        assert "total_chunks" in stats
        assert "total_embeddings" in stats
        assert "index_size" in stats
        assert stats["total_chunks"] == 3
        assert stats["total_embeddings"] == 3

    def test_clear(self, vector_store, sample_chunks, sample_embeddings):
        """Test clearing the store"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        # Verify data is present
        assert len(vector_store.chunk_map) > 0
        assert len(vector_store.embedding_map) > 0
        
        # Clear the store
        vector_store.clear()
        
        # Verify data is removed
        assert len(vector_store.chunk_map) == 0
        assert len(vector_store.embedding_map) == 0

    def test_delete_chunk(self, vector_store, sample_chunks, sample_embeddings):
        """Test deleting individual chunks"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        initial_count = len(vector_store.chunk_map)
        
        # Delete a chunk
        vector_store.delete_chunk("chunk1")
        
        # Verify chunk is removed
        assert "chunk1" not in vector_store.chunk_map
        assert "chunk1" not in vector_store.embedding_map
        assert len(vector_store.chunk_map) == initial_count - 1

    def test_empty_search(self, vector_store):
        """Test search behavior with empty store"""
        query_embedding = np.random.rand(384).astype(np.float32)
        
        results = vector_store.search(query_embedding, top_k=5)
        assert len(results) == 0

    def test_invalid_embedding_dimension(self, vector_store):
        """Test handling of embeddings with wrong dimension"""
        wrong_dim_embedding = np.random.rand(100).astype(np.float32)
        
        with pytest.raises(ValueError):
            vector_store.search(wrong_dim_embedding, top_k=5)

    def test_search_ordering(self, vector_store, sample_chunks, sample_embeddings):
        """Test that search results are properly ordered by similarity"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=3)
        
        # Results should be ordered by similarity (descending)
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

    def test_metadata_preservation(self, vector_store, sample_chunks, sample_embeddings):
        """Test that chunk metadata is preserved during operations"""
        vector_store.add_chunks(sample_chunks, sample_embeddings)
        
        # Search and verify metadata is intact
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=3)
        
        for result in results:
            chunk = vector_store.chunk_map[result.chunk_id]
            assert chunk.metadata is not None
            assert "section" in chunk.metadata

