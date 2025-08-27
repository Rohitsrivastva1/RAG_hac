import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    @pytest.fixture
    def service(self):
        return EmbeddingService()
    
    @pytest.fixture
    def sample_texts(self):
        return [
            "This is the first sample text for testing.",
            "This is the second sample text for testing.",
            "This is the third sample text for testing."
        ]
    
    @pytest.fixture
    def sample_embedding(self):
        return np.random.rand(384).astype(np.float32)

    def test_initialization(self, service):
        """Test service initialization"""
        assert service.model is not None
        assert service.dimension > 0
        assert isinstance(service.dimension, int)

    def test_generate_single_embedding(self, service):
        """Test single embedding generation"""
        text = "Test text for embedding"
        embedding = service.generate_single_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.shape == (service.dimension,)
        assert not np.isnan(embedding).any()

    def test_generate_embeddings_single(self, service):
        """Test embedding generation for single text"""
        text = "Single text test"
        embeddings = service.generate_embeddings(text)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], np.ndarray)

    def test_generate_embeddings_multiple(self, service, sample_texts):
        """Test embedding generation for multiple texts"""
        embeddings = service.generate_embeddings(sample_texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_texts)
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (service.dimension,)

    def test_batch_generate_embeddings(self, service, sample_texts):
        """Test batch embedding generation"""
        embeddings = service.batch_generate_embeddings(sample_texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_texts)
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)

    def test_normalize_vector(self, service, sample_embedding):
        """Test vector normalization"""
        normalized = service._normalize_vector(sample_embedding)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float32
        # Check L2 norm is approximately 1
        assert np.isclose(np.linalg.norm(normalized), 1.0, atol=1e-6)

    def test_compute_similarity(self, service):
        """Test similarity computation between two embeddings"""
        # Create two similar embeddings
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.8, 0.2, 0.0], dtype=np.float32)
        
        similarity = service.compute_similarity(emb1, emb2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be reasonably similar

    def test_compute_batch_similarity(self, service):
        """Test batch similarity computation"""
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidate_embs = [
            np.array([0.8, 0.2, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
        ]
        
        similarities = service.compute_batch_similarity(query_emb, candidate_embs)
        
        assert isinstance(similarities, list)
        assert len(similarities) == len(candidate_embs)
        for sim in similarities:
            assert isinstance(sim, float)
            assert 0.0 <= sim <= 1.0

    def test_similarity_self(self, service):
        """Test that embedding is most similar to itself"""
        text = "Test text for self-similarity"
        embedding = service.generate_single_embedding(text)
        
        similarity = service.compute_similarity(embedding, embedding)
        assert np.isclose(similarity, 1.0, atol=1e-6)

    def test_embedding_consistency(self, service):
        """Test that same text produces same embedding"""
        text = "Consistency test text"
        emb1 = service.generate_single_embedding(text)
        emb2 = service.generate_single_embedding(text)
        
        # Embeddings should be identical
        np.testing.assert_array_equal(emb1, emb2)

    def test_empty_text_handling(self, service):
        """Test handling of empty text"""
        with pytest.raises(ValueError):
            service.generate_single_embedding("")

    def test_none_text_handling(self, service):
        """Test handling of None text"""
        with pytest.raises(ValueError):
            service.generate_single_embedding(None)

    def test_get_model_info(self, service):
        """Test model information retrieval"""
        info = service.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "dimension" in info
        assert "max_length" in info

    def test_update_model(self, service):
        """Test model updating"""
        original_model_name = service.model_name
        
        # Test with valid model
        service.update_model("all-MiniLM-L6-v2")
        assert service.model_name == "all-MiniLM-L6-v2"
        
        # Test with invalid model (should fallback to default)
        service.update_model("invalid-model-name")
        assert service.model_name == "all-MiniLM-L6-v2"

    def test_embedding_dimensions_consistent(self, service):
        """Test that all embeddings have consistent dimensions"""
        texts = ["Short", "Medium length text", "Very long text with many words"]
        embeddings = service.generate_embeddings(texts)
        
        expected_dim = service.dimension
        for emb in embeddings:
            assert emb.shape == (expected_dim,)

    def test_embedding_normalization(self, service):
        """Test that embeddings are properly normalized"""
        text = "Test text for normalization"
        embedding = service.generate_single_embedding(text)
        
        # Check L2 norm is approximately 1
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_similarity_bounds(self, service):
        """Test that similarity scores are within valid bounds"""
        texts = ["First text", "Second text", "Third text"]
        embeddings = service.generate_embeddings(texts)
        
        # Test all pairwise similarities
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                similarity = service.compute_similarity(embeddings[i], embeddings[j])
                assert 0.0 <= similarity <= 1.0

    def test_batch_processing_efficiency(self, service, sample_texts):
        """Test that batch processing is more efficient than individual"""
        import time
        
        # Individual processing
        start_time = time.time()
        individual_embs = []
        for text in sample_texts:
            emb = service.generate_single_embedding(text)
            individual_embs.append(emb)
        individual_time = time.time() - start_time
        
        # Batch processing
        start_time = time.time()
        batch_embs = service.batch_generate_embeddings(sample_texts)
        batch_time = time.time() - start_time
        
        # Batch should be faster (or at least not slower)
        assert batch_time <= individual_time * 1.5  # Allow some tolerance
        
        # Results should be identical
        for ind_emb, batch_emb in zip(individual_embs, batch_embs):
            np.testing.assert_array_equal(ind_emb, batch_emb)

