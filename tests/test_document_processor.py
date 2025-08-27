import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.document_processor import DocumentProcessor
from app.models.schemas import DocumentFormat, ChunkType


class TestDocumentProcessor:
    @pytest.fixture
    def processor(self):
        return DocumentProcessor()
    
    @pytest.fixture
    def sample_text(self):
        return """# Introduction
This is a sample document for testing.

## Section 1
This is the first section with some content.

## Section 2
This is the second section with more content.

### Subsection 2.1
This is a subsection with detailed information.

## Conclusion
This concludes our test document."""

    def test_detect_format_pdf(self, processor):
        """Test PDF format detection"""
        assert processor._detect_format("test.pdf") == DocumentFormat.PDF
        assert processor._detect_format("test.PDF") == DocumentFormat.PDF

    def test_detect_format_html(self, processor):
        """Test HTML format detection"""
        assert processor._detect_format("test.html") == DocumentFormat.HTML
        assert processor._detect_format("test.htm") == DocumentFormat.HTML

    def test_detect_format_markdown(self, processor):
        """Test Markdown format detection"""
        assert processor._detect_format("test.md") == DocumentFormat.MARKDOWN
        assert processor._detect_format("test.mdown") == DocumentFormat.MARKDOWN

    def test_detect_format_docx(self, processor):
        """Test DOCX format detection"""
        assert processor._detect_format("test.docx") == DocumentFormat.DOCX

    def test_detect_format_text(self, processor):
        """Test plain text format detection"""
        assert processor._detect_format("test.txt") == DocumentFormat.TEXT
        assert processor._detect_format("test.log") == DocumentFormat.TEXT

    def test_detect_format_image(self, processor):
        """Test image format detection"""
        assert processor._detect_format("test.jpg") == DocumentFormat.IMAGE
        assert processor._detect_format("test.png") == DocumentFormat.IMAGE

    def test_calculate_checksum(self, processor, sample_text):
        """Test checksum calculation"""
        checksum = processor._calculate_checksum(sample_text)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hash length
        
        # Same text should produce same checksum
        checksum2 = processor._calculate_checksum(sample_text)
        assert checksum == checksum2

    def test_generate_id(self, processor):
        """Test ID generation"""
        id1 = processor._generate_id()
        id2 = processor._generate_id()
        
        assert isinstance(id1, str)
        assert len(id1) > 0
        assert id1 != id2  # IDs should be unique

    def test_semantic_chunking(self, processor, sample_text):
        """Test semantic chunking strategy"""
        chunks = processor._semantic_chunking(sample_text, chunk_size=100)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0
            # Check that chunks contain section headers
            assert any(keyword in chunk for keyword in ["#", "##", "###"])

    def test_sliding_window_chunking(self, processor, sample_text):
        """Test sliding window chunking strategy"""
        chunks = processor._sliding_window_chunking(sample_text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_create_chunk(self, processor):
        """Test chunk creation with metadata"""
        chunk = processor._create_chunk(
            text="Test chunk content",
            chunk_type=ChunkType.SEMANTIC,
            document_id="doc123",
            chunk_index=0,
            metadata={"section": "test"}
        )
        
        assert chunk.text == "Test chunk content"
        assert chunk.chunk_type == ChunkType.SEMANTIC
        assert chunk.document_id == "doc123"
        assert chunk.chunk_index == 0
        assert chunk.metadata["section"] == "test"

    @patch('app.services.document_processor.PyPDF2.PdfReader')
    def test_extract_pdf_text(self, mock_pdf_reader, processor):
        """Test PDF text extraction"""
        mock_reader = Mock()
        mock_reader.pages = [Mock(extract_text=lambda: "Page 1 content")]
        mock_pdf_reader.return_value = mock_reader
        
        text = processor._extract_pdf_text("dummy_path.pdf")
        assert text == "Page 1 content"

    def test_extract_plain_text(self, processor):
        """Test plain text extraction"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test plain text content")
            temp_path = f.name
        
        try:
            text = processor._extract_plain_text(temp_path)
            assert text == "Test plain text content"
        finally:
            os.unlink(temp_path)

    def test_extract_markdown_text(self, processor):
        """Test Markdown text extraction"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Markdown\n\nThis is **bold** text.")
            temp_path = f.name
        
        try:
            text = processor._extract_markdown_text(temp_path)
            assert "# Test Markdown" in text
            assert "This is **bold** text." in text
        finally:
            os.unlink(temp_path)

    def test_chunk_text_semantic(self, processor, sample_text):
        """Test semantic text chunking"""
        chunks = processor.chunk_text(sample_text, strategy="semantic", chunk_size=100)
        
        assert len(chunks) > 0
        # Check that semantic chunks preserve structure
        assert any("# Introduction" in chunk.text for chunk in chunks)
        assert any("## Section 1" in chunk.text for chunk in chunks)

    def test_chunk_text_sliding_window(self, processor, sample_text):
        """Test sliding window text chunking"""
        chunks = processor.chunk_text(sample_text, strategy="sliding_window", chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].text
            next_chunk = chunks[i + 1].text
            # There should be some overlap
            assert len(set(current_chunk.split()) & set(next_chunk.split())) > 0

    def test_invalid_chunking_strategy(self, processor, sample_text):
        """Test invalid chunking strategy raises error"""
        with pytest.raises(ValueError, match="Invalid chunking strategy"):
            processor.chunk_text(sample_text, strategy="invalid_strategy")

    def test_empty_text_chunking(self, processor):
        """Test chunking empty text"""
        chunks = processor.chunk_text("", strategy="semantic")
        assert len(chunks) == 0

    def test_small_text_chunking(self, processor):
        """Test chunking text smaller than chunk size"""
        small_text = "This is a small text."
        chunks = processor.chunk_text(small_text, strategy="semantic", chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0].text == small_text

