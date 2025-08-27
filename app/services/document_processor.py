"""
Document processing service for the RAG system.
Handles multiple file formats and intelligent chunking strategies.
"""
import os
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# Document processing imports
import PyPDF2
from bs4 import BeautifulSoup
import markdown
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

# Text processing
# import nltk  # Not needed for simple word splitting
# from nltk.tokenize import word_tokenize

from app.core.config import settings
from app.models.schemas import Document, Chunk, DocumentFormat, ChunkType

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing service."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        # NLTK data download removed - using simple text splitting instead
    
    def process_document(self, file_path: str, title: str, metadata: Dict[str, Any] = None) -> Document:
        """Process a document and return the document model."""
        file_path = Path(file_path)
        
        print(f"Processing document: {file_path}")
        print(f"File exists: {file_path.exists()}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Determine document format
        doc_format = self._detect_format(file_path)
        print(f"Detected format: {doc_format}")
        
        # Calculate checksum
        checksum = self._calculate_checksum(file_path)
        print(f"Calculated checksum: {checksum[:10]}...")
        
        # Create document model
        document = Document(
            id=self._generate_id(),
            title=title,
            file_path=str(file_path),
            format=doc_format,
            checksum=checksum,
            metadata=metadata or {}
        )
        
        print(f"Created document with ID: {document.id}")
        return document
    
    def extract_text(self, document: Document) -> str:
        """Extract text content from document based on format."""
        file_path = Path(document.file_path)
        print(f"Extracting text from: {file_path}")
        print(f"Document format: {document.format}")
        
        try:
            if document.format == DocumentFormat.PDF:
                text = self._extract_pdf_text(file_path)
            elif document.format == DocumentFormat.HTML:
                text = self._extract_html_text(file_path)
            elif document.format == DocumentFormat.MARKDOWN:
                text = self._extract_markdown_text(file_path)
            elif document.format == DocumentFormat.DOCX:
                text = self._extract_docx_text(file_path)
            elif document.format == DocumentFormat.TEXT:
                text = self._extract_plain_text(file_path)
            elif document.format == DocumentFormat.IMAGE:
                text = self._extract_image_text(file_path)
            else:
                raise ValueError(f"Unsupported document format: {document.format}")
            
            print(f"Extracted text length: {len(text)}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {document.file_path}: {e}")
            print(f"Error extracting text: {e}")
            raise
    
    def chunk_text(self, text: str, document_id: str) -> List[Chunk]:
        """Split text into intelligent chunks."""
        chunks = []
        
        # First, try semantic chunking by headings
        semantic_chunks = self._semantic_chunking(text)
        
        for i, (content, heading_path, start_pos) in enumerate(semantic_chunks):
            # If chunk is too long, split it further
            if len(content) > self.chunk_size:
                sub_chunks = self._sliding_window_chunking(content, start_pos)
                chunks.extend(sub_chunks)
            else:
                chunk = self._create_chunk(
                    content, document_id, i, heading_path, start_pos, len(content)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _detect_format(self, file_path: Path) -> DocumentFormat:
        """Detect document format from file extension."""
        suffix = file_path.suffix.lower()
        
        format_mapping = {
            '.pdf': DocumentFormat.PDF,
            '.html': DocumentFormat.HTML,
            '.htm': DocumentFormat.HTML,
            '.md': DocumentFormat.MARKDOWN,
            '.markdown': DocumentFormat.MARKDOWN,
            '.txt': DocumentFormat.TEXT,
            '.docx': DocumentFormat.DOCX,
            '.png': DocumentFormat.IMAGE,
            '.jpg': DocumentFormat.IMAGE,
            '.jpeg': DocumentFormat.IMAGE,
            '.py': DocumentFormat.CODE,
            '.java': DocumentFormat.CODE,
            '.js': DocumentFormat.CODE,
            '.ts': DocumentFormat.CODE,
            '.cpp': DocumentFormat.CODE,
            '.c': DocumentFormat.CODE,
        }
        
        return format_mapping.get(suffix, DocumentFormat.TEXT)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_id(self) -> str:
        """Generate unique document ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.warning(f"PDF text extraction failed, trying OCR: {e}")
            if settings.enable_ocr:
                text = self._extract_pdf_ocr(file_path)
        
        return text.strip()
    
    def _extract_pdf_ocr(self, file_path: Path) -> str:
        """Extract text from PDF using OCR."""
        # This is a simplified OCR implementation
        # In production, you might want to use more sophisticated OCR
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img) + "\n"
            return text
        except ImportError:
            logger.warning("PyMuPDF not available, OCR extraction failed")
            return ""
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
    
    def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from Markdown file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # For semantic chunking, we want to preserve heading structure
            # So we'll use the raw markdown content instead of converting to HTML
            return content
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def _extract_plain_text(self, file_path: Path) -> str:
        """Extract text from plain text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _extract_image_text(self, file_path: Path) -> str:
        """Extract text from image using OCR."""
        if not settings.enable_ocr:
            return ""
        
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"OCR extraction failed for {file_path}: {e}")
            return ""
    
    def _semantic_chunking(self, text: str) -> List[Tuple[str, List[str], int]]:
        """Split text by semantic boundaries (headings, sections)."""
        chunks = []
        
        # Split by common heading patterns
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headings (# ## ### etc.)
            r'^[A-Z][A-Z\s]+\n[-=]+\n',  # Underlined headings
            # Removed numbered sections pattern as it was too broad
        ]
        
        lines = text.split('\n')
        current_chunk = []
        current_heading_path = []
        start_pos = 0
        
        for i, line in enumerate(lines):
            is_heading = any(re.match(pattern, line) for pattern in heading_patterns)
            
            if is_heading and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append((chunk_text, current_heading_path.copy(), start_pos))
                
                # Start new chunk
                current_chunk = [line]
                current_heading_path = [line.strip()]
                start_pos = i
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append((chunk_text, current_heading_path, start_pos))
        return chunks
    
    def _sliding_window_chunking(self, text: str, start_offset: int) -> List[Chunk]:
        """Split text using sliding window approach."""
        chunks = []
        words = text.split()  # Simple word splitting instead of NLTK
        
        if len(words) <= self.chunk_size:
            return []
        
        for i in range(0, len(words) - self.chunk_size + 1, self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk = self._create_chunk(
                chunk_text, 
                "",  # document_id will be set later
                len(chunks), 
                [], 
                start_offset + i, 
                len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, document_id: str, chunk_index: int, 
                      heading_path: List[str], start_offset: int, end_offset: int) -> Chunk:
        """Create a chunk object."""
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        checksum = hashlib.md5(content.encode()).hexdigest()
        
        # Determine chunk type
        chunk_type = ChunkType.TEXT
        if any(keyword in content.lower() for keyword in ['def ', 'class ', 'function ', 'import ']):
            chunk_type = ChunkType.CODE
        elif heading_path:
            chunk_type = ChunkType.HEADING
        
        return Chunk(
            id=chunk_id,
            document_id=document_id,
            content=content,
            chunk_type=chunk_type,
            heading_path=heading_path,
            start_offset=start_offset,
            end_offset=end_offset,
            token_count=len(content.split()),
            checksum=checksum
        )
