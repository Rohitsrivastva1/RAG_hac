"""
Pydantic schemas for the RAG system.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    DOCX = "docx"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkType(str, Enum):
    """Types of content chunks."""
    TEXT = "text"
    CODE = "code"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"


class Document(BaseModel):
    """Document model."""
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    source_url: Optional[HttpUrl] = Field(None, description="Source URL if applicable")
    file_path: str = Field(..., description="Local file path")
    format: DocumentFormat = Field(..., description="Document format")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    checksum: str = Field(..., description="File content checksum")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Chunk(BaseModel):
    """Text chunk model."""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    chunk_type: ChunkType = Field(default=ChunkType.TEXT)
    heading_path: Optional[List[str]] = Field(None, description="Hierarchical heading path")
    start_offset: int = Field(..., description="Start position in document")
    end_offset: int = Field(..., description="End position in document")
    token_count: int = Field(..., description="Number of tokens")
    checksum: str = Field(..., description="Content checksum")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")


class Embedding(BaseModel):
    """Embedding model."""
    chunk_id: str = Field(..., description="Associated chunk ID")
    vector: List[float] = Field(..., description="Embedding vector")
    model_name: str = Field(..., description="Embedding model used")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")
    include_metadata: bool = Field(default=True, description="Include chunk metadata")


class SearchResult(BaseModel):
    """Search result model."""
    chunk: Chunk
    similarity_score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Result rank")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """RAG system response."""
    answer: str = Field(..., description="Generated answer")
    sources: List[SearchResult] = Field(..., description="Source chunks used")
    confidence_score: float = Field(..., description="Overall confidence")
    processing_time: float = Field(..., description="Response time in seconds")
    query: str = Field(..., description="Original query")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentUpload(BaseModel):
    """Document upload request."""
    title: str = Field(..., description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUpdate(BaseModel):
    """Document update request."""
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SystemHealth(BaseModel):
    """System health status."""
    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")
    document_count: int = Field(..., description="Total documents")
    chunk_count: int = Field(..., description="Total chunks")
    vector_store_status: str = Field(..., description="Vector store status")
    database_status: str = Field(..., description="Database status")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for the RAG system."""
    retrieval_precision: float = Field(..., description="Retrieval precision@k")
    retrieval_recall: float = Field(..., description="Retrieval recall@k")
    generation_factuality: float = Field(..., description="Factuality score")
    response_relevance: float = Field(..., description="Response relevance score")
    user_satisfaction: float = Field(..., description="User satisfaction score")
    hallucination_rate: float = Field(..., description="Hallucination rate")
    average_response_time: float = Field(..., description="Average response time")
    total_queries: int = Field(..., description="Total queries processed")
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)
