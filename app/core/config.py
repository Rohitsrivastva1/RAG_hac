"""
Configuration settings for the RAG system.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Basic app settings
    app_name: str = Field(default="RAG System", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database settings
    database_url: str = Field(default="sqlite:///./rag_system.db", env="DATABASE_URL")
    
    # Vector store settings
    vector_store_type: str = Field(default="faiss", env="VECTOR_STORE_TYPE")
    faiss_index_path: str = Field(default="./data/embeddings/faiss_index", env="FAISS_INDEX_PATH")
    
    # Embedding settings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # Chunking settings
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    
    # Search settings
    top_k: int = Field(default=5, env="TOP_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # LLM Provider settings
    llm_provider: str = Field(default="gemini", env="LLM_PROVIDER")  # ollama, gemini, openai, anthropic
    llm_model: str = Field(default="gemini-1.5-flash", env="LLM_MODEL")
    
    # OpenAI settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Anthropic settings
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    
    # Gemini settings
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-change-this-in-production", env="SECRET_KEY")
    
    # File storage settings
    upload_dir: str = Field(default="./data/documents", env="UPLOAD_DIR")
    chunk_dir: str = Field(default="./data/chunks", env="CHUNK_DIR")
    
    # OCR settings
    enable_ocr: bool = Field(default=True, env="ENABLE_OCR")
    tesseract_path: str = Field(default="", env="TESSERACT_PATH")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=8001, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
