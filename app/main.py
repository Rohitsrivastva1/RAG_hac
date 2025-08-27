"""
Main FastAPI application for the RAG system.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from app.core.config import settings
from app.models.schemas import (
    SearchQuery, RAGResponse, DocumentUpload, DocumentUpdate, 
    SystemHealth, EvaluationMetrics
)
from pydantic import BaseModel
from app.services.rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A comprehensive RAG system for document ingestion, vector search, and grounded LLM responses",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService()

# Create data directories
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.chunk_dir, exist_ok=True)
os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting RAG system...")
    logger.info(f"Configuration: {settings.dict()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG system...")


# Health and system endpoints
@app.get("/health", response_model=SystemHealth)
async def health_check():
    """Get system health status."""
    return rag_service.get_system_health()


@app.get("/stats")
async def get_statistics():
    """Get system statistics."""
    return rag_service.get_search_statistics()


# Document management endpoints
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Upload and ingest a document."""
    try:
        # Parse optional fields
        tags_list = tags.split(',') if tags else []
        metadata_dict = {}
        if metadata:
            import json
            metadata_dict = json.loads(metadata)
        
        # Save uploaded file
        file_path = Path(settings.upload_dir) / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Ingest document
        document = rag_service.ingest_document(
            str(file_path), 
            title, 
            metadata_dict
        )
        
        return {
            "message": "Document uploaded and ingested successfully",
            "document_id": document.id,
            "chunks_created": len([c for c in rag_service.chunks.values() if c.document_id == document.id])
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all documents in the system."""
    return rag_service.list_documents()


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get information about a specific document."""
    doc_info = rag_service.get_document_info(document_id)
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc_info


@app.put("/documents/{document_id}")
async def update_document(
    document_id: str,
    file: UploadFile = File(...)
):
    """Update an existing document."""
    try:
        # Save uploaded file
        file_path = Path(settings.upload_dir) / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Update document
        document = rag_service.update_document(document_id, str(file_path))
        
        return {
            "message": "Document updated successfully",
            "document_id": document.id
        }
        
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document."""
    success = rag_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}


# Search and query endpoints
@app.post("/search", response_model=List[Dict[str, Any]])
async def search_chunks(query: SearchQuery):
    """Search for relevant chunks using vector similarity."""
    try:
        results = rag_service.search(
            query.query, 
            query.top_k, 
            query.filters
        )
        
        # Convert to serializable format
        search_results = []
        for result in results:
            search_results.append({
                "chunk_id": result.chunk.id,
                "content": result.chunk.content,
                "similarity_score": result.similarity_score,
                "rank": result.rank,
                "document_id": result.chunk.document_id,
                "chunk_type": result.chunk.chunk_type.value,
                "heading_path": result.chunk.heading_path,
                "metadata": result.metadata
            })
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=RAGResponse)
async def query_rag(query: SearchQuery, use_hybrid: bool = True):
    """Query the RAG system and get a generated response."""
    try:
        response = rag_service.query(
            query.query, 
            query.top_k, 
            query.filters, 
            use_hybrid
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error querying RAG system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SimpleQueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/query/simple")
async def simple_query(request: SimpleQueryRequest):
    """Simple query endpoint for basic usage."""
    try:
        response = rag_service.query(request.query, request.top_k)
        return response
        
    except Exception as e:
        logger.error(f"Error in simple query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch operations
@app.post("/documents/batch")
async def batch_upload_documents(documents: List[Dict[str, Any]]):
    """Upload multiple documents in batch."""
    try:
        results = rag_service.batch_ingest(documents)
        return {
            "message": f"Batch ingestion completed: {len(results)} documents processed",
            "documents": [{"id": doc.id, "title": doc.title} for doc in results]
        }
        
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System management endpoints
@app.post("/system/clear")
async def clear_system():
    """Clear all data from the system."""
    try:
        rag_service.clear_system()
        return {"message": "System cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system/export")
async def export_system_data(export_path: str = "./data/export/rag_export.json"):
    """Export system data."""
    try:
        rag_service.export_data(export_path)
        return {"message": f"System data exported to {export_path}"}
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# LLM Provider Management endpoints
@app.get("/llm/providers")
async def get_llm_providers():
    """Get available LLM providers and current configuration."""
    try:
        provider_info = rag_service.llm_service.get_provider_info()
        available_providers = rag_service.llm_service.get_available_providers()
        
        return {
            "current_provider": provider_info["current_provider"],
            "current_model": provider_info["current_model"],
            "available_providers": available_providers,
            "provider_configs": {
                "openai": {
                    "configured": provider_info["openai_configured"],
                    "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
                },
                "anthropic": {
                    "configured": provider_info["anthropic_configured"],
                    "models": ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
                },
                "ollama": {
                    "configured": provider_info["ollama_configured"],
                    "base_url": provider_info["ollama_base_url"],
                    "models": ["gpt-oss:20b", "llama2:latest", "mistral:latest", "codellama:latest"]
                },
                "gemini": {
                    "configured": provider_info["gemini_configured"],
                    "model": provider_info["gemini_model"],
                    "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-pro-vision"]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting LLM providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class LLMProviderUpdate(BaseModel):
    provider: str
    model: Optional[str] = None

@app.post("/llm/provider")
async def update_llm_provider(update: LLMProviderUpdate):
    """Update the LLM provider and model."""
    try:
        rag_service.llm_service.update_provider(update.provider, update.model)
        return {
            "message": f"LLM provider updated to {update.provider}",
            "provider": update.provider,
            "model": update.model or rag_service.llm_service.model
        }
        
    except Exception as e:
        logger.error(f"Error updating LLM provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/status")
async def get_llm_status():
    """Get current LLM service status."""
    try:
        provider_info = rag_service.llm_service.get_provider_info()
        return {
            "status": "active",
            "provider": provider_info["current_provider"],
            "model": provider_info["current_model"],
            "available_providers": rag_service.llm_service.get_available_providers()
        }
        
    except Exception as e:
        logger.error(f"Error getting LLM status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Simple web interface
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Get a modern dark glossy chatbot-style web interface for the RAG system."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Chatbot - Document Q&A</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
                height: 100vh;
                overflow: hidden;
                color: #ffffff;
            }
            
            .chat-container {
                max-width: 1200px;
                margin: 20px auto;
                background: rgba(15, 15, 15, 0.8);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 25px;
                box-shadow: 
                    0 25px 50px rgba(0, 0, 0, 0.5),
                    0 0 0 1px rgba(255, 255, 255, 0.05),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                display: flex;
                height: calc(100vh - 40px);
                overflow: hidden;
                position: relative;
            }
            
            .chat-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(135deg, rgba(138, 43, 226, 0.1) 0%, rgba(75, 0, 130, 0.1) 100%);
                border-radius: 25px;
                pointer-events: none;
            }
            
            .sidebar {
                width: 300px;
                background: rgba(20, 20, 35, 0.9);
                backdrop-filter: blur(15px);
                border-right: 1px solid rgba(255, 255, 255, 0.1);
                padding: 25px;
                display: flex;
                flex-direction: column;
                position: relative;
                z-index: 1;
            }
            
            .sidebar h2 {
                color: #ffffff;
                margin-bottom: 25px;
                font-size: 1.6rem;
                text-align: center;
                text-shadow: 0 0 20px rgba(138, 43, 226, 0.5);
                font-weight: 700;
            }
            
            .demo-toggle-section {
                background: rgba(30, 30, 45, 0.8);
                backdrop-filter: blur(10px);
                padding: 25px;
                border-radius: 20px;
                margin-bottom: 25px;
                border: 1px solid rgba(0, 212, 170, 0.3);
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
            }
            
            .demo-toggle-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(0, 212, 170, 0.1), transparent);
                transition: left 0.5s;
            }
            
            .demo-toggle-section:hover::before {
                left: 100%;
            }
            
            .demo-toggle-section h3 {
                color: #ffffff;
                margin-bottom: 20px;
                font-size: 1.2rem;
                font-weight: 600;
                text-shadow: 0 0 10px rgba(0, 212, 170, 0.3);
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .demo-status {
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 20px;
                padding: 15px;
                background: rgba(0, 212, 170, 0.1);
                border-radius: 12px;
                border: 1px solid rgba(0, 212, 170, 0.2);
            }
            
            .demo-status.active {
                background: rgba(0, 212, 170, 0.15);
                border-color: rgba(0, 212, 170, 0.4);
            }
            
            .demo-status.inactive {
                background: rgba(255, 107, 107, 0.1);
                border-color: rgba(255, 107, 107, 0.2);
            }
            
            .demo-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #00d4aa;
                box-shadow: 0 0 10px rgba(0, 212, 170, 0.5);
                animation: pulse 2s infinite;
            }
            
            .demo-indicator.inactive {
                background: #ff6b6b;
                box-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
                animation: none;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .demo-info {
                flex: 1;
                font-size: 14px;
                color: rgba(255, 255, 255, 0.9);
            }
            
            .demo-toggle-btn {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #00d4aa 0%, #0099cc 100%);
                color: white;
                border: none;
                border-radius: 15px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 25px rgba(0, 212, 170, 0.3);
            }
            
            .demo-toggle-btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 12px 35px rgba(0, 212, 170, 0.5);
            }
            
            .demo-toggle-btn:active {
                transform: translateY(-1px);
            }
            
            .upload-section {
                background: rgba(30, 30, 45, 0.8);
                backdrop-filter: blur(10px);
                padding: 25px;
                border-radius: 20px;
                margin-bottom: 25px;
                border: 1px solid rgba(138, 43, 226, 0.3);
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            
            .upload-section.disabled {
                opacity: 0.5;
                pointer-events: none;
                border-color: rgba(255, 255, 255, 0.1);
            }
            
            .upload-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(138, 43, 226, 0.1), transparent);
                transition: left 0.5s;
            }
            
            .upload-section:hover::before {
                left: 100%;
            }
            
            .upload-section h3 {
                color: #ffffff;
                margin-bottom: 20px;
                font-size: 1.2rem;
                font-weight: 600;
                text-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
            }
            
            .file-input-wrapper {
                position: relative;
                margin-bottom: 20px;
            }
            
            .file-input {
                width: 100%;
                padding: 15px;
                border: 2px dashed rgba(138, 43, 226, 0.4);
                border-radius: 15px;
                background: rgba(15, 15, 25, 0.6);
                cursor: pointer;
                transition: all 0.3s ease;
                color: #ffffff;
                font-size: 14px;
            }
            
            .file-input:hover {
                border-color: rgba(138, 43, 226, 0.8);
                background: rgba(20, 20, 35, 0.8);
                box-shadow: 0 0 20px rgba(138, 43, 226, 0.3);
            }
            
            .file-input:focus {
                outline: none;
                border-color: #8a2be2;
                box-shadow: 0 0 25px rgba(138, 43, 226, 0.4);
            }
            
            .title-input {
                width: 100%;
                padding: 15px;
                border: 1px solid rgba(138, 43, 226, 0.3);
                border-radius: 15px;
                margin-bottom: 20px;
                font-size: 14px;
                transition: all 0.3s ease;
                background: rgba(15, 15, 25, 0.6);
                color: #ffffff;
                backdrop-filter: blur(10px);
            }
            
            .title-input::placeholder {
                color: rgba(255, 255, 255, 0.6);
            }
            
            .title-input:focus {
                outline: none;
                border-color: #8a2be2;
                box-shadow: 0 0 25px rgba(138, 43, 226, 0.4);
                background: rgba(20, 20, 35, 0.8);
            }
            
            .upload-btn {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #8a2be2 0%, #4b0082 100%);
                color: white;
                border: none;
                border-radius: 15px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 25px rgba(138, 43, 226, 0.3);
            }
            
            .upload-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: left 0.5s;
            }
            
            .upload-btn:hover::before {
                left: 100%;
            }
            
            .upload-btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 12px 35px rgba(138, 43, 226, 0.5);
            }
            
            .upload-btn:active {
                transform: translateY(-1px);
            }
            
            .provider-section {
                background: rgba(30, 30, 45, 0.8);
                backdrop-filter: blur(10px);
                padding: 25px;
                border-radius: 20px;
                border: 1px solid rgba(138, 43, 226, 0.3);
                box-shadow: 
                    0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }
            
            .provider-section h3 {
                color: #ffffff;
                margin-bottom: 20px;
                font-size: 1.2rem;
                font-weight: 600;
                text-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
            }
            
            .provider-selector {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .provider-selector select {
                padding: 12px;
                border: 1px solid rgba(138, 43, 226, 0.3);
                border-radius: 12px;
                font-size: 14px;
                transition: all 0.3s ease;
                background: rgba(15, 15, 25, 0.6);
                color: #ffffff;
                backdrop-filter: blur(10px);
            }
            
            .provider-selector select:focus {
                outline: none;
                border-color: #8a2be2;
                box-shadow: 0 0 20px rgba(138, 43, 226, 0.3);
            }
            
            .provider-selector select option {
                background: #1a1a2e;
                color: #ffffff;
            }
            
            .update-provider-btn {
                padding: 12px;
                background: linear-gradient(135deg, #00d4aa 0%, #0099cc 100%);
                color: white;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 6px 20px rgba(0, 212, 170, 0.3);
            }
            
            .update-provider-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 212, 170, 0.5);
            }
            
            .chat-main {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: rgba(15, 15, 15, 0.6);
                backdrop-filter: blur(15px);
                position: relative;
                z-index: 1;
            }
            
            .chat-header {
                padding: 25px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                background: linear-gradient(135deg, rgba(138, 43, 226, 0.2) 0%, rgba(75, 0, 130, 0.2) 100%);
                backdrop-filter: blur(15px);
                position: relative;
                overflow: hidden;
            }
            
            .chat-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
                animation: shimmer 3s infinite;
            }
            
            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
            
            .chat-header h1 {
                font-size: 2rem;
                font-weight: 700;
                text-shadow: 0 0 30px rgba(138, 43, 226, 0.5);
                position: relative;
                z-index: 1;
            }
            
            .chat-header p {
                margin-top: 8px;
                opacity: 0.9;
                font-size: 15px;
                position: relative;
                z-index: 1;
            }
            
            .chat-messages {
                flex: 1;
                padding: 25px;
                overflow-y: auto;
                background: rgba(10, 10, 20, 0.4);
                backdrop-filter: blur(10px);
            }
            
            .message {
                margin-bottom: 25px;
                display: flex;
                align-items: flex-start;
                gap: 15px;
                animation: fadeInUp 0.4s ease;
                position: relative;
            }
            
            .message.user {
                flex-direction: row-reverse;
            }
            
            .message-avatar {
                width: 45px;
                height: 45px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                color: white;
                flex-shrink: 0;
                position: relative;
                overflow: hidden;
            }
            
            .message.user .message-avatar {
                background: linear-gradient(135deg, #8a2be2 0%, #4b0082 100%);
                box-shadow: 0 0 25px rgba(138, 43, 226, 0.5);
            }
            
            .message.bot .message-avatar {
                background: linear-gradient(135deg, #00d4aa 0%, #0099cc 100%);
                box-shadow: 0 0 25px rgba(0, 212, 170, 0.5);
            }
            
            .message-avatar::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                border-radius: 50%;
            }
            
            .message-content {
                max-width: 70%;
                padding: 20px 25px;
                border-radius: 25px;
                position: relative;
                word-wrap: break-word;
                backdrop-filter: blur(10px);
            }
            
            .message.user .message-content {
                background: linear-gradient(135deg, rgba(138, 43, 226, 0.9) 0%, rgba(75, 0, 130, 0.9) 100%);
                color: white;
                border-bottom-right-radius: 8px;
                box-shadow: 0 8px 25px rgba(138, 43, 226, 0.3);
            }
            
            .message.bot .message-content {
                background: rgba(30, 30, 45, 0.9);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-bottom-left-radius: 8px;
                box-shadow: 
                    0 8px 25px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
            }
            
            .message-time {
                font-size: 12px;
                opacity: 0.7;
                margin-top: 8px;
                font-weight: 500;
            }
            
            .message.user .message-time {
                text-align: right;
            }
            
            .sources-section {
                margin-top: 20px;
                padding: 20px;
                background: rgba(20, 20, 35, 0.8);
                border-radius: 15px;
                border: 1px solid rgba(138, 43, 226, 0.2);
                backdrop-filter: blur(10px);
            }
            
            .sources-section h4 {
                color: #ffffff;
                margin-bottom: 15px;
                font-size: 14px;
                font-weight: 600;
                text-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
            }
            
            .source-item {
                background: rgba(15, 15, 25, 0.6);
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 10px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                font-size: 13px;
                color: rgba(255, 255, 255, 0.8);
                backdrop-filter: blur(5px);
                transition: all 0.3s ease;
            }
            
            .source-item:hover {
                background: rgba(20, 20, 35, 0.8);
                border-color: rgba(138, 43, 226, 0.3);
                transform: translateY(-2px);
            }
            
            .chat-input-section {
                padding: 25px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                background: rgba(20, 20, 35, 0.8);
                backdrop-filter: blur(15px);
            }
            
            .chat-input-wrapper {
                display: flex;
                gap: 20px;
                align-items: flex-end;
            }
            
            .chat-input {
                flex: 1;
                padding: 18px 25px;
                border: 2px solid rgba(138, 43, 226, 0.3);
                border-radius: 30px;
                font-size: 16px;
                resize: none;
                min-height: 55px;
                max-height: 120px;
                transition: all 0.3s ease;
                font-family: inherit;
                background: rgba(15, 15, 25, 0.6);
                color: #ffffff;
                backdrop-filter: blur(10px);
            }
            
            .chat-input::placeholder {
                color: rgba(255, 255, 255, 0.6);
            }
            
            .chat-input:focus {
                outline: none;
                border-color: #8a2be2;
                box-shadow: 0 0 30px rgba(138, 43, 226, 0.4);
                background: rgba(20, 20, 35, 0.8);
            }
            
            .send-btn {
                width: 55px;
                height: 55px;
                border-radius: 50%;
                background: linear-gradient(135deg, #8a2be2 0%, #4b0082 100%);
                color: white;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                transition: all 0.3s ease;
                flex-shrink: 0;
                box-shadow: 0 8px 25px rgba(138, 43, 226, 0.3);
                position: relative;
                overflow: hidden;
            }
            
            .send-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                border-radius: 50%;
                transform: scale(0);
                transition: transform 0.3s ease;
            }
            
            .send-btn:hover::before {
                transform: scale(1);
            }
            
            .send-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 12px 35px rgba(138, 43, 226, 0.5);
            }
            
            .send-btn:active {
                transform: scale(0.95);
            }
            
            .typing-indicator {
                display: none;
                padding: 20px 25px;
                background: rgba(30, 30, 45, 0.9);
                border-radius: 25px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin-bottom: 25px;
                border-bottom-left-radius: 8px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }
            
            .typing-dots {
                display: flex;
                gap: 6px;
                justify-content: center;
            }
            
            .typing-dot {
                width: 10px;
                height: 10px;
                background: #8a2be2;
                border-radius: 50%;
                animation: typing 1.4s infinite ease-in-out;
                box-shadow: 0 0 10px rgba(138, 43, 226, 0.5);
            }
            
            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            
            .status-message {
                padding: 15px 20px;
                border-radius: 12px;
                margin: 15px 0;
                font-size: 14px;
                display: none;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                font-weight: 500;
            }
            
            .status-success {
                background: rgba(0, 212, 170, 0.2);
                color: #00d4aa;
                border-color: rgba(0, 212, 170, 0.3);
                box-shadow: 0 0 20px rgba(0, 212, 170, 0.2);
            }
            
            .status-error {
                background: rgba(255, 107, 107, 0.2);
                color: #ff6b6b;
                border-color: rgba(255, 107, 107, 0.3);
                box-shadow: 0 0 20px rgba(255, 107, 107, 0.2);
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes typing {
                0%, 80%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            .scrollbar::-webkit-scrollbar {
                width: 8px;
            }
            
            .scrollbar::-webkit-scrollbar-track {
                background: rgba(20, 20, 35, 0.5);
                border-radius: 4px;
            }
            
            .scrollbar::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #8a2be2, #4b0082);
                border-radius: 4px;
                box-shadow: 0 0 10px rgba(138, 43, 226, 0.3);
            }
            
            .scrollbar::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #9d4edd, #5a189a);
                box-shadow: 0 0 15px rgba(138, 43, 226, 0.5);
            }
            
            @media (max-width: 768px) {
                .chat-container {
                    margin: 10px;
                    height: calc(100vh - 20px);
                }
                
                .sidebar {
                    width: 100%;
                    height: auto;
                    border-right: none;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .chat-container {
                    flex-direction: column;
                }
                
                .message-content {
                    max-width: 85%;
                }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="sidebar">
                <h2><i class="fas fa-robot"></i> RAG Chatbot</h2>
                
                <div class="demo-toggle-section">
                    <h3><i class="fas fa-play-circle"></i> Demo Mode</h3>
                    <div class="demo-status active" id="demoStatus">
                        <div class="demo-indicator" id="demoIndicator"></div>
                        <div class="demo-info">
                            <strong>Demo Active</strong><br>
                            <small>582-LSS ST.pdf loaded with 82 chunks</small>
                        </div>
                    </div>
                    <button onclick="toggleDemoMode()" class="demo-toggle-btn" id="demoToggleBtn">
                        <i class="fas fa-toggle-on"></i> Turn Off Demo
                    </button>
                </div>
                
                <div class="upload-section" id="uploadSection">
                    <h3><i class="fas fa-upload"></i> Upload Document</h3>
                    <div class="file-input-wrapper">
                        <input type="file" id="documentFile" class="file-input" accept=".pdf,.docx,.txt,.md,.html">
                    </div>
                    <input type="text" id="documentTitle" class="title-input" placeholder="Document Title">
                    <button onclick="uploadDocument()" class="upload-btn">
                        <i class="fas fa-cloud-upload-alt"></i> Upload & Process
                    </button>
                </div>
                
                <div class="provider-section">
                    <h3><i class="fas fa-cog"></i> AI Provider</h3>
                    <div class="provider-selector">
                        <select id="providerSelect" onchange="updateProvider()">
                            <option value="ollama">Ollama (Local)</option>
                            <option value="gemini" selected>Gemini (Google)</option>
                            <option value="openai">OpenAI</option>
                            <option value="anthropic">Anthropic</option>
                        </select>
                        <select id="modelSelect">
                            <option value="gpt-oss:20b">gpt-oss:20b</option>
                            <option value="gemini-1.5-flash" selected>gemini-1.5-flash</option>
                            <option value="gemini-1.5-pro">gemini-1.5-pro</option>
                            <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                            <option value="claude-3-sonnet-20240229">claude-3-sonnet-20240229</option>
                        </select>
                        <button onclick="updateProvider()" class="update-provider-btn">
                            <i class="fas fa-sync-alt"></i> Update
                        </button>
                    </div>
                </div>
                
                <div id="statusMessage" class="status-message"></div>
            </div>
            
            <div class="chat-main">
                <div class="chat-header">
                    <h1><i class="fas fa-comments"></i> Document Q&A Chat</h1>
                    <p>Ask questions about your uploaded documents and get AI-powered answers</p>
                </div>
                
                <div class="chat-messages scrollbar" id="chatMessages">
                    <div class="message bot">
                        <div class="message-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="message-content">
                            <div>Hello! I'm your AI assistant powered by Gemini. I have a demo document (582-LSS ST.pdf) loaded with 82 chunks. You can ask me questions about it, or upload your own documents by turning off demo mode!</div>
                            <div class="message-time">Just now</div>
                        </div>
                    </div>
                </div>
                
                <div class="typing-indicator" id="typingIndicator">
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
                
                <div class="chat-input-section">
                    <div class="chat-input-wrapper">
                        <textarea 
                            id="chatInput" 
                            class="chat-input" 
                            placeholder="Ask me anything about the demo document..."
                            rows="1"
                            onkeydown="handleKeyDown(event)"
                            oninput="autoResize(this)"
                        ></textarea>
                        <button onclick="sendMessage()" class="send-btn" id="sendBtn">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let isProcessing = false;
            let isDemoMode = true;
            
            // Load current provider settings on page load
            window.onload = function() {
                loadProviderStatus();
                // Pre-load demo document
                loadDemoDocument();
                // Initialize model dropdown for Gemini (default provider)
                updateModelOptions('gemini');
            };
            
            function toggleDemoMode() {
                isDemoMode = !isDemoMode;
                const demoStatus = document.getElementById('demoStatus');
                const demoIndicator = document.getElementById('demoIndicator');
                const demoToggleBtn = document.getElementById('demoToggleBtn');
                const uploadSection = document.getElementById('uploadSection');
                const chatInput = document.getElementById('chatInput');
                
                if (isDemoMode) {
                    // Enable demo mode
                    demoStatus.className = 'demo-status active';
                    demoIndicator.className = 'demo-indicator';
                    demoToggleBtn.innerHTML = '<i class="fas fa-toggle-on"></i> Turn Off Demo';
                    uploadSection.classList.add('disabled');
                    chatInput.placeholder = 'Ask me anything about the demo document...';
                    
                    // Add demo mode message
                    addMessage('🔄 Demo mode activated! I have the 582-LSS ST.pdf document loaded with 82 chunks. Ask me anything about it!', false);
                    
                } else {
                    // Disable demo mode
                    demoStatus.className = 'demo-status inactive';
                    demoIndicator.className = 'demo-indicator inactive';
                    demoToggleBtn.innerHTML = '<i class="fas fa-toggle-off"></i> Turn On Demo';
                    uploadSection.classList.remove('disabled');
                    chatInput.placeholder = 'Ask me anything about your documents...';
                    
                    // Add demo mode message
                    addMessage('🔄 Demo mode deactivated! You can now upload your own documents. The demo document has been cleared.', false);
                }
            }
            
            async function loadDemoDocument() {
                try {
                    // Simulate loading the demo document
                    addMessage('📚 Loading demo document: 582-LSS ST.pdf...', false);
                    
                    // Add success message after a short delay
                    setTimeout(() => {
                        addMessage('✅ Demo document loaded successfully! I have 82 chunks from "582-LSS ST.pdf" ready for questions. Try asking me about the content!', false);
                    }, 1000);
                    
                } catch (error) {
                    addMessage('❌ Error loading demo document. Please try refreshing the page.', false);
                }
            }
            
            function autoResize(textarea) {
                textarea.style.height = 'auto';
                textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
            }
            
            function handleKeyDown(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            }
            
            function addMessage(content, isUser = false, sources = null, timestamp = null) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const time = timestamp || new Date().toLocaleTimeString();
                
                let sourcesHtml = '';
                if (sources && sources.length > 0) {
                    sourcesHtml = `
                        <div class="sources-section">
                            <h4><i class="fas fa-link"></i> Sources:</h4>
                            ${sources.map(s => `
                                <div class="source-item">
                                    ${s.chunk.content.substring(0, 150)}...
                                </div>
                            `).join('')}
                        </div>
                    `;
                }
                
                messageDiv.innerHTML = `
                    <div class="message-avatar">
                        <i class="fas fa-${isUser ? 'user' : 'robot'}"></i>
                    </div>
                    <div class="message-content">
                        <div>${content}</div>
                        ${sourcesHtml}
                        <div class="message-time">${time}</div>
                    </div>
                `;
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function showTypingIndicator() {
                document.getElementById('typingIndicator').style.display = 'block';
                document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
            }
            
            function hideTypingIndicator() {
                document.getElementById('typingIndicator').style.display = 'none';
            }
            
            function showStatus(message, type = 'success') {
                const statusDiv = document.getElementById('statusMessage');
                statusDiv.textContent = message;
                statusDiv.className = `status-message status-${type}`;
                statusDiv.style.display = 'block';
                
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 5000);
            }
            
            async function sendMessage() {
                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                
                if (!message || isProcessing) return;
                
                isProcessing = true;
                const sendBtn = document.getElementById('sendBtn');
                sendBtn.disabled = true;
                
                // Add user message
                addMessage(message, true);
                input.value = '';
                input.style.height = 'auto';
                
                // Show typing indicator
                showTypingIndicator();
                
                try {
                    const response = await fetch('/query/simple', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: message, top_k: 5})
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        
                        // Hide typing indicator
                        hideTypingIndicator();
                        
                        // Add bot response
                        addMessage(result.answer, false, result.sources);
                        
                    } else {
                        hideTypingIndicator();
                        const errorText = await response.text();
                        addMessage(`Sorry, I encountered an error: ${errorText}`, false);
                    }
                } catch (error) {
                    hideTypingIndicator();
                    addMessage(`Sorry, I encountered an error: ${error.message}`, false);
                } finally {
                    isProcessing = false;
                    sendBtn.disabled = false;
                }
            }
            
            async function uploadDocument() {
                if (isDemoMode) {
                    showStatus('Please turn off demo mode first to upload documents', 'error');
                    return;
                }
                
                const fileInput = document.getElementById('documentFile');
                const titleInput = document.getElementById('documentTitle');
                
                if (!fileInput.files[0] || !titleInput.value.trim()) {
                    showStatus('Please select a file and enter a title', 'error');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('title', titleInput.value.trim());
                
                try {
                    const response = await fetch('/documents/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        showStatus(`Document uploaded successfully! Created ${result.chunks_created} chunks.`, 'success');
                        
                        // Clear inputs
                        fileInput.value = '';
                        titleInput.value = '';
                        
                        // Add success message to chat
                        addMessage(`✅ Document "${result.document_id}" uploaded successfully! I can now answer questions about it.`, false);
                        
                    } else {
                        const errorText = await response.text();
                        showStatus(`Upload failed: ${errorText}`, 'error');
                    }
                } catch (error) {
                    showStatus(`Error: ${error.message}`, 'error');
                }
            }
            
            async function loadProviderStatus() {
                try {
                    const response = await fetch('/llm/status');
                    if (response.ok) {
                        const status = await response.json();
                        document.getElementById('providerSelect').value = status.provider;
                        document.getElementById('modelSelect').value = status.model;
                    }
                } catch (error) {
                    console.error('Error loading provider status:', error);
                }
            }
            
            async function updateProvider() {
                const provider = document.getElementById('providerSelect').value;
                const model = document.getElementById('modelSelect').value;
                
                try {
                    const response = await fetch('/llm/provider', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({provider: provider, model: model})
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        showStatus(`Provider updated to ${result.provider} with model ${result.model}`, 'success');
                        
                        // Add message to chat
                        addMessage(`🔄 AI provider updated to ${result.provider} (${result.model})`, false);
                        
                    } else {
                        const errorText = await response.text();
                        showStatus(`Failed to update provider: ${errorText}`, 'error');
                    }
                } catch (error) {
                    showStatus(`Error updating provider: ${error.message}`, 'error');
                }
            }
            
            // Update model options based on selected provider
            function updateModelOptions(provider) {
                const modelSelect = document.getElementById('modelSelect');
                
                // Clear existing options
                modelSelect.innerHTML = '';
                
                // Add provider-specific models
                if (provider === 'ollama') {
                    ['gpt-oss:20b', 'llama2:latest', 'mistral:latest', 'codellama:latest'].forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        modelSelect.appendChild(option);
                    });
                } else if (provider === 'gemini') {
                    ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-pro-vision'].forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        modelSelect.appendChild(option);
                    });
                    // Set default Gemini model
                    modelSelect.value = 'gemini-1.5-flash';
                } else if (provider === 'openai') {
                    ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'].forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        modelSelect.appendChild(option);
                    });
                } else if (provider === 'anthropic') {
                    ['claude-3-sonnet-20240229', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'].forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        modelSelect.appendChild(option);
                    });
                }
            }
            
            document.getElementById('providerSelect').addEventListener('change', function() {
                const provider = this.value;
                updateModelOptions(provider);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


