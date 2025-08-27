#!/usr/bin/env python3
"""
Demo script for the RAG system.
Shows how to use the system with sample documents and queries.
"""
import os
import sys
import time
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.rag_service import RAGService
from app.core.config import settings

def create_sample_documents():
    """Create sample documents for demonstration."""
    sample_docs = [
        {
            "title": "Python Programming Guide",
            "content": """
# Python Programming Guide

## Introduction
Python is a high-level, interpreted programming language known for its simplicity and readability.

## Basic Syntax
Python uses indentation to define code blocks. Here's a simple example:

```python
def greet(name):
    print(f"Hello, {name}!")

greet("World")
```

## Data Types
Python has several built-in data types:
- Strings: "Hello World"
- Numbers: 42, 3.14
- Lists: [1, 2, 3]
- Dictionaries: {"key": "value"}

## Best Practices
1. Use meaningful variable names
2. Follow PEP 8 style guidelines
3. Write docstrings for functions
4. Handle exceptions properly
            """,
            "filename": "python_guide.md"
        },
        {
            "title": "Machine Learning Basics",
            "content": """
# Machine Learning Fundamentals

## What is Machine Learning?
Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning
1. **Supervised Learning**: Learning from labeled data
2. **Unsupervised Learning**: Finding patterns in unlabeled data
3. **Reinforcement Learning**: Learning through interaction with environment

## Common Algorithms
- Linear Regression
- Decision Trees
- Random Forests
- Neural Networks
- Support Vector Machines

## Example: Linear Regression
```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[5]])
print(f"Prediction: {prediction[0]}")
```

## Evaluation Metrics
- Mean Squared Error (MSE)
- R-squared (R²)
- Accuracy (for classification)
- Precision and Recall
            """,
            "filename": "ml_basics.md"
        },
        {
            "title": "Web Development with FastAPI",
            "content": """
# FastAPI Web Development

## Introduction
FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.

## Key Features
- Fast: Very high performance
- Fast to code: Fewer bugs
- Intuitive: Great editor support
- Easy: Designed to be easy to use
- Short: Minimize code duplication

## Installation
```bash
pip install fastapi uvicorn
```

## Basic Example
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

## Running the Server
```bash
uvicorn main:app --reload
```

## API Documentation
FastAPI automatically generates interactive API documentation at:
- `/docs` - Swagger UI
- `/redoc` - ReDoc

## Data Validation
FastAPI uses Pydantic for data validation:
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None
```
            """,
            "filename": "fastapi_guide.md"
        }
    ]
    
    # Create documents directory if it doesn't exist
    docs_dir = Path(settings.upload_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Write sample documents
    for doc in sample_docs:
        file_path = docs_dir / doc["filename"]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["content"])
        print(f"✓ Created sample document: {doc['filename']}")
    
    return sample_docs

def run_demo():
    """Run the RAG system demo."""
    print("🚀 RAG System Demo")
    print("=" * 50)
    
    # Initialize RAG service
    print("\nInitializing RAG service...")
    rag_service = RAGService()
    
    # Create sample documents
    print("\nCreating sample documents...")
    sample_docs = create_sample_documents()
    
    # Ingest documents
    print("\nIngesting documents...")
    for doc in sample_docs:
        file_path = Path(settings.upload_dir) / doc["filename"]
        try:
            document = rag_service.ingest_document(
                str(file_path), 
                doc["title"]
            )
            print(f"✓ Ingested: {doc['title']} ({document.id})")
        except Exception as e:
            print(f"✗ Failed to ingest {doc['title']}: {e}")
    
    # Wait a moment for processing
    time.sleep(2)
    
    # Demo queries
    demo_queries = [
        "What is Python?",
        "How do I create a function in Python?",
        "What are the types of machine learning?",
        "How do I install FastAPI?",
        "What is the difference between supervised and unsupervised learning?",
        "How do I run a FastAPI server?",
        "What are Python best practices?",
        "How do I evaluate machine learning models?"
    ]
    
    print("\n" + "=" * 50)
    print("Running Demo Queries")
    print("=" * 50)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = rag_service.query(query, top_k=3)
            query_time = time.time() - start_time
            
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Response Time: {query_time:.2f}s")
            print(f"Sources: {len(response.sources)} chunks")
            
            # Show top source
            if response.sources:
                top_source = response.sources[0]
                print(f"Top Source: {top_source.chunk.content[:100]}...")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()
    
    # System statistics
    print("=" * 50)
    print("System Statistics")
    print("=" * 50)
    
    health = rag_service.get_system_health()
    stats = rag_service.get_search_statistics()
    
    print(f"Documents: {health.document_count}")
    print(f"Chunks: {health.chunk_count}")
    print(f"System Status: {health.status}")
    print(f"Uptime: {health.uptime:.1f}s")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("\nYou can now:")
    print("1. Open http://localhost:8000 for the web interface")
    print("2. Use the API endpoints at http://localhost:8000/docs")
    print("3. Try your own queries and documents")

if __name__ == "__main__":
    run_demo()
