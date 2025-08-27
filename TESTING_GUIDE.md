# RAG System Testing Guide

This guide covers all the different ways to test the RAG system we've built.

## 🚀 Quick Start Testing

### 1. **Fix the Pydantic Issue First**
The system had a Pydantic v2 compatibility issue. We've fixed it by:
- Updating `app/core/config.py` to use `pydantic-settings`
- Updating `requirements.txt` to include `pydantic-settings>=2.0.0`

### 2. **Install Dependencies**
```bash
# Install the updated requirements
pip install -r requirements.txt

# Or use the quick start script
python quick_start.py
```

## 🧪 Testing Methods

### **Method 1: Manual Testing with Demo Script**
```bash
python scripts/demo.py
```
This will:
- Create sample documents
- Ingest them into the system
- Run test queries
- Show results and confidence scores

### **Method 2: API Testing with Web Interface**
```bash
# Start the FastAPI server
python main.py

# Open browser to: http://localhost:8000
# Or API docs: http://localhost:8000/docs
```

### **Method 3: Unit Testing with pytest**
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_document_processor.py
pytest tests/test_embedding_service.py
pytest tests/test_vector_store.py
pytest tests/test_rag_service.py

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test methods
pytest tests/test_rag_service.py::TestRAGService::test_ingest_document
```

### **Method 4: Integration Testing**
```bash
# Run integration tests only
pytest -m integration

# Run tests excluding slow ones
pytest -m "not slow"
```

## 📋 Test Coverage

Our test suite covers:

### **DocumentProcessor Tests**
- ✅ Format detection (PDF, HTML, Markdown, DOCX, Text, Image)
- ✅ Text extraction from various formats
- ✅ Semantic and sliding window chunking
- ✅ Checksum calculation and ID generation
- ✅ Error handling for invalid inputs

### **EmbeddingService Tests**
- ✅ Single and batch embedding generation
- ✅ Vector normalization (L2 norm)
- ✅ Similarity computation
- ✅ Model information and updates
- ✅ Consistency and performance testing

### **VectorStore Tests**
- ✅ FAISS index operations
- ✅ Vector search with different parameters
- ✅ Hybrid search (vector + text)
- ✅ Metadata filtering
- ✅ Index persistence (save/load)
- ✅ Statistics and cleanup

### **RAGService Tests**
- ✅ End-to-end document ingestion
- ✅ Search functionality
- ✅ Query generation with LLM
- ✅ Batch operations
- ✅ Document management (CRUD)
- ✅ System health and statistics

## 🔧 Test Configuration

### **Environment Variables for Testing**
```bash
DEBUG=true
APP_NAME="RAG Test"
EMBEDDING_MODEL="all-MiniLM-L6-v2"
CHUNK_SIZE=200
CHUNK_OVERLAP=50
```

### **Pytest Configuration**
- **Coverage**: HTML, XML, and terminal reports
- **Markers**: slow, integration, unit, api
- **Output**: Verbose with color coding
- **Failures**: Max 5 failures before stopping

## 🚨 Common Issues and Solutions

### **Issue 1: Pydantic Import Error**
```
pydantic.errors.PydanticImportError: `BaseSettings` has been moved to the `pydantic-settings` package
```
**Solution**: Install `pydantic-settings` and update imports

### **Issue 2: Missing Dependencies**
```
ModuleNotFoundError: No module named 'sentence_transformers'
```
**Solution**: Run `pip install -r requirements.txt`

### **Issue 3: FAISS Installation Issues**
```
ERROR: Could not find a version that satisfies the requirement faiss-cpu
```
**Solution**: Use `pip install faiss-cpu --no-cache-dir` or install from conda

### **Issue 4: Memory Issues with Large Documents**
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce chunk size or use smaller documents for testing

## 📊 Performance Testing

### **Benchmark Tests**
```bash
# Run performance benchmarks
pytest tests/ -k "benchmark" --benchmark-only

# Test with different document sizes
pytest tests/ -k "performance" -v
```

### **Load Testing**
```bash
# Test with multiple concurrent requests
python -m pytest tests/test_load.py -v

# Test batch processing performance
python -m pytest tests/test_batch_performance.py -v
```

## 🧹 Test Data Management

### **Sample Documents**
The test suite includes sample documents covering:
- Machine Learning Basics
- Deep Learning Fundamentals  
- Data Science Workflow

### **Test Queries**
Pre-defined test queries for evaluation:
- "What is machine learning?"
- "How do neural networks work?"
- "What are the steps in data science?"

## 🔍 Debugging Tests

### **Verbose Output**
```bash
pytest -v -s --tb=long
```

### **Debug Specific Tests**
```bash
# Run with debugger
pytest --pdb

# Run with print statements visible
pytest -s
```

### **Coverage Analysis**
```bash
# Generate detailed coverage report
pytest --cov=app --cov-report=html

# Open coverage report
open htmlcov/index.html
```

## 📈 Continuous Integration

### **GitHub Actions Example**
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest --cov=app
```

## 🎯 Testing Best Practices

1. **Isolation**: Each test should be independent
2. **Fixtures**: Use pytest fixtures for common setup
3. **Mocking**: Mock external dependencies (LLM APIs, file I/O)
4. **Edge Cases**: Test error conditions and boundary values
5. **Performance**: Include performance benchmarks for critical paths
6. **Coverage**: Aim for >90% code coverage

## 🚀 Next Steps

1. **Run the demo**: `python scripts/demo.py`
2. **Start the API**: `python main.py`
3. **Run unit tests**: `pytest`
4. **Check coverage**: `pytest --cov=app --cov-report=html`
5. **Explore the web interface**: http://localhost:8000

## 📞 Getting Help

If you encounter issues:
1. Check the error messages carefully
2. Verify all dependencies are installed
3. Check the test logs with `pytest -v`
4. Review the coverage report for untested code
5. Check the README.md for setup instructions

Happy testing! 🎉

