import pytest
import tempfile
import os
from pathlib import Path

# Test configuration
pytest_plugins = ["pytest_asyncio"]

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(scope="session")
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "title": "Machine Learning Basics",
            "content": """# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled training data to learn the mapping between inputs and outputs.

### Unsupervised Learning
Unsupervised learning finds hidden patterns in data without labeled examples.

### Reinforcement Learning
Reinforcement learning learns through interaction with an environment to maximize rewards.
""",
            "format": "markdown",
            "topic": "machine_learning"
        },
        {
            "title": "Deep Learning Fundamentals",
            "content": """# Deep Learning Fundamentals

Deep learning is a subset of machine learning that uses neural networks with multiple layers.

## Neural Networks
Neural networks are composed of interconnected nodes that process information.

## Training Process
The training process involves:
1. Forward propagation
2. Loss calculation
3. Backward propagation
4. Weight updates

## Applications
Deep learning is used in:
- Computer vision
- Natural language processing
- Speech recognition
""",
            "format": "markdown",
            "topic": "deep_learning"
        },
        {
            "title": "Data Science Workflow",
            "content": """# Data Science Workflow

The data science workflow consists of several key steps:

## 1. Data Collection
Gathering relevant data from various sources.

## 2. Data Cleaning
Removing inconsistencies and handling missing values.

## 3. Data Exploration
Understanding data patterns and relationships.

## 4. Feature Engineering
Creating meaningful features for modeling.

## 5. Model Building
Training and evaluating machine learning models.

## 6. Deployment
Putting models into production use.
""",
            "format": "markdown",
            "topic": "data_science"
        }
    ]

@pytest.fixture(scope="session")
def test_queries():
    """Sample test queries for evaluation"""
    return [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the steps in data science?",
        "Explain supervised learning",
        "What is deep learning used for?",
        "How to clean data?",
        "What is feature engineering?",
        "Explain the training process"
    ]

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Set test environment variables
    os.environ["DEBUG"] = "true"
    os.environ["APP_NAME"] = "RAG Test"
    os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    os.environ["CHUNK_SIZE"] = "200"
    os.environ["CHUNK_OVERLAP"] = "50"
    
    yield
    
    # Cleanup after each test
    pass

