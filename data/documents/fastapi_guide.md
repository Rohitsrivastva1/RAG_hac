
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
            