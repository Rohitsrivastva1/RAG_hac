
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
            