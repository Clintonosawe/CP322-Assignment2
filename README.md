# ML Regression and Optimization

This project explores some of the fundamentals of supervised machine learning, with a focus on feature mapping, linear regression, logistic regression, and gradient descent.

The goal was to get a better understanding of how these models actually work, rather than relying entirely on built-in machine learning libraries. Through a combination of manual implementations and practical experiments, the project looks at how models learn from data, how their parameters are optimized, and how well they perform on new examples.

## Project Overview

The project is divided into three parts:

- **Feature Mapping:** Exploring how transforming input features can make a dataset linearly separable.
- **Logistic Regression:** Implementing a binary classifier from scratch to solve the XOR problem using batch gradient descent.
- **Linear Regression:** Training and evaluating a regression model while experimenting with learning rates and cross-validation.

## Project Structure

```text
ML-Regression-and-Optimization/
├── A2Q2_LOG_REG/
│   ├── log_reg.py
│   └── log_reg.ipynb
├── regression-dataset/
├── Part3_Linear_Regression.ipynb
└── README.md
```

## 1. Feature Mapping

The first part focuses on linear separability and how feature transformations can help with classification.

Some datasets cannot be separated using a single linear decision boundary in their original feature space. By introducing additional features, we can represent the data in a way that makes classification easier.

For this problem, the following feature map was used:

$$
\psi(x) = (x, x^2)
$$

This transforms a one-dimensional input into a two-dimensional feature vector, allowing a linear classifier to correctly separate the examples.

## 2. Logistic Regression

For the second part, I implemented logistic regression from scratch to classify the XOR dataset.

XOR is a useful example because its classes cannot be separated by a straight line in the original input space. To address this, I introduced an interaction feature:

$$
\psi(x_1, x_2) = (x_1, x_2, x_1x_2)
$$

The implementation includes:

- **Linear predictions:** Computing the weighted sum of the input features.
- **Sigmoid activation:** Converting the model's output into a probability between 0 and 1.
- **Binary cross-entropy loss:** Measuring how well the predicted probabilities match the actual labels.
- **Batch gradient descent:** Updating the model's weights using all training examples at each iteration.

Rather than using an existing logistic regression package, I implemented these calculations directly in Python to better understand the training process.

### Results

| Metric | Result |
|---|---|
| Initial weights | (-0.6, 2.0, 1.2, -2.8) |
| Learning rate | 0.1 |
| Training iterations | 25,356 |
| Final cross-entropy loss | 0.00999966 |
| Training accuracy | 100% |

After training, the model correctly classified all four XOR examples.

| Input | Actual Class | Predicted Class |
|---|---|---|
| (0, 0) | 0 | 0 |
| (0, 1) | 1 | 1 |
| (1, 0) | 1 | 1 |
| (1, 1) | 0 | 0 |

## 3. Linear Regression

The final part focuses on training and evaluating a linear regression model using a numerical dataset.

I used 10-fold cross-validation to compare different learning rates and determine which value produced the lowest average mean squared error (MSE).

The learning rates were tested from 0.00 to 0.10 in increments of 0.01. After selecting the best value, I evaluated the model on a separate test dataset to see how well it performed on unseen data.

### Results

| Metric | Result |
|---|---|
| Cross-validation folds | 10 |
| Best learning rate | 0.02 |
| Cross-validation MSE | 1.3438 |
| Test MSE | 1.4459 |

The test MSE was slightly higher than the cross-validation MSE, but the results were reasonably close. This suggests that the model performed similarly on the validation and test data, without a major drop in performance.

## Technologies Used

- **Python** – Model implementation and experimentation
- **NumPy** – Numerical calculations and array operations
- **scikit-learn** – Regression and model evaluation
- **Matplotlib** – Visualizing model performance
- **Jupyter Notebook** – Running experiments and documenting results

The logistic regression implementation uses Python's built-in `math` library without relying on prebuilt machine learning classifiers.
