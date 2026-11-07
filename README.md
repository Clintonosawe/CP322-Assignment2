ML Regression and Optimization

A Python-based machine learning project exploring linear regression, logistic regression, feature engineering, and gradient descent optimization. The project focuses on understanding how supervised learning models are built, trained, and evaluated through mathematical implementations and practical experiments.

Project Overview

This project covers three core areas of supervised machine learning:

Feature Mapping: Transforming input data into a higher-dimensional feature space to make previously non-linearly separable datasets linearly separable.

Logistic Regression: Building a binary classifier from scratch using sigmoid activation, cross-entropy loss, and batch gradient descent.

Linear Regression: Training and evaluating a regression model using hyperparameter tuning and 10-fold cross-validation.

The goal is to better understand the mathematics behind these algorithms and how different modelling and optimization decisions affect their performance.

Project Structure

ML-Regression-and-Optimization/
├── A2Q2_LOG_REG/
│   ├── log_reg.py
│   └── log_reg.ipynb
├── regression-dataset/
├── Part3_Linear_Regression.ipynb
└── README.md

1. Feature Mapping

The first part explores linear separability in binary classification.

A one-dimensional dataset may not be linearly separable because a single decision boundary cannot correctly distinguish the classes.

To address this, a nonlinear feature mapping is applied:

[
\psi(x)=(x,x^2)
]

This transforms the original input into a two-dimensional representation, allowing a linear classifier to separate the training examples.

Key concepts: Linear separability, decision boundaries, feature transformations, and linear classification.

2. Logistic Regression — XOR Classification

A logistic regression classifier is implemented from scratch to solve the XOR classification problem.

The XOR dataset is not linearly separable in its original two-dimensional representation. To overcome this, an interaction feature is introduced:

[
\psi(x_1,x_2)=(x_1,x_2,x_1x_2)
]

The model uses the following components:

Linear model: Calculates the weighted sum of input features.

Sigmoid activation: Converts the linear output into a probability between 0 and 1.

Binary cross-entropy: Measures the difference between predicted probabilities and actual target values.

Batch gradient descent: Updates model weights using the average gradient across all training examples.

The logistic regression algorithm is implemented manually without using a prebuilt machine learning classifier.

Training Results

Metric

Result

Initial weights

(-0.6, 2.0, 1.2, -2.8)

Learning rate

0.1

Training iterations

25,356

Final cross-entropy loss

0.00999966

Training accuracy

100%

The trained model correctly classifies all four XOR examples.

Input

Actual Class

Predicted Class

(0, 0)

0

0

(0, 1)

1

1

(1, 0)

1

1

(1, 1)

0

0

3. Linear Regression

The final part focuses on training and evaluating a linear regression model using a numerical dataset sampled from a two-dimensional surface.

The model is evaluated using 10-fold cross-validation to determine an appropriate learning rate. Different learning rates are tested, and their average mean squared errors are compared.

Model Evaluation Results

Metric

Result

Cross-validation folds

10

Learning rate search

0.00 to 0.10

Best learning rate

0.02

Cross-validation MSE

1.3438

Test MSE

1.4459

Cross-validation helps select a learning rate based on performance across different subsets of the training data, rather than relying on a single split.

The final model is then evaluated on a separate test dataset to measure how well it generalizes to unseen examples.

Technologies Used

Python — Model implementations and numerical computation

NumPy — Array operations and numerical calculations

scikit-learn — Linear regression modelling and evaluation

Matplotlib — Visualization of model performance

Jupyter Notebook — Experimentation, analysis, and presentation of results

The logistic regression component uses Python's built-in math library and does not rely on scikit-learn's logistic regression implementation.

