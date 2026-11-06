
"""
CP322 - Assignment 2
Question 2: Logistic Regression for XOR

Implement logistic regression from scratch using
batch gradient descent and cross-entropy loss.
"""

import math

# --------------------------------------------------
# 1. XOR Dataset and Feature Map
# --------------------------------------------------

# Feature map: psi(x) = (x1, x2, x1*x2)
# Include x0 = 1 as the bias feature.

X = [
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1]
]

# Target values for XOR
t = [0, 1, 1, 0]

# Initial weights specified in assignment
weights = [-0.6, 2.0, 1.2, -2.8]

# Learning rate (same as lecture example)
learning_rate = 0.1

# Training parameters
max_iterations = 100000
loss_threshold = 0.01


# --------------------------------------------------
# 2. Sigmoid Activation Function
# --------------------------------------------------

def sigmoid(z):
    """
    Sigmoid activation:
    y = 1 / (1 + e^(-z))
    """
    return 1 / (1 + math.exp(-z))


# --------------------------------------------------
# 3. Linear Model and Predictions
# --------------------------------------------------

def predict(X, weights):
    """
    Calculate:
    z = w0*x0 + w1*x1 + w2*x2 + w3*x3
    y = sigmoid(z)
    """

    z_values = []
    y_values = []

    for x in X:
        z = sum(
            weights[j] * x[j]
            for j in range(len(weights))
        )

        y = sigmoid(z)

        z_values.append(z)
        y_values.append(y)

    return z_values, y_values


# --------------------------------------------------
# 4. Binary Cross-Entropy Loss
# --------------------------------------------------

def cross_entropy(y_values, targets):
    """
    Cross-entropy:
    L = -t*log(y) - (1-t)*log(1-y)

    Returns individual losses and average loss.
    """

    losses = []

    for y, target in zip(y_values, targets):

        # Avoid log(0) due to floating-point rounding
        epsilon = 1e-15
        y = max(epsilon, min(1 - epsilon, y))

        loss = -(
            target * math.log(y)
            + (1 - target) * math.log(1 - y)
        )

        losses.append(loss)

    average_loss = sum(losses) / len(losses)

    return losses, average_loss


# --------------------------------------------------
# 5. Batch Gradient Descent
# --------------------------------------------------

def calculate_gradients(X, t, y_values):
    """
    Gradient:
    dL/dwj = (1/N) * sum((y - t) * xj)

    Batch gradient descent uses all training
    examples before updating the weights.
    """

    N = len(X)
    gradients = [0.0] * len(X[0])

    for i in range(N):

        error = y_values[i] - t[i]

        for j in range(len(gradients)):
            gradients[j] += error * X[i][j]

    for j in range(len(gradients)):
        gradients[j] /= N

    return gradients


# --------------------------------------------------
# 6. Train Logistic Regression Model
# --------------------------------------------------
 
 
# --------------------------------------------------
# 7. Display Last 10 Updates
# --------------------------------------------------



# --------------------------------------------------
# 8. Evaluate Final Model
# --------------------------------------------------



# --------------------------------------------------
# 9. Main Program
# --------------------------------------------------

def main():

    print("CP322 Assignment 2 - Question 2")
    print("Logistic Regression for XOR")
    print("=" * 65)

    print(f"Initial Weights: {weights}")
    print(f"Learning Rate: {learning_rate}")

    final_weights, history = train(
        X, t, weights, learning_rate
    )

    # Show last 10 updates
    display_last_updates(history)

    # Show final weights
    print("\nFINAL WEIGHTS")
    print("=" * 65)

    for j, weight in enumerate(final_weights):
        print(f"w{j} = {weight:.6f}")

    print(f"\nTotal Iterations: {len(history)}")

    print(
        f"Final Average Cross-Entropy Loss: "
        f"{history[-1]['average_loss']:.8f}"
    )

    # Evaluate predictions
    evaluate(X, t, final_weights)


if __name__ == "__main__":
    main()
