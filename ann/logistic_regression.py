import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Logistic Regression

    Linear regression predicts continous values, like price, temperature or weight.

    But some common problems require that y is categorical:
    - 0 or 1 (spam/not spam)
    - yes/no (disease present)
    - win/lose

    We may want a model that outputs a probability **P(y=1|x)**

    In *Logistic Regression* we use the sigmoid (logistic) function to squash the linear combination of features into [0,1]:

    y' = sig(wTx + b)

    sig(x) = 1 / (1 + e^(-x))

    Because of it's probabilistic nature, it isn't linear, and can't be solved with the normal formula, needing Gradient Descent.
    """
    )
    return


@app.cell
def _(np):
    def sig(z):
        return 1 / (1 + np.exp(-z))

    class LogisticRegressor:
        def __init__(self):
            self.weights = None

        def _augment_with_bias(self, X):
            X = X[:, np.newaxis] if X.ndim == 1 else X
            return np.hstack([np.ones((X.shape[0], 1)), X])

        def train(self, X, y, lr=0.001, epochs=1000):
            X = self._augment_with_bias(X)
            n, d = X.shape
            self.weights = np.zeros(d)
            for _ in range(epochs):
                grad = (1/n) * X.T @ ((sig(X @ self.weights)) - y)
                self.weights -= self.lr * grad

        def predict_proba(self, X):
            X = self._augment_with_bias(X)
            return sig(X @ self.weights)

        def predict (self, X):
            return 1 if self.predict_proba(X) >= 0.5 else 0

        def evaluate(self, X, y):
            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == y)
            return accuracy
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Softmax (Multiclass Logistic) Regression

    Softmax regression predicts probabilities for multiple classes:
    P(y=j | x) for j in [0..K-1]

    1. Linear scores
    Compute a score for each class:
      z_j = w_j^T x + b_j

    2. Softmax
    Convert scores into probabilities:
      P(y=j | x) = exp(z_j) / sum_k exp(z_k)
    - Probabilities sum to 1
    - Higher score → higher probability

    3. One-hot encoding
    Convert integer labels into vectors with 1 at the correct class:
      y = [0, 2, 1] → y_onehot =
      [[1,0,0],
       [0,0,1],
       [0,1,0]]

    4. Cross-entropy loss
      L = - (1/n) sum_i sum_j y_onehot[i,j] * log(P(y=j|x_i))

    5. Gradient descent
      scores = X @ weights
      probs = softmax(scores)
      grad = (1/n) * X.T @ (probs - y_onehot)
      weights -= lr * grad
    - Repeat for epochs

    6. Predictions
    - Probabilities: probs = softmax(X @ weights)
    - Predicted class: y_pred = argmax(probs, axis=1)

    Summary:
    - Generalizes logistic regression to multiple classes
    - Trains using gradient descent on cross-entropy
    - Outputs probability distribution; predict class with max probability

    """
    )
    return


@app.cell
def _(np):
    def softmax(z):
        z = np.array(z)
        if z.ndim == 1:
            z = z - np.max(z)
            exp_z = np.exp(z)
            return exp_z / np.sum(exp_z)
        else:
            z = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    class SoftmaxLogisticRegressor:
        def __init__(self):
            self.weights = None

        def _augment_with_bias(self, X):
            X = X[:, np.newaxis] if X.ndim == 1 else X
            return np.hstack([np.ones((X.shape[0], 1)), X])

        def train(self, X, y, lr=0.01, epochs=1000):
            X = self._augment_with_bias(X)
            n, d = X.shape

            classes = np.unique(y)
            k = len(classes)
            y_onehot = np.zeros((n, k))
            for i, c in enumerate(classes):
                y_onehot[:, i] = (y == c).astype(float)

            self.weights = np.zeros((d, k))

            for _ in range(epochs):
                scores = X @ self.weights
                probs = softmax(scores)
                grad = (1/n) * X.T @ (probs - y_onehot)
                self.weights -= lr * grad

        def predict_proba(self, X):
            X = self._augment_with_bias(X)
            scores = X @ self.weights
            return softmax(scores)

        def predict(self, X):
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)

        def evaluate(self, X, y):
            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == y)
            return accuracy
    return


if __name__ == "__main__":
    app.run()
