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
    # Gradient Descent

    The goal of any model is to choose a **hypothesis** — among all possible hypotheses in the set **H** — that minimizes a **loss function** *L*.  
    If we plot every possible hypothesis against its corresponding loss over the dataset *X*, we obtain a continuous surface (the **loss landscape**).  

    Starting from any point on this surface, we can compute the **gradient** — the direction of the steepest increase in loss.  
    By moving a small step in the *opposite* direction of the gradient (scaled by a learning rate **λ**), we move closer to the minimum of *L*.

    **Gradient Descent** is an algorithm that automates this process:

    1. Choose an initial set of parameters **w**
    2. For each iteration **t** in [1..T]:
        a) Compute the partial derivatives of *L* with respect to **w**  
        b) Update **w** by moving a small step (**λ**) in the opposite direction of the gradient  
        c) Repeat until **T** iterations or convergence

    For example, look at this linear regression class:
    """
    )
    return


@app.cell
def _(np):
    class GradientLinearRegressor:
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
                y_pred = X @ self.weights
                grad = (1/n) * (X.T @ (y_pred - y))
                self.weights -= lr * grad

        def predict(self, X):
            X = self._augment_with_bias(X)
            return X @ self.weights

        def evaluate(self, X, y):
            y_pred = self.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            return mse

    """
    Will not implement tests for now. But analyzing the line `self.weights` generates updating over time, it should look like a line that slowly rotates and shifts to fit the data.
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Stochastic Gradient Descent (SGD)

    Gradient Descent is powerful, but it has a limitation: it computes the gradient using the entire dataset — all of X and y — at every step. This works fine for small datasets, but becomes very expensive for large ones.

    Stochastic Gradient Descent (SGD) solves this problem by using only a small sample of the data for each update.

    Instead of calculating the gradient with all data (using all samples in X and y), SGD randomly selects a subset (sometimes just one sample, sometimes a small batch) and computes the gradient from that sample.

    Because of this, the gradient estimate is noisy, but that noise can actually help the model escape local minima. It also makes the algorithm much faster and more scalable, since it doesn’t require processing the entire dataset every time.

    - Faster: uses small samples instead of the full dataset
    - Noisy: the randomness helps avoid local minima
    - Scalable: works well for large or streaming data
    """
    )
    return


app._unparsable_cell(
    r"""
    class SGDLinearRegressor:
        def __init__(self):
            self.weights = None

        def _augment_with_bias(self, X):
            X = X[:, np.newaxis] if X.ndim == 1 else X
            return np.hstack([np.ones((X.shape[0], 1)), X])

        def train(self, X, y, lr=0.001, epochs=1000, batch_size=1)
            X = self._augment_with_bias(X)
            n, d = X.shape
            self.weights = np.zeros(d)

            for _ in range(epochs):
                indices = np.random.permutation(n)
                X, y = X[indices], y[indices]

                for i in range(0, n, batch_size):
                    X_batch = X[i:i+batch_size]
                    y_batch = y[i:i+batch_size]

                    y_pred = X_batch @ self.weights
                    grad = (1/len(y_batch)) * (X_batch.T @ (y_pred - y_batch))
                    self.weights -= lr * grad

        def predict(self, X):
            X = self._augment_with_bias(X)
            return X @ self.weights
    \"\"\"
    Will also not implement tests for now. But analyzing the line `self.weights` generates updating over time, it should look like a line that slowly rotates and shifts to fit the data.
    \"\"\"
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also decrease the learning rate over time, so it converges more easily.""")
    return


if __name__ == "__main__":
    app.run()
