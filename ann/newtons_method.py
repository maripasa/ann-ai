import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(mo):
    mo.md(
        r"""
    # Newton's Method

    Gradient Descent takes too many iterations.

    Newton's Method, for small datasets, is way better and converges in less iterations.

    Basically, Newton's Method gets the derivative on a point. We follow the derivative until the derivative crosses the 0 in the x axis. There, we calculate the derivative again. From there we do all again. It follows "Quadratic Convergence"
    """
    )
    return


@app.cell
def _(np):
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def logistic_regression(X, y, iters=10): # Could have some tolerance here too so it stops, it converges fast
        n, d = X.shape
        w = np.zeros(d)
        for _ in range(iters):
            y_pred = sigmoid(X @ w)

            # Partial (Gradient)
            grad = X.T @ (y_pred - y)

            # Hessian
            R = np.diag(y_pred * (1 - y_pred))
            hess = X.T @ R @ X

            # Newton
            delta = np.linalg.solve(hess, grad)
            w -= delta

            # It would check here, is delta < tolerance?
        return w
        
        
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
