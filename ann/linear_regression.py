import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import train_test_split
    return mo, np, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Linear Regressor

    - **Type:** Multivariable Linear Regressor
    - **Input:** Matrix X, [x1, x2, x3, x4 ... ], xn e R^d; Vector y, [y1, y2, y3, y4 ...], yn e R, 
    - **Weights:** Vector `w = [w1, w2, ..., wn]` plus bias `b`
    - **Learning process:**
        - Gradient Descent
        - Normal Equation (We want w such that Aw = b)
    """
    )
    return


@app.cell
def _(np):
    class LinearRegressor:
        def __init__(self):
            self.A = None  # X^T X
            self.b = None  # X^T y
            self.w = None  # weights including bias

        def _augment_with_bias(self, X):
            X = X[:, np.newaxis] if X.ndim == 1 else X # Turn into matrix if it's 1d array.
            return np.hstack([np.ones((X.shape[0], 1)), X])

        def train(self, X, y):
            Xn = self._augment_with_bias(X)
            self.A = Xn.T @ Xn
            self.b = Xn.T @ y
            self.w = np.linalg.solve(self.A, self.b)

        def predict(self, X):
            Xn = self._augment_with_bias(X)
            return Xn @ self.w

        def evaluate(self, X, y):
            y_pred = self.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            return mse

        def retrain(self, X_new, y_new):
            Xn_new = self._augment_with_bias(X_new)
            self.A += Xn_new.T @ Xn_new
            self.b += Xn_new.T @ y_new
            self.w = np.linalg.solve(self.A, self.b)
    return (LinearRegressor,)


@app.cell
def _(LinearRegressor, np, plt, train_test_split):
    fig, axs = plt.subplots()

    np.random.seed(42)

    n = 100

    ep = np.random.normal(0, 1, n)
    x = np.random.uniform(-10, 10, n)
    y = 0.8 * x + 1 + ep

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    lg = LinearRegressor()
    lg.train(X_train, y_train)
    print(lg.evaluate(X_test, y_test))

    w_x = np.linspace(-10, 10, 100)
    w_y = lg.predict(w_x)

    axs.set_aspect('equal')
    axs.set_xlim(-10, 10)
    axs.set_ylim(-10, 10)

    axs.plot(w_x, w_y, "r-")
    axs.scatter(X_train, y_train, c="blue")
    axs.scatter(X_test, y_test, c="green")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
