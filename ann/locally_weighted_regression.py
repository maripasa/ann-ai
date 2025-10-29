import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Locally Weighted Regression
    - Parametric (Linear to the size of the dataset, since it needs the entire dataset for every prediction)
    - Bad at extrapolating
    - Can predict non linear functions
    """
    )
    return


@app.cell
def _(np):
    def w_func(x_input, x, tau=1):
        return np.exp(-(x - x_input)**2 / (2 * tau**2))

    def make_2d_and_bias(X):
        X = X[:, np.newaxis] if X.ndim == 1 else X
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def locally_weighted_regression(x_input, X, y, tau=1):
        X_ = make_2d_and_bias(X)
        x_input_ = make_2d_and_bias(np.array([x_input]))

        W = np.exp(- (X - x_input)**2 / (2 * tau**2))
        W = np.diag(W)

        theta = np.linalg.pinv(X_.T @ W @ X_) @ X_.T @ W @ y
        return (x_input_ @ theta)[0]
    return (locally_weighted_regression,)


@app.cell(hide_code=True)
def _(mo):
    slider = mo.ui.slider(0.1,10,0.1)
    mo.md(
        f"""
    tau:
    {slider}
    """
    )
    return (slider,)


@app.cell
def _(locally_weighted_regression, np, plt, slider):
    np.random.seed(42)

    x = np.random.uniform(-10, 10, 100)
    y = x ** 4 + 1000 * np.sin(x) + 2000 * np.cos(x)
    y += np.random.normal(0, 100, size=x.shape)

    x_test = np.sort(np.random.uniform(-10, 10, 40))

    tau_ = slider.value
    y_test = [locally_weighted_regression(xi, x, y, tau_) for xi in x_test]

    plt.scatter(x, y)
    plt.plot(x_test, y_test, "r-")
    plt.xlabel("x")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
