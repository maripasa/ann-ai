import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Gaussian Discriminative Analysis

    - "GDA, in some cases, is more efficient and simpler than logistic regression"
    - We calculate p(y|x) based on Bayes Rule
    - As a generative model, GDA learns what "y=0" X's look like, and what "y=1" X's look like, almost in isolation.
    - We learn p(x|y), and p(y)
    - Basically it fits a gaussian to each of the label groupings
    - The covariance is the same for both mu_0 and mu_1 just because of convention, but you can have two covariance matrices.
    """
    )
    return


@app.cell
def _(np):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    class GDAmodel:
        def __init__(self):
            self.phi = None
            self.mu = None
            self.sigma = None
        
        # I haven't derived this, but these are the partial derivatives accordingly to the expressions for p(x|y = 0) ... etc
        def train(self, X, y):
            m = len(y)

            self.phi = np.mean(y)

            mu0 = X[y == 0].mean(axis=0)
            mu1 = X[y == 1].mean(axis=0)
            self.mu = np.vstack([mu0, mu1])

            diff = X - self.mu[y.astype(int)]
            self.sigma = (diff.T @ diff) / m

        def predict_proba(self, X):
            inv_sigma = np.linalg.inv(self.sigma)
            mu0, mu1 = self.mu
            phi = self.phi

            w = inv_sigma @ (mu1 - mu0)
            b = (
                -0.5 * (mu1 @ inv_sigma @ mu1 - mu0 @ inv_sigma @ mu0)
                + np.log(phi / (1 - phi))
            )

            z = X @ w + b
            return sigmoid(z)

        def predict(self, X, threshold=0.5):
            return (self.predict_proba(X) >= threshold).astype(int)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # GDA or a linear model?

    ## Generative (GDA)
    x|y = 0 ~ N(mu_0, sigma)
    x|y = 1 ~ N(mu_1, sigma)
    y ~ Ber(phi)

    ## Logistic Regression (discriminative)
    p(y = 1| x) = sigmoid(-w^T x)

    GDA implies p(y=1|x) is sigmoid.
    sigmoid does not imply GDA assumptions.

    GDA makes stronger assumptions.

    In a lot of linear algorithms, stronger assumptions (if roughly correct) makes your model do better (more information).
    The problem of GDA are those strong assumptions! For example, if the data isn't using a gaussian distribution, it will behave poorly.

    Logistic Regression does not care about the distribution type (Is it gaussian? Poisson?)

    If you have a lot of data, Logistic Regression does not fall behind (It has a lot of data, it does not need the assumptions!), so logistic regression is a great default.

    GDA has more performance anyway (no iterations)
    """
    )
    return


if __name__ == "__main__":
    app.run()
