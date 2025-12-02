import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Naive Bayes

    Let's suppose we need to process some text data, and fit it in some class: (spam / no spam, tech / no tech)

    We could do something, we will have a dictionary (10k most used words, for example), and each X is the size of the dict, and each feature is if the word was used or not.

    For example, with the dict (car, buy, rob), the email "I won't buy it. I prefer to rob." (does not make sense, i know)

    | Dictionary    | X (example) |
    | -------- | ------- |
    | car  | 0    |
    | buy | 1     |
    | rob    | 1    |

    X = [0,1,1]

    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - X in {0,1}^n (n is the size of the dict)
    - Xi = I{word i appears in email}

    We want to model p(x|y), p(y)
    But there are 2^10.000 possible values of x, so we will need some assumptions to model this.

    Assume x's are conditionally independent given y. (Naive Bayes assumption. This is totally not true, but it isn't horrible)

    So

    p(x1,...,x10k|y) = p(x1|y)p(x2|x1,y)p(x3|x1,x2,y)...(p(x10k|...))

    we assume as

    = p(x1|y)p(x2|y)...p(x10k|y)

    ### Parameters:
    phi_j|y=1 = p(xj=1 | y = 1) (If y is spam, whats the chance of j appearing?)
    phi_j|y=0 = p(xj=1 | y = 0) (If y is not spam, whats the chance of j appearing?)
    phi_y = p(y=1) (Chance of the next email is spam)

    - L(phi_y,phi_i|y) = product(p(xi,yi, phi_y, phi_j|y))
    - MLE(phi_y) = sum(I{yi = 1}) / m
    - phi_j|y=1 = sum(I{xi_j - 1, yi = 1}) / sum(I{yi = 1})
    - phi_j|y=0 = sum(I{xi_j - 1, yi = 0}) / sum(I{yi = 0})

    It should nearly work, it isn't horrible (with one fix). Logistic Regression will always beat this, but it is more efficient to train.
    """
    )
    return


if __name__ == "__main__":
    app.run()
