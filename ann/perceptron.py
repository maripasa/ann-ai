import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product
    import operator as opr
    return mo, np, opr, plt, product


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Perceptron

    - **Tipo:** Classificador linear binário  
    - **Entrada:** Vetor `x = [x₁, x₂, ..., xₙ]`  
    - **Pesos:** Vetor `w = [w₁, w₂, ..., wₙ]` mais o viés `b`  
    - **Saída:**  

      ```latex
      y' = sign(w^T x + b)
      ```

    - **Regra de aprendizado:** Para cada amostra mal classificada `(x, y)`:

      ```latex
      w <- w + y * x
      b <- b + y
      ```

    - **Propriedades:**
      - Apenas aprende padrões linearmente separáveis  
      - Base das redes neurais  
      - Atualiza pesos iterativamente até convergência ou número máximo de iterações
    """
    )
    return


@app.cell
def _(np):
    class Perceptron:
        def __init__(self, n: int):
            self.weights = np.ones(n + 1)

        def train(self, samples, epochs: int):
            for _ in range(epochs):
                for x, y in samples:
                    if self.evaluate(x) != y:
                        x = np.insert(x, 0, 1)
                        self.weights += y * x

        def evaluate(self, x):
            x = np.insert(x, 0, 1)
            return np.sign(self.weights @ x)
    return (Perceptron,)


@app.cell
def _(Perceptron, opr, product):
    """
    Maps {0,1} to {-1,1}
    """
    def binary_to_bipolar(x):
        return 2 * x - 1

    inputs = list(product([0, 1], repeat=2))
    d = 2
    t = 100
    tables = list(zip(
        ["AND -> Should work", "OR -> Should work", "XOR -> Should not work"],
        [
            list(zip(inputs, [binary_to_bipolar(op(pair[0], pair[1])) for pair in inputs]))
            for op in [opr.and_, opr.or_, opr.xor]
        ],
    ))

    perceptrons = []
    for op_name, samples in tables:
        print(f"{op_name}")
        per = Perceptron(d)
        per.train(samples, t)
        perceptrons.append(per)
        for _x, _y in samples:
            print(f"{_x} -> {_y};; prediction: {per.evaluate(_x)} ")
    return inputs, perceptrons


@app.cell
def _(inputs, np, perceptrons, plt):
    _x = [p[1] for p in inputs]
    _y = [p[0] for p in inputs]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.scatter(_x, _y)
        for _xi, _yi in zip(_x, _y):
            ax.text(_xi + 0.08, _yi + 0.08, f"({_xi}, {_yi})", fontsize=8)

    ops = ["And", "Or", "Xor"]

    x = np.linspace(-1, 4, 100)
    def y_line(x, w):
        return -(w[0] + w[1]*x) / w[2]

    for i in range(3):
        axs[i].set_title(ops[i])
        y = y_line(x, perceptrons[i].weights)
        axs[i].plot(x, y, 'r-', label='Decision boundary')


    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Xor even gives an error! Since it's weights averaged out: 0. 0. 0.""")
    return


if __name__ == "__main__":
    app.run()
