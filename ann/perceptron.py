import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib as plt
    import numpy as np
    from itertools import product
    import operator as opr
    return mo, np, opr, product


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
      y' = \text{sign}(w^T x + b)
      ```

    - **Regra de aprendizado:** Para cada amostra mal classificada `(x, y)`:

      ```latex
      w \leftarrow w + y \cdot x
      b \leftarrow b + y
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
            self.weights = np.zeros(n + 1)

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
    inputs = list(product([0, 1], repeat=2))
    d = 2
    t = 100
    tables = list(zip(
        ["AND", "OR", "XOR"],
        [
            list(zip(inputs, [2 * op(pair[0], pair[1]) - 1 for pair in inputs]))
            for op in [opr.and_, opr.or_, opr.xor]
        ],
    ))

    for op_name, samples in tables:
        print(f"{op_name}")
        per = Perceptron(d)
        per.train(samples, t)
        for x, y in samples:
            print(f"{x} -> {y};; prediction: {per.evaluate(x)} ")
    return


if __name__ == "__main__":
    app.run()
