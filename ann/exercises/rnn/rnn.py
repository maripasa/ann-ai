import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import csv
    import kagglehub
    return (kagglehub,)


@app.cell
def _(kagglehub):
    path = kagglehub.dataset_download("ranjitha1/hotel-reviews-city-chennai")
    print(path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
