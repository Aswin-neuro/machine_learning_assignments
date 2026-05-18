import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3.(a) For each molecule in the MUTAG dataset, compute the number of 6-membered rings as a new annotation. You may use an automated/manual approach.

    (b) Using these ring counts as target labels, train a GNN to predict the number of 6-membered rings in a molecule. You may treat this as a regression or a multi-class

    classification problem; justify your choice. Report appropriate evaluation metrics on a held-out test set.
    """)
    return


@app.cell
def _():
    import torch
    import networkx as nx
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_networkx
    from torch_geometric.loader import DataLoader
    from torch.nn import Linear
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
