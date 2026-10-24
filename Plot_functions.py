import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.decomposition import PCA



# ----------------------------------------------------------------
# PCA and Pyplot
# ----------------------------------------------------------------
"""
Helper function to easily save plots into folder in workspace
figuer: current produced plot
filename: filename to be used (e.g. procrustes_error)
dataset_name: assiging image to the write folder path
"""
def save_figure(figure, filename, dataset_name):
    image_directory = Path(__file__).with_name("images")
    image_directory = image_directory / dataset_name
    image_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(image_directory / filename, dpi=300, bbox_inches="tight")


"""
Plots the results of one metric of each layer and model into one plot
IMPORTANT: DOES NOT CHECK LAYER DIFFERENCES 12 vs 24
results: layer metrics of each model
dataset_name: assigining each plot to each language pair correctly
Returns plots for each metric containing the results of each layer and model
"""
def plot_layer_metrics(results, dataset_name):
    metric_plots = [
        ("avg_cosine", "Cosine similarity", None),
        ("top1_source", "Top-1 translation retrieval", (0, 1)),
        ("top5_source", "Top-5 translation retrieval", (0, 1)),
        ("top1_target", "Top-1 translation retrieval", (0, 1)),
        ("top5_target", "Top-5 translation retrieval", (0, 1)),
        ("procrustes_error", "Procrustes alignment error", None),
    ]
    for metric_key, title, y_limits in metric_plots:
        figure, axis = plt.subplots(figsize=(9, 6))
        for model_name, model_result in results.items():
            layer_metrics = model_result["layer_metrics"]
            layer_indices = range(len(layer_metrics))
            values = [metric[metric_key] for metric in layer_metrics]
            axis.plot(layer_indices, values, marker="o", markersize=3, label=model_name)

        axis.set_title(title + " for " + dataset_name)
        axis.set_xlabel("Hidden-state index")
        axis.set_ylabel("Error" if metric_key == "procrustes_error" else "Accuracy")
        axis.grid(True, alpha=0.3)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
        axis.legend(title="Model")
        figure.tight_layout()
        save_figure(figure, f"{metric_key}.png", dataset_name)

    plt.show()


"""
Plots post-Procrustes top-k retrieval accuracy for each model and layer.
"""
def plot_procrustes_retrieval(results, dataset_name):
    table_rows = []
    for model_name, model_result in results.items():
        for layer in model_result["layer_metrics"]:
            table_rows.append([
                model_name,
                layer["layer"],
                f"{layer['aligned_top1_source']:.1%}",
                f"{layer['aligned_top5_source']:.1%}",
                f"{layer['aligned_top1_target']:.1%}",
                f"{layer['aligned_top5_target']:.1%}",
            ])

    figure, axis = plt.subplots(
        figsize=(13, max(4, len(table_rows) * 0.3 + 1.5))
    )
    axis.axis("off")
    axis.set_title(f"Post-Procrustes retrieval accuracy by layer: {dataset_name}")
    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "Model",
            "Layer",
            "Top-1 source→target",
            "Top-5 source→target",
            "Top-1 target→source",
            "Top-5 target→source",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    figure.tight_layout()
    save_figure(figure, "procrustes_retrieval_table.png", dataset_name)
    plt.show()


"""
Plots a table show casing the average of all metrics across layers for each model
So average of cosine similarity of e.g. XLM of all layers
results: calculated layer metrics of all models
dataset_name: assigning each plot the respective language combination
Returns plots of tables with averages
"""
def plot_average_metrics(results, dataset_name):
    summary_metrics = [
        ("avg_cosine", "Cosine Similarity"),
        ("top1_source", "Top-1 Retrieval (source)"),
        ("top5_source", "Top-5 Retrieval (source)"),
        ("top1_target", "Top-1 Retrieval (target)"),
        ("top5_target", "Top-5 Retrieval (target)"),
    ]
    model_names = list(results)
    summary_rows = [
        [
            f"{np.mean([layer[metric_key] for layer in results[model_name]['layer_metrics']]):.3%}"
            for metric_key, _ in summary_metrics
        ]
        for model_name in model_names
    ]
    summary_figure, summary_axis = plt.subplots(figsize=(12, max(3, len(model_names) * 0.7)))
    summary_axis.axis("off")
    summary_axis.set_title(f"Average performance across layers for: {dataset_name}")
    summary_table = summary_axis.table(
        cellText=summary_rows,
        rowLabels=model_names,
        colLabels=[label for _, label in summary_metrics],
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10)
    summary_table.scale(1, 1.5)
    summary_figure.tight_layout()
    save_figure(summary_figure, "average_metrics.png", dataset_name)
    plt.show()


def plot_procrustes_error_table(results, dataset_name):
    layer_names = list(dict.fromkeys(
        layer["layer"]
        for model_result in results.values()
        for layer in model_result["layer_metrics"]
    ))
    model_names = list(results)
    table_rows = []
    for model_name in model_names:
        errors_by_layer = {
            layer["layer"]: f"{layer['procrustes_error']:.4f}"
            for layer in results[model_name]["layer_metrics"]
        }
        table_rows.append([errors_by_layer.get(layer_name, "") for layer_name in layer_names])

    figure, axis = plt.subplots(figsize=(max(10, len(layer_names) * 1.2), max(3, len(model_names) * 0.7)))
    axis.axis("off")
    axis.set_title(f"Procrustes alignment error by layer: {dataset_name}")
    table = axis.table(
        cellText=table_rows,
        rowLabels=model_names,
        colLabels=layer_names,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    figure.tight_layout()
    save_figure(figure, "procrustes_error_table.png", dataset_name)
    plt.show()

"""
Plots 3 dimensional graphs showcasing dimension reduction using PCA
model_name: given model name of current model used for plotting
german_result: factorized results from calculating results on eng_ger
french_results: factorized results from calculating results on eng_fre
Returns 3d graph with all language representations
"""
def plot_language_cloud(model_name, german_result, french_result):
    english_embeddings = np.vstack([
        german_result["source_embeddings"],
        french_result["source_embeddings"],
    ])
    german_embeddings = german_result["target_embeddings"]
    french_embeddings = french_result["target_embeddings"]
    all_embeddings = np.vstack([
        english_embeddings,
        german_embeddings,
        french_embeddings,
    ])
    labels = (
        ["English"] * len(english_embeddings)
        + ["German"] * len(german_embeddings)
        + ["French"] * len(french_embeddings)
    )

    reducer = PCA(n_components=3)
    coords = reducer.fit_transform(all_embeddings)
    explained_variance = reducer.explained_variance_ratio_

    figure = plt.figure(figsize=(9, 7))
    axis: Axes3D = figure.add_subplot(111, projection="3d")
    for language in ("English", "German", "French"):
        idx = [i for i, label in enumerate(labels) if label == language]
        axis.scatter(
            coords[idx, 0],
            coords[idx, 1],
            cast(Any, coords[idx, 2]),
            label=language,
            alpha=0.75,
        )

    axis.set_title(
        f"English-German-French - {model_name} - PCA\n"
        f"First three components explain {explained_variance.sum():.1%} of variance"
    )
    axis.set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
    axis.set_ylabel(f"PC2 ({explained_variance[1]:.1%})")
    axis.set_zlabel(f"PC3 ({explained_variance[2]:.1%})")
    axis.legend()
    figure.tight_layout()
    safe_model_name = model_name.replace("/", "_")
    save_figure(figure, f"language_cloud_{safe_model_name}.png", "english_german_french")
    plt.show()
