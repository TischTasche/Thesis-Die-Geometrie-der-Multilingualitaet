import csv
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.decomposition import PCA


access_token = os.getenv("HF_TOKEN")

# ----------------------------------------------------------------
# Model list
# ----------------------------------------------------------------
MODEL_NAMES = [
    "xlm-mlm-en-2048",
    "xlm-roberta-base",
    "bert-base-cased",
    "bert-base-multilingual-cased",   
]



# ----------------------------------------------------------------
# Word Pairs for comparison
# ----------------------------------------------------------------

DATASETS = [
    {
        "name": "english_german",
        "filename": "english_german_words.csv",
        "target_column": "german",
        "target_language": "German",
    },
    {
        "name": "english_french",
        "filename": "english_french_words.csv",
        "target_column": "french",
        "target_language": "French",
    },
]


# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------

"""
Turns sequence of token vectors into one vector for whole input with shape 
(batch_size, sequence_length, hidden_size) for each model
token_vector: output of the current hidden layer
attention_mask: mask to detect which are padding and which are real values
Returns the sequence of tokens as one mean vector 
"""
def mean_pool(token_vectors, attention_mask=None):
    if attention_mask is not None:
        # Mask value: 
        # 0: ignored (padding)
        # 1: included (value)
        mask = attention_mask.unsqueeze(-1).to(token_vectors.dtype)
        token_vectors = token_vectors * mask
        token_count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return token_vectors.sum(dim=1) / token_count

    return token_vectors.mean(dim=1)


"""
Center the representations embeddings by taking the mean of all vectors
and subtracting it from each embedding
embeddings: mean vec"""
def mean_center(embeddings):
    embedding_mean = embeddings.mean(axis=0, keepdims=True)
    return embeddings - embedding_mean

"""
Uses a tokenizer on the given text, loads the model and then inserts tokens on the model 
to retrieve the embeddings for the tokenized word pairs for each layer
model: current language model used for the representation embeddings
tokenizer: Tokenizer to transform word pairs into tokens
text: word pairs from two seperate languages
Returns the embeddings from word pairs inserted into the model 
"""
def get_layer_embeddings(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt") # tokenize word pairs
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_embeddings = []
    for hidden_state in outputs.hidden_states:
        pooled = mean_pool(hidden_state, inputs.get("attention_mask"))
        layer_embeddings.append(pooled[0].to(torch.float32).cpu().numpy())

    return np.stack(layer_embeddings)

"""
Computing the similarity between two vectors in their respective vector space
Result <= 1: vectors point in same direction
Result = 0: vectors are orthogonal to each other
Result >= -1: vectors point in opposite direction
Equation: (a * b) / (||a|| * ||b||)
a, b: two vector (word embeddings) from different languages
Returns a value between 1 and -1
"""
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


# ----------------------------------------------------------------
# Procrutes Alignment
# ----------------------------------------------------------------
"""
Rotating a word embedding using singular value decomposition to align it with 
a target word embedding 
source, target: word embeddings from source and target languages
Returns rotation matrix R, an aligned target embedding and a procrustes error coefficient
"""
def procrustes_align(source, target):
    # Center each language's embeddings before finding the rotation
    source_mean = source.mean(axis=0, keepdims=True)
    src_centered = mean_center(source)
    tgt_centered = mean_center(target)

    # Calculating the rotation matrix R using SVD
    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Appling rotation matrix R on target embedding
    aligned_tgt = (tgt_centered @ R) + source_mean

    # Calculating the procrustes error
    mse = np.mean(np.linalg.norm(source - aligned_tgt, axis=1))
    return R, aligned_tgt, mse

# ----------------------------------------------------------------
# K-Nearest Neigbour
# ----------------------------------------------------------------

"""
Calculates the cosine similarity for source and target embeddings,
the highest cosine similarities (top 1 and 5) are compared to correct
word embedding in target embedding, if top1 or top5 contains correct translation
accumulator is increased
source, target: word embeddings from two different languages
correct_indices: array with indices to correct translation
Returns float number of correct guesses between 0 and 1
"""
def nearest_neighbour(source, target, correct_indices):
    top1 = 0
    top5 = 0

    for i, src in enumerate(source):
        sims = []
        # Calculating cosine sim for every word embed in source and target
        for j, tgt in enumerate(target):
            sims.append((j, cosine_similarity(src, tgt)))

        # Sorting the cosine sim so "highest" (~1) are at the front of sims[]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)

        # Selecting the top 5 cosine sim values
        topk = [idx for idx, _ in sims[:5]]

        if correct_indices[i] in topk[:1]: # acc if highest cosine sim is correct
            top1 += 1 
        if correct_indices[i] in topk[:5]: # acc if one of the highest 5 cosine sim is correct
            top5 += 1

    # Normalize the values by dividing by size
    n = len(source)
    return top1 / n, top5/ n



# ----------------------------------------------------------------
# Model call, embedding generation and evaluation
# ----------------------------------------------------------------
"""
Handles list of words, calls get_layer_embedding on each word and
stacks the resulting embedding into one embedding
Words: list of words from word pair data set
Model: current model used for comparison
Tokenizer: respective tokenizer for given model to tokenize words
Returns embedding of the selected words list of a given model

"""
def embed_words_by_layer(words, model, tokenizer):
    embeddings = [
        get_layer_embeddings(model, tokenizer, word)
        for word in words
    ]
    return np.stack(embeddings, axis=1)


"""
Main computation of all the metrics
Calls on the cosine sim, nearest neighbour and procrustes methods to calculate
respective metrics and returns them as a dictionary with keys and values
source_by_layer: source language embeddings from each layer for a given model
target_by_layer: target language embeddings from each layer for a given model
Returns layer_metrics containing the name and the metrics for each method
"""

def evaluate_layers(source_by_layer, target_by_layer):
    layer_metrics = []
    for layer_index, (source, target) in enumerate(zip(source_by_layer, target_by_layer)):
        scores = [cosine_similarity(src, tgt) for src, tgt in zip(source, target)]
        correct_indices = list(range(len(source)))
        top1_source, top5_source = nearest_neighbour(source, target, correct_indices)
        top1_target, top5_target = nearest_neighbour(target, source, correct_indices)
        _, aligned_target, procrustes_error = procrustes_align(source[:30], target[:30])
        aligned_top1_source, aligned_top5_source = nearest_neighbour(
            source[30:], aligned_target, correct_indices
        )
        aligned_top1_target, aligned_top5_target = nearest_neighbour(
            aligned_target, source, correct_indices
        )

        layer_name = "Embedding output" if layer_index == 0 else f"Layer {layer_index}"
        metrics = {
            "layer": layer_name,
            "avg_cosine": float(np.mean(scores)),
            "top1_source": top1_source,
            "top5_source": top5_source,
            "top1_target": top1_target,
            "top5_target": top5_target,
            "aligned_top1_source": aligned_top1_source,
            "aligned_top5_source": aligned_top5_source,
            "aligned_top1_target": aligned_top1_target,
            "aligned_top5_target": aligned_top5_target,
            "procrustes_error": procrustes_error,
        }
        layer_metrics.append(metrics)

        print(
            f"{layer_name}: cosine={metrics['avg_cosine']:.4f}, "
            f"top-1 source={top1_source:.4f}, top-5 source={top5_source:.4f}, "
            f"top-1 target={top1_target:.4f}, top-5 target={top5_target:.4f}, "
            f"aligned top-1 source={aligned_top1_source:.4f}, "
            f"aligned top-5 source={aligned_top5_source:.4f}, "
            f"aligned top-1 target={aligned_top1_target:.4f}, "
            f"aligned top-5 target={aligned_top5_target:.4f}, "
            f"Procrustes error={procrustes_error:.4f}"
        )

    return layer_metrics


"""
Loads model and respective tokenizer locally to be used for computation
model_name: name of the current model to load from Hugging Face
Returns model and tokenizer
"""

def load_model(model_name):
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        token=access_token
    )
    model.eval()
    return model, tokenizer


"""
Combines all outputs from other functions by retrieving data set and splitting
into source and target words, getting layer embeddings for source and target
and retrieving the respective metrics for the words
model_name: name of the current model 
pairs: dataset word pairs with a source (pos [0]) and target (pos [1]) splitting
dataset_name: for retrieval of either eng_ger or eng_fre
model: model that was load to be used locally
tokenizer: respective tokenizer for the given model
Returns calculated metrics of dataset word pairs for given model 
"""
def compare_model(model_name, pairs, dataset_name, model, tokenizer):
    print(f"\nComparing model: {model_name} using {dataset_name}")
    source_words = [pair[0] for pair in pairs]
    target_words = [pair[1] for pair in pairs]
    source_layer = embed_words_by_layer(source_words, model, tokenizer)
    target_layer = embed_words_by_layer(target_words, model, tokenizer)
    layer_metrics = evaluate_layers(source_layer, target_layer)

    return {
        "layer_metrics": layer_metrics,
        "source_embeddings": source_layer[-1],
        "target_embeddings": target_layer[-1],
    }


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



# ----------------------------------------------------------------
# Load word pairs from csv files
# ----------------------------------------------------------------
def load_word_pairs(dataset):
    dataset_path = Path(__file__).with_name(dataset["filename"])
    with dataset_path.open(encoding="utf-8", newline="") as dataset_file:
        return [
            (row["english"], row[dataset["target_column"]])
            for row in csv.DictReader(dataset_file)
        ]

# ----------------------------------------------------------------
# Main call function 
# ----------------------------------------------------------------

def main():
    word_pairs_by_dataset = {
        dataset["name"]: load_word_pairs(dataset)
        for dataset in DATASETS
    }
    results_by_dataset = {
        dataset["name"]: {}
        for dataset in DATASETS
    }

    for model_name in MODEL_NAMES:
        model, tokenizer = load_model(model_name)
        for dataset in DATASETS:
            dataset_name = dataset["name"]
            results_by_dataset[dataset_name][model_name] = compare_model(
                model_name,
                word_pairs_by_dataset[dataset_name],
                dataset_name,
                model,
                tokenizer,
            )
        del model, tokenizer

    for dataset in DATASETS:
        dataset_name = dataset["name"]
        results = results_by_dataset[dataset_name]
        plot_layer_metrics(results, dataset_name)
        plot_procrustes_retrieval(results, dataset_name)
        plot_average_metrics(results, dataset_name)
        plot_procrustes_error_table(results, dataset_name)

    german_results = results_by_dataset["english_german"]
    french_results = results_by_dataset["english_french"]
    for model_name in MODEL_NAMES:
        plot_language_cloud(
            model_name,
            german_results[model_name],
            french_results[model_name],
        )


if __name__ == "__main__":
    main()
