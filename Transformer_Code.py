import csv
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from Plot_functions import (
    plot_aligned_topk_retrieval,
    plot_average_metrics,
    plot_cosine_similarity,
    plot_language_cloud,
    plot_procrustes_error,
    plot_topk_retrieval,
)



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
Fits a rotation that maps target embeddings into the source embedding space.
Returns the rotation, aligned target embeddings, mean alignment error, and means.
"""
def procrustes_align(source, target):
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)

    src_centered = source - source_mean
    tgt_centered = target - target_mean

    # Find the orthogonal map from target coordinates into source coordinates.
    H = tgt_centered.T @ src_centered
    U, _, Vt = np.linalg.svd(H)
    rotation = U @ Vt

    aligned_target = (tgt_centered @ rotation) + source_mean

    mean_error = np.mean(np.linalg.norm(source - aligned_target, axis=1))
    return rotation, aligned_target, mean_error, source_mean, target_mean

# ----------------------------------------------------------------
# CSLS
# ----------------------------------------------------------------

def csls(source, target, correct_indices, csls_k=3):
    src_norm = source / np.linalg.norm(source, axis=1, keepdims=True)
    tgt_norm = target / np.linalg.norm(target, axis=1, keepdims=True)

    sim_matrix = src_norm @ tgt_norm.T

    if csls_k > 0:
        k = min(csls_k, target.shape[0], source.shape[0])

        topk_tgt_sims = np.sort(sim_matrix, axis=1)[:, -k:]
        r_T = np.mean(topk_tgt_sims, axis=1, keepdims=True)


        topk_src_sims = np.sort(sim_matrix, axis=0)[-k:, :]
        r_S = np.mean(topk_src_sims, axis=0, keepdims=True)

        csls_score = 2 * sim_matrix - r_T - r_S

    else:
        csls_score = sim_matrix

    top_indices = np.argsort(csls_score, axis=1)[:, ::-1]
    correct = np.array(correct_indices)

    top1 = np.sum(top_indices[:, 0] == correct)
    top5 = np.sum(np.any(top_indices[:, :5] == correct[:, None], axis=1))

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
Main caller for the computation of all the metrics
Calls on the cosine sim, nearest neighbour and procrustes methods to calculate
respective metrics and returns them as a dictionary with keys and values
source_by_layer: source language embeddings from each layer for a given model
target_by_layer: target language embeddings from each layer for a given model
Returns layer_metrics dictionary with the name of the metric and its results
"""

def evaluate_layers(source_by_layer, target_by_layer):
    layer_metrics = []
    for layer_index, (source, target) in enumerate(zip(source_by_layer, target_by_layer)):

        # Cosine similarity
        scores = [cosine_similarity(src, tgt) for src, tgt in zip(source, target)]

        # Top-k retrieval
        correct_indices = list(range(len(source)))
        top1_source, top5_source = csls(
            source, target, correct_indices, csls_k=3
        )
        top1_target, top5_target = csls(
            target, source, correct_indices, csls_k=3
        )

        # procrustes 
        train_number = 80
        source_train = source[:train_number]
        target_train = target[:train_number]
        source_test = source[train_number:]
        target_test = target[train_number:]
        rotation, aligned_source, procrustes_error, source_mean, target_mean = procrustes_align(
            source_train, target_train
            )
        aligned_source_test = (source_test - source_mean) @ rotation + target_mean
        aligned_target_test = (target_test - target_mean) @ rotation.T + source_mean
        correct_idx = list(range(len(source_test)))
        ali_top1_source, ali_top5_source = csls(
            aligned_source_test, target_test, correct_idx, csls_k=3
        )
        ali_top1_target, ali_top5_target = csls(
            target_test, aligned_source_test, correct_idx, csls_k=3
        )

        layer_name = "Embedding output" if layer_index == 0 else f"Layer {layer_index}"
        metrics = {
            "layer": layer_name,
            "avg_cosine": float(np.mean(scores)),
            "top1_source": top1_source,
            "top5_source": top5_source,
            "top1_target": top1_target,
            "top5_target": top5_target,
            "aligned_top1_source": ali_top1_source,
            "aligned_top5_source": ali_top5_source,
            "aligned_top1_target": ali_top1_target,
            "aligned_top5_target": ali_top5_target,
            "procrustes_error": procrustes_error,
        }
        layer_metrics.append(metrics)

        print(
            f"{layer_name}: cosine={metrics['avg_cosine']:.4f}, "
            f"top-1 source={top1_source:.4f}, top-5 source={top5_source:.4f}, "
            f"top-1 target={top1_target:.4f}, top-5 target={top5_target:.4f}, "
            f"ali top-1 source={ali_top1_source:.4f}, ali top-5 source={ali_top5_source:.4f}, "
            f"ali top-1 target={ali_top1_target:.4f}, ali top-5 target={ali_top5_target:.4f}, "
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
        plot_cosine_similarity(results, dataset_name)
        plot_topk_retrieval(results, dataset_name)
        plot_procrustes_error(results, dataset_name)
        plot_aligned_topk_retrieval(results, dataset_name)
        plot_average_metrics(results, dataset_name)

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
