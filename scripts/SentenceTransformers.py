import argparse
import json
import os
import re
import sys

import pickle
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

sys.path.append(os.getcwd())

from base_args import add_base_args
from evaluation.checkpoints import compute_distances_by_checkpoint, compute_correlation_by_checkpoint, compute_accuracy_by_checkpoint
from evaluation.correlation import compute_correlation_personal_marks, compute_correlation_form_marks
from evaluation.distance import partial_distance_knn_to_excel
from evaluation.accuracy import compute_accuracy, compute_auroc
from evaluation.quantity import compute_information_quantity_personal, compute_information_quantity_form, compute_information_quantity_all_thr_form, compute_information_quantity_all_thr_personal
from utils.utils import get_test_df, get_basic_stat_clustering, get_personal_scores_df, get_label_df, apply_filtering
from clustering.clustering import apply_clustering

# load configuration from JSON
with open("base_config.json", "r") as f:
    BASE_CONFIG = json.load(f)

def compute_embeddings(args, model, save_path="datasets/training_set/precomputed_embeddings"):
    model_name = args.model.replace("/", "_")
    dataset_version = args.dataset

    os.makedirs(save_path, exist_ok=True)
    embedding_file = os.path.join(save_path, f"{dataset_version}_{model_name}_embeddings.npy")

    if os.path.exists(embedding_file):
        print(f"Loading precomputed embeddings from {embedding_file}...")
        embeddings = np.load(embedding_file)

    else:
        df = pd.read_excel(f"datasets/training_set/environmental_sentences_filtered_{dataset_version}.xlsx")
        df['Label'] = df['Environmental Sentences']  # Modify xlsx
        sentences = df.Label.tolist()
        print(f"{len(sentences)} training sentences read")

        print(f"Computing embeddings for dataset {dataset_version}...")
        embeddings = model.encode(sentences, show_progress_bar=True)
        np.save(embedding_file, embeddings)


    return embeddings


def compute_test_sentences_embeddings(args, model, save_path="datasets/test_set/precomputed_embeddings"):
    model_name = args.model.replace("/", "_")

    os.makedirs(save_path, exist_ok=True)
    embedding_file = os.path.join(save_path, f"{model_name}_test_sentences_embeddings.pkl")
    if os.path.exists(embedding_file):
        print(f"Loading precomputed embeddings from {embedding_file}...")
        return pickle.load(open(embedding_file, "rb"))

    df = get_test_df()
    print(f"{len(df)} test labels read")

    print(f"Computing {model_name} embeddings for test_set...")
    embedded_sentences = {}
    for row in df.itertuples(index=False):
        embedded_sentences[row.Name] = []
        # Divide il testo dell'etichetta in frasi usando il punto come separatore
        frasi = row.Label.split('.')
        # Itera su ogni frase estratta
        for frase in frasi:
            frase = frase.strip()  # Rimuove eventuali spazi bianchi
            if not frase:  # Salta le frasi vuote
                continue
                # Calcola l'embedding della frase
            embedding = model.encode(frase)
            embedded_sentences[row.Name].append((frase, embedding))
    pickle.dump(embedded_sentences, open(embedding_file, "wb"))

    return embedded_sentences


def main():
    parser = argparse.ArgumentParser()  # Can be simplified
    parser = add_base_args(parser)

    args = parser.parse_args()

    if args.run == "ckp_distance":  # Temp
        compute_distances_by_checkpoint(args)
        return
    if args.run == "ckp_correlation":
        scores_df = get_personal_scores_df()
        compute_correlation_by_checkpoint(args, scores_df)
        return
    if args.run == "ckp_accuracy":
        compute_accuracy_by_checkpoint(args)
        return

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    embeddings = compute_embeddings(args=args, model=model)
    # Write compute_test_sentences_label

    if args.filtering:
        filter_eps = BASE_CONFIG["filter_eps"]
        embeddings = apply_filtering(embeddings, filter_eps)

    test_df = get_test_df()

    yhat, centroids = apply_clustering(args=args, embeddings=embeddings)

    clustering_stats = get_basic_stat_clustering(yhat, embeddings)

    if clustering_stats["n_clusters"] > 1:
        print(f"n_clusters={clustering_stats['n_clusters']}, m={clustering_stats['min_cluster_size']}, M={clustering_stats['max_cluster_size']}, " +
              f"mean={clustering_stats['mean_cluster_size']}, S={clustering_stats['silhouette_score']}")
        if args.run == "quantity":
            embedded_sentences = compute_test_sentences_embeddings(args=args, model=model)
            if args.marks == "personal":
              scores_df = get_personal_scores_df()
              compute_information_quantity_all_thr_personal(args, embedded_sentences, embeddings, centroids, test_df, scores_df, verbose=True)
            else:
              compute_information_quantity_all_thr_form(args, embedded_sentences, embeddings, centroids, test_df, verbose=True)
        elif args.run == "correlation":
            embedded_sentences = compute_test_sentences_embeddings(args=args, model=model)
            if args.marks == "personal":
                scores_df = get_personal_scores_df()
                compute_correlation_personal_marks(args, embedded_sentences, centroids, test_df, scores_df, verbose=True)
            else:
                compute_correlation_form_marks(args, embedded_sentences, centroids, test_df, verbose=True)
        elif args.run == "distance":
            embedded_sentences = compute_test_sentences_embeddings(args=args, model=model)
            partial_distance_knn_to_excel(args, embedded_sentences, centroids, test_df)
        elif args.run == "accuracy":
            label_df = get_label_df()
            compute_accuracy(args, model, centroids, label_df)
        elif args.run == "auroc":
            label_df = get_label_df()
            compute_auroc(args, model, centroids, label_df)
        else:
            raise NotImplementedError


    else:
        pass
        # break  # Se c'è un solo cluster, interrompiamo il ciclo
    # reduced_embeddings = apply_pca(embeddings)

    # clustering_algorithms = args.clustering_alg.split(",") if args.clustering_alg != "all" else [
    #    "kmeans", "affinity", "meanshift", "spectral", "agglomerative", "dbscan", "optics", "birch", "bisecting_kmeans"]

    # for alg in clustering_algorithms:
    # print(f"Running clustering with {alg}...")
    # labels = apply_clustering(reduced_embeddings, alg)
    # score = silhouette_score(reduced_embeddings, labels)
    # print(f"Silhouette Score for {alg}: {score:.4f}")


if __name__ == "__main__":
    main()