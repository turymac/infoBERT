import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


sys.path.append(os.getcwd())

from base_args import add_base_args
from evaluation.correlation import compute_correlation_personal_marks, compute_correlation_form_marks
from utils.utils import get_test_df, get_basic_stat_clustering, get_personal_scores_df
from clustering.clustering import apply_clustering

def compute_embeddings(args, model, save_path="datasets/training_set/precomputed_embeddings"):
    model_name = args.model
    dataset_version = args.dataset

    os.makedirs(save_path, exist_ok=True)
    embedding_file = os.path.join(save_path, f"{dataset_version}_{model_name}_embeddings.npy")

    if os.path.exists(embedding_file):
        print(f"Loading precomputed embeddings from {embedding_file}...")
        return np.load(embedding_file)

    df = pd.read_excel(f"datasets/training_set/environmental_sentences_filtered_{dataset_version}.xlsx")
    df['Label'] = df['Environmental Sentences'] # Modify xlsx
    sentences = df.Label.tolist()
    print(f"{len(sentences)} training sentences read")

    print(f"Computing embeddings for dataset {dataset_version}...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    np.save(embedding_file, embeddings)
    return embeddings

def compute_test_sentences_embeddings(args, model, save_path="datasets/test_set/precomputed_embeddings"):
    model_name = args.model

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
    parser = argparse.ArgumentParser() # Can be simplified
    parser = add_base_args(parser)

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    embeddings = compute_embeddings(args=args, model=model)
    embedded_sentences = compute_test_sentences_embeddings(args=args, model=model)

    test_df = get_test_df()
    scores_df = get_personal_scores_df()

    yhat, centroids = apply_clustering(args=args, embeddings=embeddings)

    n_clusters, min_cluster_size, mean_cluster_size, max_cluster_size = get_basic_stat_clustering(yhat)

    if n_clusters > 1:
        # print(f"@ {bandwidth:.1f}")
        # silhouette_meanshift = silhouette_score(embeddings, yhat) create an alternative function or put inside get_basic_stat_clustering
        # print(f"- S: {silhouette_meanshift:.3f}")
        if args.marks == "personal":
            correlations = compute_correlation_personal_marks(args, embedded_sentences, centroids, test_df, scores_df, verbose=True)
        else:
            compute_correlation_form_marks(args, embedded_sentences, centroids, test_df, verbose=True)
    else:
        pass
        # break  # Se c'è un solo cluster, interrompiamo il ciclo
    #reduced_embeddings = apply_pca(embeddings)

    #clustering_algorithms = args.clustering_alg.split(",") if args.clustering_alg != "all" else [
    #    "kmeans", "affinity", "meanshift", "spectral", "agglomerative", "dbscan", "optics", "birch", "bisecting_kmeans"]

    #for alg in clustering_algorithms:
        # print(f"Running clustering with {alg}...")
        # labels = apply_clustering(reduced_embeddings, alg)
        # score = silhouette_score(reduced_embeddings, labels)
        # print(f"Silhouette Score for {alg}: {score:.4f}")


if __name__ == "__main__":
    main()