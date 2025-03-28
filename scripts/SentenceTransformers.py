import argparse
import os
import re
import sys

import pickle
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


sys.path.append(os.getcwd())

from base_args import add_base_args
from evaluation.correlation import compute_correlation_personal_marks, compute_correlation_form_marks
from evaluation.distance import partial_distance_knn_to_excel
from evaluation.accuracy import compute_accuracy, compute_auroc
from utils.utils import get_test_df, get_basic_stat_clustering, get_personal_scores_df, get_label_df
from clustering.clustering import apply_clustering

def compute_embeddings(args, model, save_path="datasets/training_set/precomputed_embeddings"):
    model_name = args.model.replace("/", "_")
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

# def split_sentences(text):
#     # Non splitta URL (es: www.example.com)
#     pattern = r'(?<!\b\w{2,10})\.(?!\w{2,4}\b|com|org|net|it|html|php)'
#     parts = [s.strip() for s in re.split(pattern, text) if s.strip()]
#     return parts

def compute_distances_by_checkpoint(args, checkpoints_dir, output_dir="checkpoints_eval"):
    os.makedirs(output_dir, exist_ok=True)

    df = get_test_df()
    print(f"{len(df)} test labels read")

    # Step 1: Creazione DataFrame frasi splittate (mantenendo traccia del documento originale)
    all_sentences = []
    for row in df.itertuples(index=False):
        frasi = row.Label.split('.') #split_sentences(getattr(row, sentence_column))
        all_sentences.extend(frasi)

    # Rimuove spazi e filtra le frasi vuote ex post
    all_sentences = [s.strip() for s in all_sentences if s.strip()]
    # Ordina
    all_sentences = sorted(all_sentences)

    # Per costruzione: Excel base da restituire
    excel_df = pd.DataFrame({"Sentence": all_sentences})

    # Step 2: Caricamento dei checkpoint e calcolo delle distanze
    for file in sorted(os.listdir(checkpoints_dir)):
        ckpt_path = os.path.join(checkpoints_dir, file)
        if not os.path.isdir(ckpt_path):
            continue
        try:
            model = SentenceTransformer(ckpt_path)
        except Exception as e:
            print(f"Errore nel caricamento del modello da {ckpt_path}: {e}")
            continue

        embeddings = compute_embeddings(args=args, model=model)

        yhat, centroids = apply_clustering(args=args, embeddings=embeddings)

        n_clusters, min_cluster_size, mean_cluster_size, max_cluster_size = get_basic_stat_clustering(yhat)

        if n_clusters > 1:
            test_embeddings = model.encode(all_sentences, show_progress_bar=True, convert_to_numpy=True)

            # Calcola le distanze di ciascun embedding da tutti i centroidi
            compute_distance = get_metric(args.metric)
            # Calcola una matrice (n_frasi x n_centroidi)
            distance_matrix = np.array([
                [compute_distance(embedding, centroid) for centroid in centroids]
                for embedding in test_embeddings
            ])

            # Per ogni frase, seleziona la distanza dal centroide più vicino
            min_dists = np.min(distance_matrix, axis=1)

            # # Salva anche come .npy
            # ckpt_name = file.replace("/", "_")
            # npy_path = os.path.join(output_dir, f"{ckpt_name}_min_distances.npy")
            # np.save(npy_path, min_dists)

            # Aggiunge la colonna delle distanze nel DataFrame finale
            excel_df[file] = min_dists

    # Salva l'excel finale
    excel_path = os.path.join(output_dir, "distances_by_checkpoint.xlsx")
    excel_df.to_excel(excel_path, index=False)

    return excel_df

def main():
    parser = argparse.ArgumentParser() # Can be simplified
    parser = add_base_args(parser)

    args = parser.parse_args()

    if args.run == "ckp_distance": # Temp
        compute_distances_by_checkpoint(args)
        return

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    embeddings = compute_embeddings(args=args, model=model)
    # Write compute_test_sentences_label

    test_df = get_test_df()

    yhat, centroids = apply_clustering(args=args, embeddings=embeddings)

    n_clusters, min_cluster_size, mean_cluster_size, max_cluster_size = get_basic_stat_clustering(yhat)

    if n_clusters > 1:
        # print(f"@ {bandwidth:.1f}")
        # silhouette_meanshift = silhouette_score(embeddings, yhat) create an alternative function or put inside get_basic_stat_clustering
        # print(f"- S: {silhouette_meanshift:.3f}")
        if args.run == "correlation":
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