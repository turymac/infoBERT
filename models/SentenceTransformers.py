import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

sys.path.append(os.getcwd())
from base_args import add_base_args


def compute_embeddings(model, dataset_version, save_path="datasets/training_set/precomputed_embeddings"):
    model_name = model.model_name_or_path

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

def compute_test_sentences_embeddings(model, save_path="datasets/test_set/precomputed_embeddings"):
    model_name = model.model_name_or_path

    os.makedirs(save_path, exist_ok=True)
    embedding_file = os.path.join(save_path, f"{model_name}_test_sentences_embeddings.pkl")
    if os.path.exists(embedding_file):
        print(f"Loading precomputed embeddings from {embedding_file}...")
        return pickle.load(open(embedding_file, "rb"))

    df = pd.read_excel(f"datasets/test_set/testset_eng_categories.xlsx")
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

    embeddings = compute_embeddings(model=model, dataset_version="v3")
    embedded_sentences = compute_test_sentences_embeddings(model=model)
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