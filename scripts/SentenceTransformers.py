import argparse
import os
import re
import sys

import pickle
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from base_args import add_base_args
from evaluation.correlation import compute_correlation_personal_marks, compute_correlation_form_marks
from evaluation.distance import partial_distance_knn_to_excel
from evaluation.accuracy import compute_accuracy, compute_auroc
from utils.utils import get_test_df, get_basic_stat_clustering, get_personal_scores_df, get_label_df, get_metric, get_label_score, get_best_threshold
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

def extract_ckpt_number(name):
    match = re.search(r"checkpoint-(\d+)", name)
    return int(match.group(1)) if match else float('inf')

def compute_distances_by_checkpoint(args, output_dir="checkpoints_eval"):

    checkpoints_dir = args.model
    model_name = args.model.replace("/", "_")
    dataset_version = args.dataset

    os.makedirs(output_dir, exist_ok=True)

    # Legge direttamente l'excel con frasi già splittate
    df = pd.read_excel("datasets/test_set/test_sentences.xlsx")
    print(f"{len(df)} test sentences read")

    # Ordina le frasi mantenendo category e label associati
    df = df.sort_values(by="sentence").reset_index(drop=True)

    # Crea il DataFrame finale con le colonne sentence e category
    excel_df = df[["category", "sentence"]].copy()

    # Step 2: Caricamento dei checkpoint e calcolo delle distanze
    for file in sorted(os.listdir(checkpoints_dir), key=extract_ckpt_number):
        ckpt_path = os.path.join(checkpoints_dir, file)
        if not os.path.isdir(ckpt_path):
            continue

        try:
            model = SentenceTransformer(ckpt_path)
        except Exception as e:
            print(f"Errore nel caricamento del modello da {ckpt_path}: {e}")
            continue

        # Embedding set di training per clustering
        train_df = pd.read_excel(f"datasets/training_set/environmental_sentences_filtered_{dataset_version}.xlsx")
        train_df['Label'] = train_df['Environmental Sentences']
        sentences = train_df.Label.tolist()

        print(f"Computing embeddings for checkpoint: {file} ...")
        embeddings = model.encode(sentences, show_progress_bar=True)

        yhat, centroids = apply_clustering(args=args, embeddings=embeddings)

        n_clusters, *_ = get_basic_stat_clustering(yhat)

        if n_clusters > 1:
            # Calcola embeddings per le frasi di test (già ordinate)
            test_sentences = df["sentence"].tolist()
            test_embeddings = model.encode(test_sentences, show_progress_bar=True, convert_to_numpy=True)

            # Calcola le distanze di ciascun embedding da tutti i centroidi
            compute_distance = get_metric(args.metric)
            distance_matrix = np.array([
                [compute_distance(embedding, centroid) for centroid in centroids]
                for embedding in test_embeddings
            ])

            # Distanza minima per ciascuna frase
            min_dists = np.min(distance_matrix, axis=1)

            # Aggiungi al DataFrame principale
            excel_df[file] = min_dists

    # Aggiunge l'ultima colonna: label
    excel_df["environment_info"] = df["environment_info"]

    # Salva l'excel finale
    excel_path = os.path.join(output_dir, f"{model_name}_{dataset_version}_ckp_distances.xlsx")
    excel_df.to_excel(excel_path, index=False)

    print(f"Salvato: {excel_path}")
    return excel_df


def compute_accuracy_by_checkpoint(args, output_dir="checkpoints_eval"):
    compute_distance = get_metric(args.metric)

    checkpoints_dir = args.model
    model_name = args.model.replace("/", "_")
    dataset_version = args.dataset

    os.makedirs(output_dir, exist_ok=True)

    label_df = get_label_df()
    print(f"{len(label_df)} test labels read")

    categories = sorted(label_df["category"].unique())
    excel_df = pd.DataFrame({"category": categories})

    # Step 2: Caricamento dei checkpoint e calcolo delle distanze
    for file in sorted(os.listdir(checkpoints_dir), key=extract_ckpt_number):
        ckpt_path = os.path.join(checkpoints_dir, file)
        if not os.path.isdir(ckpt_path):
            continue

        try:
            model = SentenceTransformer(ckpt_path)
        except Exception as e:
            print(f"Errore nel caricamento del modello da {ckpt_path}: {e}")
            continue

        # Embedding set di training per clustering
        train_df = pd.read_excel(f"datasets/training_set/environmental_sentences_filtered_{dataset_version}.xlsx")
        train_df['Label'] = train_df['Environmental Sentences']
        sentences = train_df.Label.tolist()

        print(f"Computing embeddings for checkpoint: {file} ...")
        embeddings = model.encode(sentences, show_progress_bar=True)

        yhat, centroids = apply_clustering(args=args, embeddings=embeddings)

        n_clusters, *_ = get_basic_stat_clustering(yhat)

        if n_clusters > 1:
            threshold = -get_best_threshold(args, model, centroids, label_df)
            print(f"Best threshold: {threshold}")

            accuracies = []
            categories = sorted(label_df["category"].unique())
            for cat in categories:
                cat_df = label_df.loc[label_df["category"] == cat].copy()
                cat_embeddings = model.encode(cat_df["sentence"].tolist())  # Calcola gli embeddings con model.encode()
                cat_labels = np.array(cat_df["environment_info"].tolist())

                model_labels = []
                for embedding in cat_embeddings:
                    # Calcola la distanza tra l'embedding della frase e ogni centroide
                    distances = [compute_distance(embedding, centroid) for centroid in centroids]
                    label = 1 if min(distances) < threshold else 0  # Assegna l'etichetta sulla base della soglia
                    model_labels.append(label)

                # Calcolo dell'accuratezza per questa categoria
                acc = accuracy_score(cat_labels, model_labels)
                accuracies.append(acc)

            # Aggiungi al DataFrame principale
            excel_df[file] = accuracies

    # Salva l'excel finale
    excel_path = os.path.join(output_dir, f"{model_name}_{dataset_version}_ckp_accuracies.xlsx")
    excel_df.to_excel(excel_path, index=False)

    print(f"Salvato: {excel_path}")
    return excel_df

def compute_correlation_by_checkpoint(args, scores_df, output_dir="checkpoints_eval"):

    compute_distance = get_metric(args.metric)

    checkpoints_dir = args.model
    model_name = args.model.replace("/", "_")
    dataset_version = args.dataset
    knn = args.knn

    os.makedirs(output_dir, exist_ok=True)

    test_df = get_test_df()
    print(f"{len(test_df)} test labels read")

    categories = sorted(test_df["Category"].unique())
    excel_df = pd.DataFrame({"category": categories})

    # Step 2: Caricamento dei checkpoint e calcolo delle distanze
    for file in sorted(os.listdir(checkpoints_dir), key=extract_ckpt_number):
        ckpt_path = os.path.join(checkpoints_dir, file)
        if not os.path.isdir(ckpt_path):
            continue

        try:
            model = SentenceTransformer(ckpt_path)
        except Exception as e:
            print(f"Errore nel caricamento del modello da {ckpt_path}: {e}")
            continue

        # Embedding set di training per clustering
        train_df = pd.read_excel(f"datasets/training_set/environmental_sentences_filtered_{dataset_version}.xlsx")
        train_df['Label'] = train_df['Environmental Sentences']
        sentences = train_df.Label.tolist()

        print(f"Computing embeddings for checkpoint: {file} ...")
        embeddings = model.encode(sentences, show_progress_bar=True)

        yhat, centroids = apply_clustering(args=args, embeddings=embeddings)

        n_clusters, *_ = get_basic_stat_clustering(yhat)

        if n_clusters > 1:
            label_df = get_label_df()
            threshold = -get_best_threshold(args, model, centroids, label_df)
            print(f"Best threshold: {threshold:.4f}")

            print(f"Computing {model_name} embeddings for test_set...")
            embedded_sentences = {}
            for row in test_df.itertuples(index=False):
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

            cat_correlations = []
            for cat in categories:
                cat_df = test_df.loc[test_df['Category'] == cat].copy()
                cat_df.sort_values(by="Name", inplace=True)

                model_cat_scores = []
                # Itera su ciascuna etichetta in `etichette`
                for product in cat_df.Name.tolist():
                    partial_scores = []
                    for _, emb_sentence in embedded_sentences[product]:
                        sentence_score = []
                        # Calcola la distanza tra l'embedding della frase e ogni centroide
                        distances = [compute_distance(emb_sentence, centroid) for centroid in centroids]
                        knn_distances = sorted(distances)[:knn]
                        # Calcola lo score parziale della frase
                        sentence_score = [max(0, threshold - distance) for k, distance in
                                          enumerate(knn_distances)]

                        partial_scores.append(sentence_score)
                    partial_scores = np.array(partial_scores)
                    punteggio_totale = get_label_score(args, partial_scores)

                    # Aggiunge la riga per l'etichetta
                    model_cat_scores.append(punteggio_totale)

                my_cat_scores_df = scores_df.loc[scores_df['Category'] == cat].copy()
                my_cat_scores_df.sort_values(by="Name", inplace=True)
                my_cat_scores = my_cat_scores_df.Norm_score.tolist()

                cat_correlation = np.corrcoef(model_cat_scores, my_cat_scores)[0, 1]
                cat_correlations.append(cat_correlation)

            # Aggiungi al DataFrame principale
            excel_df[file] = cat_correlations

    # Salva l'excel finale
    excel_path = os.path.join(output_dir, f"{model_name}_{dataset_version}_ckp_correlations.xlsx")
    excel_df.to_excel(excel_path, index=False)

    print(f"Salvato: {excel_path}")
    return excel_df


def main():
    parser = argparse.ArgumentParser() # Can be simplified
    parser = add_base_args(parser)

    args = parser.parse_args()

    if args.run == "ckp_distance": # Temp
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