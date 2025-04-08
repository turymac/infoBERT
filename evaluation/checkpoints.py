import os
import sys

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())

from utils.utils import get_test_df, get_basic_stat_clustering, get_label_df, get_metric, \
    get_label_score, get_best_threshold, extract_ckpt_number
from clustering.clustering import apply_clustering


def compute_distances_by_checkpoint(args, output_dir="checkpoints_eval"):
    model_name = args.model.replace("/", "_")
    alias = args.alias
    dataset_version = args.dataset

    os.makedirs(output_dir, exist_ok=True)

    # Legge direttamente l'excel con frasi già splittate
    df = pd.read_excel("datasets/test_set/test_sentences.xlsx")
    print(f"{len(df)} test sentences read")

    # Ordina le frasi mantenendo category e label associati
    df = df.sort_values(by="sentence").reset_index(drop=True)

    # Crea il DataFrame finale con le colonne sentence e category
    excel_df = df[["category", "sentence"]].copy()

    checkpoints_dir = f"models/{alias}"
    # Step 2: Caricamento dei checkpoint e calcolo delle distanze
    all_checkpoints = os.listdir(checkpoints_dir) + [args.model]
    for file in sorted(all_checkpoints, key=extract_ckpt_number):
        ckpt_path = os.path.join(checkpoints_dir, file)
        if not os.path.isdir(ckpt_path):
            ckpt_path = file

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
    excel_path = os.path.join(output_dir, f"{alias}_{model_name}_{dataset_version}_ckp_distances.xlsx")
    excel_df.to_excel(excel_path, index=False)

    print(f"Salvato: {excel_path}")
    return excel_df


def compute_accuracy_by_checkpoint(args, output_dir="checkpoints_eval"):
    compute_distance = get_metric(args.metric)

    alias = args.alias
    model_name = args.model.replace("/", "_")
    dataset_version = args.dataset

    os.makedirs(output_dir, exist_ok=True)

    label_df = get_label_df()
    print(f"{len(label_df)} test labels read")

    categories = sorted(label_df["category"].unique())
    excel_df = pd.DataFrame({"category": categories})

    checkpoints_dir = f"models/{alias}"
    # Step 2: Caricamento dei checkpoint e calcolo delle distanze
    all_checkpoints = os.listdir(checkpoints_dir) + [args.model]
    for file in sorted(all_checkpoints, key=extract_ckpt_number):
        ckpt_path = os.path.join(checkpoints_dir, file)
        if not os.path.isdir(ckpt_path):
            ckpt_path = file

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
    excel_path = os.path.join(output_dir, f"{alias}_{model_name}_{dataset_version}_ckp_accuracies.xlsx")
    excel_df.to_excel(excel_path, index=False)

    print(f"Salvato: {excel_path}")
    return excel_df


def compute_correlation_by_checkpoint(args, scores_df, output_dir="checkpoints_eval"):
    compute_distance = get_metric(args.metric)

    alias = args.alias
    model_name = args.model.replace("/", "_")
    dataset_version = args.dataset
    knn = args.knn

    os.makedirs(output_dir, exist_ok=True)

    test_df = get_test_df()
    print(f"{len(test_df)} test labels read")

    categories = sorted(test_df["Category"].unique())
    excel_df = pd.DataFrame({"category": categories})

    checkpoints_dir = f"models/{alias}"
    # Step 2: Caricamento dei checkpoint e calcolo delle distanze
    all_checkpoints = os.listdir(checkpoints_dir) + [args.model]
    for file in sorted(all_checkpoints, key=extract_ckpt_number):
        ckpt_path = os.path.join(checkpoints_dir, file)
        if not os.path.isdir(ckpt_path):
            ckpt_path = file
            print(file)

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

            print(f"Computing {file} embeddings for test_set...")
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
    excel_path = os.path.join(output_dir, f"{alias}_{model_name}_{dataset_version}_ckp_correlations.xlsx")
    excel_df.to_excel(excel_path, index=False)

    print(f"Salvato: {excel_path}")
    return excel_df