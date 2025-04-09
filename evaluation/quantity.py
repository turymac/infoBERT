import json
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances

from utils.utils import get_label_score, get_metric

# load configuration from JSON
with open("evaluation/quantity_config.json", "r") as f:
    QUANTIFY_CONFIG = json.load(f)

def rbf_weights(X, C, gamma): #Make gamma a parameter in args
    distances = cosine_distances(X, C)
    similarities = np.exp(-gamma * distances**2)
    weights = similarities.sum(axis=1)
    return weights

def compute_information_quantity_personal(args, embedded_sentences, embeddings, centroids, test_df, scores_df, get_avg=False, verbose=False):
    compute_distance = get_metric(args.metric)
    knn = args.knn
    thresholds = args.thresholds

    if knn != len(thresholds):
        raise ValueError('knn must be equal to len(thresholds)')  # Work on 2 thresholds and parameter alpha

    gamma = QUANTIFY_CONFIG["rbf_gamma"]
    weights = rbf_weights(embeddings, centroids, gamma)

    cat_correlations = {}
    categories = sorted(test_df["Category"].unique())
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
                distances = [(compute_distance(emb_sentence, emb), weight) for emb, weight in zip(embeddings, weights)]
                knn_distances = sorted(distances)[:knn]
                # Calcola lo score parziale della frase
                sentence_score = [max(0, (thresholds[id] - distance) * weight) for id, (distance, weight) in enumerate(knn_distances)]

                partial_scores.append(sentence_score)
            partial_scores = np.array(partial_scores)
            punteggio_totale = get_label_score(args, partial_scores)

            # Aggiunge la riga per l'etichetta
            model_cat_scores.append(punteggio_totale)

        my_cat_scores_df = scores_df.loc[scores_df['Category'] == cat].copy()
        my_cat_scores_df.sort_values(by="Name", inplace=True)
        my_cat_scores = my_cat_scores_df.Norm_score.tolist()

        cat_correlation = np.corrcoef(model_cat_scores, my_cat_scores)[0, 1]
        if verbose == True:
            print(f"> {cat}: {cat_correlation:.3f}")
        cat_correlations[cat] = cat_correlation

    print(f" ----- Avg_correlation: {np.mean(list(cat_correlations.values())):.3f} -----")

    if get_avg == True:
        return np.mean(list(cat_correlations.values()))
    else:
        return cat_correlations

def compute_information_quantity_form(args, embedded_sentences, embeddings, centroids, test_df, verbose=False):  # Make get_avg, verbose args parameters

    compute_distance = get_metric(args.metric)
    knn = args.knn
    thresholds = args.thresholds

    if knn != len(thresholds):
        raise ValueError('knn must be equal to len(thresholds)')  # Work on 2 thresholds and parameter alpha

    gamma = QUANTIFY_CONFIG["rbf_gamma"]
    weights = rbf_weights(embeddings, centroids, gamma)

    categories = sorted(test_df["Category"].unique())
    for cat in categories:
        print(f">> {cat}:")

        cat_df = test_df.loc[test_df['Category'] == cat].copy()
        # cat_df.sort_values(by="Name", inplace=True)

        model_cat_scores = {}
        # Itera su ciascuna etichetta in `etichette`
        for product in cat_df.Name.tolist():
            partial_scores = []
            for _, emb_sentence in embedded_sentences[product]:
                sentence_score = []
                # Calcola la distanza tra l'embedding della frase e ogni centroide
                distances = [(compute_distance(emb_sentence, emb), weight) for emb, weight in zip(embeddings, weights)]
                knn_distances = sorted(distances)[:knn]
                # Calcola lo score parziale della frase
                sentence_score = [max(0, (thresholds[id] - distance) * weight) for id, (distance, weight) in enumerate(knn_distances)]

                partial_scores.append(sentence_score)
            partial_scores = np.array(partial_scores)
            punteggio_totale = get_label_score(args, partial_scores)

            # Aggiunge la riga per l'etichetta
            model_cat_scores[product] = punteggio_totale

        np_order = np.load(f"datasets/scores/np_order/{cat}.npy")
        ordered_model_scores = np.array([model_cat_scores[prod] for prod in np_order])
        np_scores = np.load(f"datasets/scores/np_scores/{cat}.npy")
        np_scores = np_scores[np.ptp(np_scores, axis=1) >= QUANTIFY_CONFIG["min_spread"]]

        if QUANTIFY_CONFIG["normalize"]:
            # min-max normalization
            row_min = np.min(np_scores, axis=1, keepdims=True)
            row_max = np.max(np_scores, axis=1, keepdims=True)
            np_scores = (np_scores - row_min) / (row_max - row_min)

        cat_correlation = [np.corrcoef(ordered_model_scores.flatten(), np_scores[row, :])[0, 1] for row in
                           range(np_scores.shape[0])]
        if verbose == True:
            for corr in cat_correlation:
                print(f">> {corr:.3f}")
        print()

        avg_score = np.mean(np_scores, axis=0)
        corr_avg_score = np.corrcoef(avg_score, ordered_model_scores)[0, 1]
        print(f">>> Correlation wrt averaged_score: {corr_avg_score:.3f}")

def compute_information_quantity_all_thr_form(args, embedded_sentences, embeddings, centroids, test_df, verbose=False):
    from copy import deepcopy

    compute_distance = get_metric(args.metric)
    knn = args.knn

    if knn != 1:
        raise ValueError("Questa versione supporta solo knn = 1 (una sola threshold)")

    gamma = QUANTIFY_CONFIG["rbf_gamma"]
    weights = rbf_weights(embeddings, centroids, gamma)
    thresholds_range = np.arange(0.3, 0.8 + 1e-6, 0.02)

    categories = sorted(test_df["Category"].unique())
    category_to_corrs = {cat: [] for cat in categories}
    mean_corrs = []

    for threshold in thresholds_range:
        thresholds = [threshold]
        corr_across_categories = []

        for cat in categories:
            cat_df = test_df.loc[test_df['Category'] == cat].copy()
            model_cat_scores = {}

            for product in cat_df.Name.tolist():
                partial_scores = []
                for _, emb_sentence in embedded_sentences[product]:
                    distances = [(compute_distance(emb_sentence, emb), weight) for emb, weight in zip(embeddings, weights)]
                    knn_distances = sorted(distances)[:knn]
                    sentence_score = [max(0, (thresholds[0] - distance) * weight) for distance, weight in knn_distances]
                    partial_scores.append(sentence_score)

                partial_scores = np.array(partial_scores)
                punteggio_totale = get_label_score(args, partial_scores)
                model_cat_scores[product] = punteggio_totale

            np_order = np.load(f"datasets/scores/np_order/{cat}.npy")
            ordered_model_scores = np.array([model_cat_scores[prod] for prod in np_order])
            np_scores = np.load(f"datasets/scores/np_scores/{cat}.npy")
            np_scores = np_scores[np.ptp(np_scores, axis=1) >= QUANTIFY_CONFIG["min_spread"]]

            if QUANTIFY_CONFIG["normalize"]:
                row_min = np.min(np_scores, axis=1, keepdims=True)
                row_max = np.max(np_scores, axis=1, keepdims=True)
                np_scores = (np_scores - row_min) / (row_max - row_min)

            avg_score = np.mean(np_scores, axis=0)
            corr_avg_score = np.corrcoef(avg_score, ordered_model_scores)[0, 1]
            category_to_corrs[cat].append(corr_avg_score)
            corr_across_categories.append(corr_avg_score)

        mean_corr = np.mean(corr_across_categories)
        mean_corrs.append(mean_corr)

        if verbose:
            print(f"Threshold: {threshold:.2f} -> Mean correlation w/ avg_score: {mean_corr:.3f}")

    # Plot
    plt.figure(figsize=(12, 7))
    for cat, corrs in category_to_corrs.items():
        plt.plot(thresholds_range, corrs, marker='o', label=f"{cat}")
    plt.plot(thresholds_range, mean_corrs, color='black', linewidth=2.5, linestyle='--', label="Media")

    plt.title("Correlazione vs Threshold\n(punteggi del modello vs media dei voti per categoria)")
    plt.xlabel("Threshold")
    plt.ylabel("Correlazione")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("correlation_quantity_vs_threshold_per_category.png")
    plt.show()

    return {
        "thresholds": thresholds_range,
        "mean": mean_corrs,
        "per_category": category_to_corrs
    }


def compute_information_quantity_all_thr_personal(args, embedded_sentences, embeddings, centroids, test_df, scores_df, verbose=False):
    compute_distance = get_metric(args.metric)
    knn = args.knn

    if knn != 1:
        raise ValueError("Questa versione supporta solo knn = 1 (una sola threshold)")

    gamma = QUANTIFY_CONFIG["rbf_gamma"]
    weights = rbf_weights(embeddings, centroids, gamma)

    thresholds_range = np.arange(0.3, 0.8 + 1e-6, 0.02)
    categories = sorted(test_df["Category"].unique())
    category_to_corrs = {cat: [] for cat in categories}
    mean_corrs = []

    for threshold in thresholds_range:
        thresholds = [threshold]
        corr_across_categories = []

        for cat in categories:
            cat_df = test_df.loc[test_df['Category'] == cat].copy()
            cat_df.sort_values(by="Name", inplace=True)

            model_cat_scores = []

            for product in cat_df.Name.tolist():
                partial_scores = []
                for _, emb_sentence in embedded_sentences[product]:
                    distances = [(compute_distance(emb_sentence, emb), weight) for emb, weight in zip(embeddings, weights)]
                    knn_distances = sorted(distances)[:knn]
                    sentence_score = [max(0, (thresholds[0] - distance) * weight) for distance, weight in knn_distances]
                    partial_scores.append(sentence_score)

                partial_scores = np.array(partial_scores)
                punteggio_totale = get_label_score(args, partial_scores)
                model_cat_scores.append(punteggio_totale)

            my_cat_scores_df = scores_df.loc[scores_df['Category'] == cat].copy()
            my_cat_scores_df.sort_values(by="Name", inplace=True)
            my_cat_scores = my_cat_scores_df.Norm_score.tolist()

            cat_corr = np.corrcoef(model_cat_scores, my_cat_scores)[0, 1]
            category_to_corrs[cat].append(cat_corr)
            corr_across_categories.append(cat_corr)

            if verbose:
                print(f"[{threshold:.2f}] {cat} → Correlation: {cat_corr:.3f}")

        mean_corr = np.mean(corr_across_categories)
        mean_corrs.append(mean_corr)

    # --- Plot ---
    plt.figure(figsize=(12, 7))
    for cat, corrs in category_to_corrs.items():
        plt.plot(thresholds_range, corrs, marker='o', label=f"{cat}")
    plt.plot(thresholds_range, mean_corrs, color='black', linewidth=2.5, linestyle='--', label="Media")

    plt.title("Correlazione vs Threshold\n(punteggi del modello vs voto soggettivo normalizzato)")
    plt.xlabel("Threshold")
    plt.ylabel("Correlazione")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("correlation_personal_vs_threshold_per_category.png")
    plt.show()

    return {
        "thresholds": thresholds_range,
        "mean": mean_corrs,
        "per_category": category_to_corrs
    }
