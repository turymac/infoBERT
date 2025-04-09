import numpy as np
import json

from openpyxl import Workbook

from utils.utils import get_metric, get_label_score

# Carica la configurazione dei clustering da JSON
with open("evaluation/correlation_config.json", "r") as f:
    CORRELATION_CONFIG = json.load(f)


def compute_correlation_form_marks(args, embedded_sentences, centroids, test_df,
                                   verbose=False):  # Make get_avg, verbose args parameters

    compute_distance = get_metric(args.metric)
    knn = args.knn
    thresholds = args.thresholds

    if knn != len(thresholds):
        raise ValueError('knn must be equal to len(thresholds)')  # Work on 2 thresholds and parameter alpha

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
                distances = [compute_distance(emb_sentence, centroid) for centroid in centroids]
                knn_distances = sorted(distances)[:knn]
                # Calcola lo score parziale della frase
                sentence_score = [max(0, thresholds[id] - distance) for id, distance in enumerate(knn_distances)]

                partial_scores.append(sentence_score)
            partial_scores = np.array(partial_scores)
            punteggio_totale = get_label_score(args, partial_scores)

            # Aggiunge la riga per l'etichetta
            model_cat_scores[product] = punteggio_totale

        np_order = np.load(f"datasets/scores/np_order/{cat}.npy")
        ordered_model_scores = np.array([model_cat_scores[prod] for prod in np_order])
        np_scores = np.load(f"datasets/scores/np_scores/{cat}.npy")
        np_scores = np_scores[np.ptp(np_scores, axis=1) >= CORRELATION_CONFIG["min_spread"]]

        if CORRELATION_CONFIG["normalize"]:
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


def compute_correlation_personal_marks(args, embedded_sentences, centroids, test_df, scores_df, get_avg=False,
                                       verbose=False):  # Make get_avg, verbose args parameters

    compute_distance = get_metric(args.metric)
    knn = args.knn
    thresholds = args.thresholds

    if knn != len(thresholds):
        raise ValueError('knn must be equal to len(thresholds)')  # Work on 2 thresholds and parameter alpha

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
                distances = [compute_distance(emb_sentence, centroid) for centroid in centroids]
                knn_distances = sorted(distances)[:knn]
                # Calcola lo score parziale della frase
                sentence_score = [max(0, thresholds[id] - distance) for id, distance in enumerate(knn_distances)]

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


def partial_score_to_excel(args, embedded_sentences, centroids, test_df):
    compute_distance = get_metric(args.metric)
    threshold = float(args.threshold)

    model_name = args.model
    dataset_version = args.dataset

    # Crea un nuovo workbook e seleziona il foglio attivo
    wb = Workbook()
    ws = wb.active

    # Intestazioni
    ws.append(["Name", "Category", "Frase", "Score Parziale", "", "Distanza dal Centroide"])

    cat_correlations = {}
    for cat in set(test_df.Category.tolist()):
        cat_df = test_df.loc[test_df['Category'] == cat].copy()
        cat_df.sort_values(by="Name", inplace=True)

        # Itera su ciascuna etichetta in `etichette`
        for product in cat_df.Name.tolist():

            # Aggiunge la riga per l'etichetta
            ws.append([product, cat])

            punteggio_totale = 0
            # Itera su ogni frase estratta
            for original_sentence, emb_sentence in embedded_sentences[product]:

                # Inizializza variabili per tenere traccia della distanza minima e del centroide più vicino
                distanza_minima = float('inf')
                indice_centroide_minimo = -1

                # Calcola la distanza tra l'embedding della frase e ogni centroide
                for i, centroid in enumerate(centroids):
                    distanza = compute_distance(emb_sentence, centroid)
                    if distanza < distanza_minima:
                        distanza_minima = distanza

                # Calcola lo score parziale della frase
                punteggio_frase = (threshold - distanza_minima) if distanza_minima < threshold else 0
                # * scores[indice_centroide_minimo]
                punteggio_totale += punteggio_frase

                # Aggiunge una riga per la frase con i dettagli richiesti
                ws.append(["", "", original_sentence, f"{punteggio_frase:.3f}", "", f"{distanza_minima:.3f}"])

            # Aggiunge la riga per l'etichetta
            ws.append(["", "", "", punteggio_totale, "", ""])
            ws.append(["", "", "", "", "", ""])

    # Salva il workbook in un file .xlsx
    wb.save(f"{model_name}@{threshold}-{dataset_version}.xlsx")

# def partial_prob_to_excel(embeddings, embedded_sentences, test_df, clustering_model):
#   # Crea un nuovo workbook e seleziona il foglio attivo
#     wb = Workbook()
#     ws = wb.active
#     #ws.title = f"{model_name}@{threshold}-{dataset_version}.xlsx"
#
#     # Intestazioni
#     ws.append(["Name", "Category", "Frase", "Sum(Top 5)"])
#
#     cat_correlations = {}
#     for cat in categories_set:
#       cat_df  = test_df.loc[test_df['Category'] == cat].copy()
#       cat_df.sort_values(by="Name", inplace=True)
#
#       # Itera su ciascuna etichetta in `etichette`
#       for product in cat_df.Name.tolist():
#
#           # Aggiunge la riga per l'etichetta
#           ws.append([product, cat])
#
#           # Itera su ogni frase estratta
#           for original_sentence, emb_sentence in embedded_sentences[product]:
#
#               distances = np.linalg.norm(clustering_model.cluster_centers_ - emb_sentence[0], axis=1)
#               prob_kmeans = distances
#               #prob_kmeans /= prob_kmeans.sum()  # Normalizzazione
#
#               # Ordina le probabilità
#               sorted_probs = np.sort(prob_kmeans)[::-1]  # Ordine decrescente
#               # Stampa delle prime 5 probabilità
#               top_5_prob = sorted_probs[:5]
#
#               # Aggiunge una riga per la frase con i dettagli richiesti
#               ws.append(["","",original_sentence, f"{top_5_prob.sum():.4f}", "", top_5_prob[0], top_5_prob[1], top_5_prob[2], top_5_prob[3], top_5_prob[4]])
#
#           ws.append([""])
#
#     # Salva il workbook in un file .xlsx
#     wb.save(f"{model_name}_MeanShift@{1.5}-{dataset_version}.xlsx")
#
#     # Scarica il workbook
#     files.download(f"{model_name}_MeanShift@{1.5}-{dataset_version}.xlsx")

