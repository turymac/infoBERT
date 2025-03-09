import numpy as np
from google.colab import files
from numpy import unique, where, mean
from scipy.spatial import distance
from openpyxl import Workbook

from utils.utils import get_metric, get_label_score



def compute_correlation_personal_marks(args, embedded_sentences, centroids, test_df, scores_df, get_avg=False, verbose=False): #Make get_avg, verbose args parameters

  compute_distance = get_metric(args.metric)
  knn = args.knn
  thresholds = args.thresholds

  if knn != len(thresholds):
      raise ValueError('knn must be equal to len(thresholds)') # Work on 2 thresholds and parameter alpha

  cat_correlations = {}
  for cat in set(test_df.Category.tolist()):
    cat_df  = test_df.loc[test_df['Category'] == cat].copy()
    cat_df.sort_values(by="Name", inplace=True)

    model_cat_scores = []
      # Itera su ciascuna etichetta in `etichette`
    for product in cat_df.Name.tolist():
        partial_scores = []
        for _, emb_sentence in embedded_sentences[product]:

            # Calcola la distanza tra l'embedding della frase e ogni centroide
            distances = [compute_distance(emb_sentence, centroid) for centroid in centroids]
            knn_distances = sorted(distances)[:knn]
            # Calcola lo score parziale della frase
            for id, distance in knn_distances:
                if distance < thresholds[id]:
                    partial_scores.append(thresholds[id] - distance)
                else:
                    partial_scores.append(0)

        punteggio_totale = get_label_score(args, partial_scores)

        # Aggiunge la riga per l'etichetta
        model_cat_scores.append(punteggio_totale)

    my_cat_scores_df = scores_df.loc[scores_df['Category'] == cat].copy()
    my_cat_scores_df.sort_values(by="Name", inplace=True)
    my_cat_scores = my_cat_scores_df.Norm_score.tolist()

    cat_correlation = np.corrcoef(model_cat_scores, my_cat_scores)[0,1]
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
      cat_df  = test_df.loc[test_df['Category'] == cat].copy()
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
              ws.append(["","",original_sentence, f"{punteggio_frase:.3f}", "", f"{distanza_minima:.3f}"])

          # Aggiunge la riga per l'etichetta
          ws.append(["","","", punteggio_totale, "", ""])
          ws.append(["", "", "", "", "", ""])

    # Salva il workbook in un file .xlsx
    wb.save(f"{model_name}@{threshold}-{dataset_version}.xlsx")

    # Scarica il workbook
    files.download(f"{model_name}@{threshold}-{dataset_version}.xlsx")

def partial_distance_knn_to_excel(args, embedded_sentences, centroids, test_df):
    compute_distance = get_metric(args.metric)
    threshold = float(args.threshold)
    knn = args.knn

    model_name = args.model
    dataset_version = args.dataset

    # Crea un nuovo workbook e seleziona il foglio attivo
    wb = Workbook()
    ws = wb.active

    # Intestazioni
    headers = ["Name", "Category", "Frase"] + [f"Distanza Top {i+1}" for i in range(knn)]
    ws.append(headers)

    for cat in set(test_df.Category.tolist()):
        cat_df = test_df.loc[test_df['Category'] == cat].copy()
        cat_df.sort_values(by="Name", inplace=True)

        for product in cat_df.Name.tolist():
            ws.append([product, cat])

            for original_sentence, emb_sentence in embedded_sentences[product]:
                # Calcola la distanza tra l'embedding della frase e ogni centroide
                distances = [compute_distance(emb_sentence, centroid) for centroid in centroids]
                top_knn_distances = sorted(distances)[:knn]

                # Aggiunge una riga con la frase e le distanze rispetto ai centroidi più vicini
                ws.append(["", "", original_sentence] + [f"{d:.3f}" for d in top_knn_distances])

            ws.append(["", "", ""])  # Riga vuota per separazione

    # Salva il workbook in un file .xlsx
    wb.save(f"{model_name}@{threshold}-{dataset_version}.xlsx")
    # Scarica il workbook
    files.download(f"{model_name}@{threshold}-{dataset_version}.xlsx")

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

