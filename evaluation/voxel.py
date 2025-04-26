import numpy as np
from scipy.special import logsumexp
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from openpyxl import Workbook

from utils.utils import get_metric, get_label_score


# ============================
# STEP 1 - Calcolo dei voxel
# ============================
def build_axis_aligned_voxels(embeddings, min_side=-np.inf, max_side=np.inf):
    embeddings = np.array(embeddings)
    n, d = embeddings.shape
    voxel_bounds = []

    for i in range(n):
        center = embeddings[i]
        half_widths = np.zeros(d)

        for k in range(d):
            # inizializza con un valore alto, per cercare la minima distanza su quell'asse
            min_diff = float("inf")

            for j in range(n):
                if i == j:
                    continue
                diff = abs(embeddings[j, k] - center[k])
                if diff < min_diff:
                    min_diff = diff

            # imposto il lato con min/max bounds
            clamped_half_width = 0.5 * np.clip(min_diff, min_side, max_side)
            half_widths[k] = clamped_half_width

        lower = center - half_widths
        upper = center + half_widths
        voxel_bounds.append((lower, upper))

    return voxel_bounds

def rbf_weights(X, C, gamma=1e-2): #Make gamma a parameter in args
    distances = euclidean_distances(X, C)
    similarities = np.exp(-gamma * distances**2)
    weights = similarities.sum(axis=1)
    return weights

# # ============================
# # STEP 2 - Clustering
# # ============================
# def cluster_embeddings(embeddings):
#     clustering = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete", distance_threshold=0.5)
#     labels = clustering.fit_predict(embeddings)
#     return labels


# ============================
# STEP 3 - Calcolo del volume
# ============================

def compute_voxel_log_volume(lower, upper, min_side=1e-10):
    diff = np.maximum(np.array(upper) - np.array(lower), min_side)
    return np.sum(np.log(diff))


def compute_cluster_weights_logsafe(voxel_bounds, labels):
    cluster_log_volumes = defaultdict(list)

    for i, (lower, upper) in enumerate(voxel_bounds):
        log_vol = compute_voxel_log_volume(lower, upper)
        cluster = labels[i]
        cluster_log_volumes[cluster].append(log_vol)

    # Normalizzazione log-safe (log-softmax)
    clusters = list(cluster_log_volumes.keys())
    log_vol_array = np.array([logsumexp(cluster_log_volumes[c]) for c in clusters])
    log_total = logsumexp(log_vol_array)

    cluster_weights = {
        cluster: logv - log_total
        for cluster, logv in zip(clusters, log_vol_array)
    }

    return cluster_log_volumes, cluster_weights


# ============================
# STEP 4 - Matching test embedding
# ============================
def is_inside_voxel(point, lower, upper, tolerance=0):
    """
    Verifica se il punto cade nel voxel su almeno min_percentage delle dimensioni.
    """
    point = np.array(point)
    lower = np.array(lower)
    upper = np.array(upper)

    # Condizione di "appartenenza" dimensione per dimensione
    inside = (point >= (lower - tolerance)) & (point <= (upper + tolerance))

    # Percentuale di dimensioni rispettate
    percentage_inside = np.sum(inside) / len(inside)

    return percentage_inside


def match_test_embedding(test_point, voxel_bounds, labels, cluster_weights, tolerance=0, min_percentage=0.9):
    """
    Trova il voxel in cui cade il test_point rispettando la tolleranza su almeno min_percentage delle dimensioni.
    """

    for i, (lower, upper) in enumerate(voxel_bounds):
        percentage = is_inside_voxel(test_point, lower, upper, tolerance=tolerance)
        if percentage >= min_percentage:
            cluster = labels[i]
            weight_log = cluster_weights.get(cluster, 0.0)  # valore logaritmico
            weight = np.exp(weight_log)  # torna al peso normale
            return {
                "matched_voxel_index": i,
                "cluster": cluster,
                "cluster_weight": weight
            }
            # print(f"> {percentage:.2f} Inside voxel: \"{sentences[i]}\"")

    return {
        "matched_voxel_index": None,
        "cluster": None,
        "cluster_weight": 0.0
    }
    # return

def compute_information_quantity_voxel_personal(args, embedded_sentences, embeddings, centroids, yhat, test_df, scores_df, verbose=True):
    compute_distance = get_metric(args.metric)
    knn = args.knn

    if knn != 1:
        raise ValueError("Questa versione supporta solo knn = 1 (una sola threshold)")

    categories = sorted(test_df["Category"].unique())
    category_to_corrs = {cat: [] for cat in categories}
    mean_corrs = []

    # Costruzione voxel
    voxel_bounds = build_axis_aligned_voxels(embeddings)
    cluster_volumes, cluster_weights_log = compute_cluster_weights_logsafe(voxel_bounds, yhat)

    # Iniziamo ad analizzare le categorie
    corr_across_categories = []

    weights = rbf_weights(embeddings, centroids)

    for cat in categories:
        cat_df = test_df.loc[test_df['Category'] == cat].copy()
        cat_df.sort_values(by="Name", inplace=True)

        model_cat_scores = []

        for product in cat_df.Name.tolist():
            partial_scores = []

            for _, emb_sentence in embedded_sentences[product]:
                # Controllo appartenenza ai voxel
                match = match_test_embedding(
                    test_point=emb_sentence,
                    voxel_bounds=voxel_bounds,
                    labels=yhat,
                    cluster_weights=cluster_weights_log,
                    tolerance=0,
                    min_percentage=0.01
                )

                if match["matched_voxel_index"] is not None:
                    # C'è stato un match
                    matched_embedding = embeddings[match["matched_voxel_index"]]
                    cluster_weight = weights[match["cluster"]]
                    distance = compute_distance(emb_sentence, matched_embedding)
                    sentence_score = cluster_weight * (1/distance)
                else:
                    # Se nessun match, score nullo
                    sentence_score = 0.0

                partial_scores.append([sentence_score])

            partial_scores = np.array(partial_scores)
            punteggio_totale = get_label_score(args, partial_scores)
            model_cat_scores.append(punteggio_totale)

        # Confronto con i veri punteggi
        my_cat_scores_df = scores_df.loc[scores_df['Category'] == cat].copy()
        my_cat_scores_df.sort_values(by="Name", inplace=True)
        my_cat_scores = my_cat_scores_df.Norm_score.tolist()

        cat_corr = np.corrcoef(model_cat_scores, my_cat_scores)[0, 1]
        category_to_corrs[cat].append(cat_corr)
        corr_across_categories.append(cat_corr)

        if verbose:
            print(f"{cat} → Correlation: {cat_corr:.3f}")

    mean_corr = np.mean(corr_across_categories)
    mean_corrs.append(mean_corr)

    return category_to_corrs, mean_corrs

def compute_information_quantity_voxel_form(args, embedded_sentences, embeddings, test_df, verbose=True):
    pass

#EXCEL
def compute_information_quantity_voxel_personal_excel(args, embedded_sentences, embeddings, centroids, yhat, test_df, scores_df, verbose=True):
    compute_distance = get_metric(args.metric)
    knn = args.knn

    if knn != 1:
        raise ValueError("Questa versione supporta solo knn = 1 (una sola threshold)")

    categories = sorted(test_df["Category"].unique())
    category_to_corrs = {cat: [] for cat in categories}
    mean_corrs = []


    weights = rbf_weights(embeddings, centroids)

    # Costruzione voxel
    voxel_bounds = build_axis_aligned_voxels(embeddings)
    cluster_volumes, cluster_weights_log = compute_cluster_weights_logsafe(voxel_bounds, yhat)

    # ====== Aggiunta Excel Workbook ======
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Category", "Frase", "Score Parziale", "", "Distanza dal Centroide"])

    # Iniziamo ad analizzare le categorie
    corr_across_categories = []

    for cat in categories:
        cat_df = test_df.loc[test_df['Category'] == cat].copy()
        cat_df.sort_values(by="Name", inplace=True)

        model_cat_scores = []

        for product in cat_df.Name.tolist():
            partial_scores = []

            # Aggiunge una riga di intestazione per il prodotto
            ws.append([product, cat])

            for original_sentence, emb_sentence in embedded_sentences[product]:
                # Controllo appartenenza ai voxel
                match = match_test_embedding(
                    test_point=emb_sentence,
                    voxel_bounds=voxel_bounds,
                    labels=yhat,
                    cluster_weights=cluster_weights_log,
                    tolerance=0.0,
                    min_percentage=0.01
                )

                if match["matched_voxel_index"] is not None:
                    matched_embedding = embeddings[match["matched_voxel_index"]]
                    cluster_weight = weights[match["cluster"]]
                    distance = compute_distance(emb_sentence, matched_embedding)
                    sentence_score = cluster_weight * (1 / distance)
                else:
                    sentence_score = 0.0
                    distance = float('inf')  # per chiarezza

                partial_scores.append([sentence_score])

                # Scrivi la frase e lo score parziale nel file Excel
                ws.append(["", "", original_sentence, f"{sentence_score:.6f}", "", f"{distance:.6f}"])

            partial_scores = np.array(partial_scores)
            punteggio_totale = get_label_score(args, partial_scores)
            model_cat_scores.append(punteggio_totale)

            # Riga finale con il punteggio totale del prodotto
            ws.append(["", "", "", f"Totale: {punteggio_totale:.6f}", "", ""])
            ws.append(["", "", "", "", "", ""])

        # Confronto con i veri punteggi
        my_cat_scores_df = scores_df.loc[scores_df['Category'] == cat].copy()
        my_cat_scores_df.sort_values(by="Name", inplace=True)
        my_cat_scores = my_cat_scores_df.Norm_score.tolist()

        cat_corr = np.corrcoef(model_cat_scores, my_cat_scores)[0, 1]
        category_to_corrs[cat].append(cat_corr)
        corr_across_categories.append(cat_corr)

        if verbose:
            print(f"{cat} → Correlation: {cat_corr:.3f}")

    mean_corr = np.mean(corr_across_categories)
    mean_corrs.append(mean_corr)

    # ====== Salvataggio Excel ======
    model_name = args.model
    dataset_version = args.dataset

    filename = f"{model_name}_voxel_scores-{dataset_version}.xlsx"
    wb.save(filename)
    print(f"Risultati salvati in '{filename}'.")

    return category_to_corrs, mean_corrs