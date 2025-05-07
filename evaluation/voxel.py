import numpy as np
from scipy.special import logsumexp
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from openpyxl import Workbook
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils.utils import get_metric, get_label_score


# ============================
# STEP 1 - Calcolo dei voxel
# ============================
from sklearn.neighbors import NearestNeighbors
import numpy as np

def build_axis_aligned_voxels(embeddings, k=1, min_side=-np.inf, max_side=np.inf):
    embeddings = np.array(embeddings)
    n, d = embeddings.shape
    voxel_bounds = []

    # Costruisci il modello KNN (k+1 per includere il punto stesso)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    for i in range(n):
        center = embeddings[i]
        neighbors_idx = indices[i][1:]  # Escludi il punto stesso
        half_widths = np.zeros(d)

        for k_dim in range(d):
            min_diff = float("inf")
            for j in neighbors_idx:
                diff = abs(embeddings[j, k_dim] - center[k_dim])
                if diff < min_diff:
                    min_diff = diff

            clamped_half_width = 0.5 * np.clip(min_diff, min_side, max_side)
            half_widths[k_dim] = clamped_half_width

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
                    min_percentage=1
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


def plot_voxels(reduced_embeddings, voxel_bounds, labels=None, n_components=2):
    # Plotting
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.6, label='Embeddings')

        for (lower, upper) in voxel_bounds:
            width, height = upper - lower
            rect = plt.Rectangle(lower, width, height, fill=False, edgecolor='red', linewidth=1)
            plt.gca().add_patch(rect)

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Voxel Visualization with PCA (2D)')
        plt.grid(True)
        plt.legend()
        plt.show()

    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c='blue', alpha=0.6,
                   label='Embeddings')

        for (lower, upper) in voxel_bounds:
            r = [lower, upper]
            for s, e in zip(*np.array(np.meshgrid(*zip(lower, upper))).reshape(2, -1, 3)):
                ax.plot3D(*zip(s, e), color="red", linewidth=0.3)

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.set_title('Voxel Visualization with PCA (3D)')
        plt.legend()
        plt.show()

    else:
        raise ValueError("n_components must be either 2 or 3")


def pca_then_voxel_and_plot(embeddings, n_components=2, min_side=-np.inf, max_side=np.inf):
    """
    Applica PCA agli embeddings, calcola i voxel bounds sugli embeddings ridotti
    e visualizza il risultato.
    """
    embeddings = np.array(embeddings)

    # PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Calcola voxel bounds sui dati ridotti
    voxel_bounds = build_axis_aligned_voxels(reduced_embeddings, min_side=min_side, max_side=max_side)

    # Visualizzazione
    plot_voxels(reduced_embeddings, voxel_bounds, n_components=n_components)

    return reduced_embeddings, voxel_bounds

def main():
    embeddings = np.load("../datasets/training_set/precomputed_embeddings/v2_paraphrase-mpnet-base-v2_embeddings.npy")
    reduced_embeddings, voxel_bounds = pca_then_voxel_and_plot(embeddings[:50])


if __name__ == "__main__":
    main()