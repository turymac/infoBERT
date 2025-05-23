import numpy as np
import pandas as pd
import re

from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, roc_curve, auc, silhouette_score
from sklearn.metrics.pairwise import cosine_distances

def apply_pca(embeddings):
    pca = PCA(n_components=0.95)  # Keep 95% variance
    return pca.fit_transform(embeddings)

def apply_filtering(embeddings, filter_eps):

    # Calcola la matrice delle distanze coseno
    dist_matrix = cosine_distances(embeddings)

    # Raggruppa frasi con distanza coseno inferiore a filter_eps
    n = embeddings.shape[0]
    groups = []
    visited = set()

    for i in range(n):
        if i in visited:
            continue
        group = {i}
        for j in range(i + 1, n):
            if dist_matrix[i, j] < filter_eps:
                group.add(j)
                visited.add(j)
        visited.add(i)
        groups.append(group)

    # Seleziona la frase più lunga (più parole) per ciascun gruppo
    selected_indices = []
    for group in groups:
        if len(group) == 1:
            selected_indices.append(group.pop())
        else:
            #longest_sentence_idx = max(group, key=lambda idx: len(sentences[idx].split())) No df available
            selected_indices.append(min(group))

    print(f"{n-len(selected_indices)} sentences filtered out")

    return embeddings[selected_indices]

def get_basic_stat_clustering(yhat, embeddings):
  labels_for_cluster = [np.where(yhat == cluster_id)[0] for cluster_id in np.unique(yhat)]
  cluster_sizes = [len(indices) for indices in labels_for_cluster]

  stats = {}

  stats["n_clusters"] = len(np.unique(yhat))
  stats["min_cluster_size"] = np.min(cluster_sizes)
  stats["mean_cluster_size"] = np.mean(cluster_sizes)
  stats["max_cluster_size"] = np.max(cluster_sizes)
  stats["silhouette_score"] = silhouette_score(embeddings, yhat)
  return stats

# Stampa le frasi raggruppate per cluster
def visualize_clusters(sentences, yhat):
  labels_for_cluster = [np.where(yhat == cluster_id)[0] for cluster_id in np.unique(yhat)]
  for cluster_id, indices in enumerate(labels_for_cluster):
    print(f"Cluster {cluster_id}\n")
    for idx in indices:
        print(f"{sentences[idx]}")
    print("\n" + "-" * 50 + "\n")  # Separatore tra cluster

def get_metric(metric):
    if metric == "cosine":
        compute_distance = distance.cosine
    elif metric == "euclidean":
        compute_distance = distance.euclidean
    elif metric == "cityblock":
        compute_distance = distance.cityblock
    elif metric == "chebyshev":
        compute_distance = distance.chebyshev
    else:
        raise "Metric not defined or not supported"

    return compute_distance

def get_best_threshold(args, model, centroids, label_df):
    compute_distance = get_metric(args.metric)

    tot_labels = []
    tot_scores = []
    categories = sorted(label_df["category"].unique())
    for cat in categories:
        cat_df = label_df.loc[label_df["category"] == cat].copy()
        cat_embeddings = model.encode(cat_df["sentence"].tolist())  # Calcola gli embeddings con model.encode()
        cat_labels = np.array(cat_df["environment_info"].tolist())
        tot_labels.append(cat_labels)

        cat_scores = []
        for embedding in cat_embeddings:
            # Calcola la distanza tra l'embedding della frase e ogni centroide
            distances = [compute_distance(embedding, centroid) for centroid in centroids]
            cat_scores.append(-min(distances))

        tot_scores.append(np.array(cat_scores))

    fpr, tpr, thresholds = roc_curve( np.concatenate(tot_labels), np.concatenate(tot_scores))

    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]

# Auxiliary function to compute the final label score from a matrix
def get_label_score(args, scores_matrix):
    score = args.aggr_score

    if score == 'sum':
        row_scores = np.sum(scores_matrix, axis=1)
    elif score == 'sqrt_sum':
        row_scores = np.sqrt(np.sum(scores_matrix**2, axis=1))
    elif score == 'inverse':
        row_scores = np.sum(1 / (1 + scores_matrix), axis=1)
    elif score == 'gaussian':
        sigma = np.mean(scores_matrix, axis=1, keepdims=True) / 2
        row_scores = np.sum(np.exp(-(scores_matrix**2) / (2 * sigma**2)), axis=1)
    else:
        raise ValueError(f"Method '{score}' not recognized.")

    # Aggregate all row scores with a simple sum
    return np.sum(row_scores)

def get_test_df(path="datasets/test_set/testset_eng_categories.xlsx"):
    return pd.read_excel(path)

def get_personal_scores_df(path="datasets/scores/personal_score.xlsx"):
    return pd.read_excel(path)

def get_label_df(path="datasets/test_set/test_sentences.xlsx"):
    return pd.read_excel(path)

def extract_ckpt_number(name):
    match = re.search(r"-(\d+)", name)
    return int(match.group(1)) if match else -1