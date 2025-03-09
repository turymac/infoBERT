import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.spatial import distance

def apply_pca(embeddings):
    pca = PCA(n_components=0.95)  # Keep 95% variance
    return pca.fit_transform(embeddings)

def get_basic_stat_clustering(yhat):
  labels_for_cluster = [np.where(yhat == cluster_id)[0] for cluster_id in np.unique(yhat)]
  cluster_sizes = [len(indices) for indices in labels_for_cluster]
  n_clusters = len(np.unique(yhat))
  min_cluster_size = np.min(cluster_sizes)
  mean_cluster_size = np.mean(cluster_sizes)
  max_cluster_size = np.max(cluster_sizes)
  return n_clusters, min_cluster_size, mean_cluster_size, max_cluster_size

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

# Funzione ausiliaria per calcolare lo score finale della label
def get_label_score(args, scores_list):

    score = args.score

    if score == 'sum':
        return np.sum(scores_list)
    if score == 'sqrt_sum':
        return np.sqrt(np.sum([s**2 for s in scores_list]))
    elif score == 'inverse':
        return np.sum([1/(1 + d) for d in scores_list])
    elif score == 'gaussian':
        sigma = np.mean(scores_list) / 2
        return np.sum([np.exp(-(d**2)/(2 * sigma**2)) for d in scores_list])
    else:
        raise ValueError(f"Metodo '{score}' non riconosciuto.")

def get_test_df(path="datasets/test_set/testset_eng_categories.xlsx"):
    return pd.read_excel(path)

def get_personal_scores_df(path="datasets/scores/personal_score.xlsx"):
    return pd.read_excel(path)