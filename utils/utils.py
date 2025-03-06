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

def get_test_df(path="datasets/test_set/testset_eng_categories.xlsx"):
    return pd.read_excel(path)

def get_personal_scores_df(path="datasets/scores/personal_score.xlsx"):
    return pd.read_excel(path)