import numpy as np
import json

from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS, Birch, BisectingKMeans

from utils.utils import get_basic_stat_clustering
from evaluation.correlation import compute_correlation_personal_marks

# Carica la configurazione dei clustering da JSON
with open("clustering/clustering_config.json", "r") as f:
    CLUSTERING_CONFIG = json.load(f)

def apply_clustering(args, embeddings):
    """
      Args:
          bandwidth : float = None - Bandwidth used in the flat kernel. If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
    """
    clustering_alg = args.clustering.lower()

    if clustering_alg not in CLUSTERING_CONFIG:
        raise ValueError(f"Clustering algorithm '{clustering_alg}' is not supported.")

    # Ottiene i parametri predefiniti dal file JSON
    clustering_params = CLUSTERING_CONFIG[clustering_alg]

    # Seleziona l'algoritmo corretto
    if clustering_alg == "meanshift":
        clustering_model = MeanShift(**clustering_params)

    yhat = clustering_model.fit_predict(embeddings)

    return yhat