import numpy as np

from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS, Birch, BisectingKMeans

from utils.utils import get_basic_stat_clustering
from evaluation.correlation import compute_correlation_personal_marks

def apply_clustering(args, embeddings):
    """
      Args:
          bandwidth : float = None - Bandwidth used in the flat kernel. If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
    """

    # Intervallo dei valori di bandwidth
    bandwidth_values = np.arange(0.1, 10, 0.1)

    for bandwidth in [2.4]:  # bandwidth_values:
        clustering_model = MeanShift(bandwidth=bandwidth)
        yhat = clustering_model.fit_predict(embeddings)

        return yhat