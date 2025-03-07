import numpy as np
import json

from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS, Birch, BisectingKMeans

from utils.utils import get_basic_stat_clustering
from evaluation.correlation import compute_correlation_personal_marks

# Carica la configurazione dei clustering da JSON
with open("clustering/clustering_config.json", "r") as f:
    CLUSTERING_CONFIG = json.load(f)

def apply_clustering(args, embeddings):
    clustering_alg = args.clustering.lower()

    if clustering_alg not in CLUSTERING_CONFIG:
        raise ValueError(f"Clustering algorithm '{clustering_alg}' is not supported.")

    clustering_params = CLUSTERING_CONFIG[clustering_alg]

    if clustering_alg == "kmeans":
        """
          Args:
            n_clusters : int = 8 - The number of clusters to form as well as the number of centroids to generate
            init : {
              'k-means++' - Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence
              'random' -  Choose n_clusters observations (rows) at random from data for the initial centroids
            } = 'k-means++' - Method for initialization
            n_init : int = 10 - Number of time the k-means algorithm will be run with different centroid seeds in each run
        """
        clustering_model = KMeans(**clustering_params)

    elif clustering_alg == "affinitypropagation":
        """
          Args:
              damping : float[0.5, 1.0) = 0.5 - Extent to which the current value is maintained relative to incoming values (weighted 1 - damping)
              affinity : {
                'euclidean' - Use the negative squared euclidean distance between points
                'precomputed' -
              } = 'euclidean' - How affinity matrix is computed
        """
        clustering_model = AffinityPropagation(**clustering_params)

    elif clustering_alg == "meanshift":
        """
          Args:
              bandwidth : float = None - Bandwidth used in the flat kernel. If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth
        """
        clustering_model = MeanShift(**clustering_params)

    elif clustering_alg == "spectralclustering":
        """
        Args:
          n_clusters : int = 8
          n_components : int = None - Number of eigenvectors to use for the spectral embedding
          gamma : float = 1.0 - Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels. Ignored for affinity='nearest_neighbors', affinity='precomputed' or affinity='precomputed_nearest_neighbors'.
          affinity : {
            'nearest_neighbors' - Construct the affinity matrix by computing a graph of nearest neighbors
            'rbf', - Construct the affinity matrix using a radial basis function (RBF) kernel
            'precomputed' - Interpret X as a precomputed affinity matrix, where larger values indicate greater similarity between instances
            'precomputed_nearest_neighbors' - Interpret X as a sparse graph of precomputed distances, and construct a binary affinity matrix from the n_neighbors nearest neighbors of each instance
          } = 'rbf' - How to construct the affinity matrix
          n_neighbors : int = 10 - Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method. Ignored for affinity='rbf'
          assign_labels : {
            'kmeans' - Assign labels using the k-means algorithm, can be sensitive to random initialization
            'discretize' - Less sensitive to random initialization
            'cluster_qr' - Directly extract clusters from eigenvectors in spectral clustering
          } = 'kmeans' - The strategy for assigning labels in the embedding space
        """
        clustering_model = SpectralClustering(**clustering_params)

    elif clustering_alg == "agglomerativeclustering":
        """
          Args:
              n_clusters : {
                int -
                None -
              } = 2 - Number of clusters to find. It must be None if distance_threshold is not None
              metric : {
                'euclidean' - If linkage is 'ward', only 'euclidean' is accepted
                'l1' -
                'l2' -
                'manhattan' -
                'cosine' -
                'precomputed' - Must input a distance matrix for the fit method
              } - Metric used to compute the linkage
              linkage : {
                'ward' - Minimizes the variance of the clusters being merged
                'complete' - Uses the maximum distances between all observations of the two sets
                'average' - Uses the average of the distances of each observation of the two sets
                'single' - Uses the minimum of the distances between all observations of the two sets
              } = 'ward' - Which linkage criterion to use
              distance_threshold : float = None - The linkage distance threshold above which, clusters will not be merged
        """
        clustering_model = AgglomerativeClustering(**clustering_params)

    elif clustering_alg == "dbscan":
        """
          Args:
              eps : float = 0.5 - Maximum distance between two samples for one to be considered as in the neighborhood of the other
              min_samples : int = 5 - Minimum number of samples in a neighborhood for a point to be considered as a core point
              metric : {'euclidean', 'precomputed'} = 'euclidean' - Metric used to calculate distance between instances
        """
        clustering_model = DBSCAN(**clustering_params)

    elif clustering_alg == "hdbscan":
        """
        Args:
            min_cluster_size : int = 5
            cluster_selection_epsilon : float = 0.0 - Distance threshold. Clusters below this value will be merged
            metric : str = 'euclidean'
            alpha : float = 1.0 - Distance scaling parameter as used in robust single linkage
            cluster_selection_method : {"eom", "leaf"} = "eom"
        """
        clustering_model = HDBSCAN(**clustering_params)

    elif clustering_alg == "optics":
        """
          Args:
            min_samples : int = 5 - Minimum number of samples in a neighborhood for a point to be considered as a core point
            max_eps : float = +inf - The maximum distance between two samples for one to be considered as in the neighborhood of the other
            metric : {'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'minkowski'} = 'minkowski' - Metric to use for distance computation
            p : int = 2 - Parameter for the Minkowski metric
            cluster_method : {"xi", "dbscan"} = "xi" - The extraction method used to extract clusters using the calculated reachability and ordering
            xi : float(0,1) - Determines the minimum steepness on the reachability plot that constitutes a cluster boundary. For example, an upwards point in the reachability plot is defined by the ratio from one point to its successor being at most 1-xi. Used only when cluster_method='xi'
        """
        clustering_model = OPTICS(**clustering_params)

    elif clustering_alg == "birch":
        """
          Args:
              threshold : float = 0.5 - The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Setting this value to be very low promotes splitting and vice-versa
              branching_factor : int = 50 - Maximum number of CF subclusters in each node. If a new samples enters such that the number of subclusters exceed the branching_factor then that node is split into two nodes with the subclusters redistributed in each. The parent subcluster of that node is removed and two new subclusters are added as parents of the 2 split nodes
              n_clusters : {
                None - the final clustering step is not performed and the subclusters are returned as they are
                sklearn.cluster Estimator - If a model is provided, the model is fit treating the subclusters as new samples and the initial data is mapped to the label of the closest subcluste
                int - the model fit is AgglomerativeClustering with n_clusters set to be equal to the int
              }
        """
        clustering_model = Birch(**clustering_params)

    elif clustering_alg == "bisectingkmeans":
        """
          Args:
              n_clusters : int = 8 - The number of clusters to form as well as the number of centroids to generate
              init : {
                'k-means++' - Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence
                'random' -  Choose n_clusters observations (rows) at random from data for the initial centroids
              } = 'random' - Method for initialization
              n_init : int = 1 - Number of time the inner k-means algorithm will be run with different centroid seeds in each bisection
              bisecting_strategy : {
                'biggest_inertia' - BisectingKMeans will always check all calculated cluster for cluster with biggest SSE (Sum of squared errors) and bisect it. This approach concentrates on precision, but may be costly in terms of execution time
                'largest_cluster' - BisectingKMeans will always split cluster with largest amount of points assigned to it from all clusters previously calculated. That should work faster than picking by SSE
              } - Defines how bisection should be performed
        """
        clustering_model = BisectingKMeans(**clustering_params)

    else:
        raise ValueError(f"There exists a configuration for '{clustering_alg}' but it's not actually implemented.")

    return clustering_model.fit_predict(embeddings)