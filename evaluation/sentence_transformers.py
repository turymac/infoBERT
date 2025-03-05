import argparse

from base_args import add_base_args
from models.embeddings import compute_embeddings

def main():
    parser = argparse.ArgumentParser()
    args = add_base_args(parser)
    embeddings = compute_embeddings(dataset_version="v3", model_name=args.model)
    #reduced_embeddings = apply_pca(embeddings)

    #clustering_algorithms = args.clustering_alg.split(",") if args.clustering_alg != "all" else [
    #    "kmeans", "affinity", "meanshift", "spectral", "agglomerative", "dbscan", "optics", "birch", "bisecting_kmeans"]

    #for alg in clustering_algorithms:
        # print(f"Running clustering with {alg}...")
        # labels = apply_clustering(reduced_embeddings, alg)
        # score = silhouette_score(reduced_embeddings, labels)
        # print(f"Silhouette Score for {alg}: {score:.4f}")


if __name__ == "__main__":
    main()