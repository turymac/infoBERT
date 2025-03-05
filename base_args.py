import argparse
import ast

def add_base_args(parser):
    parser.add_argument("--model", help="Model name to use for embedding.",
                        type=str, required=True)
    parser.add_argument("--clustering-alg", help="Clustering algorithm to test (can be a comma-separated list or 'all').",
                        type=str, required=True)
    parser.add_argument("--metric", help="Metric to measure the distance between embeddings.",
                        type=str, choices=["cosine", "euclidean", "cityblock", "chebyshev"], default="cosine")
    parser.add_argument("--threshold", help="Threshold to distinguish between informative and non-informative content.",
                        type=float, default=0.5)
    parser.add_argument("--knn", help="Number of nearest neighbors to consider for the final score.",
                        type=int, default=5)
    return parser