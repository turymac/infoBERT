def add_base_args(parser):
    parser.add_argument("--model", help="Model name to use for embedding.",
                        type=str, required=True)
    parser.add_argument("--alias", help="Alias to use for fine-tuned model.",
                        type=str)
    parser.add_argument("--marks", help="Marks to which compute correlation.",
                        type=str, choices=["personal", "form"], default="form")
    parser.add_argument("--dataset", help="Version of training dataset.",
                        type=str, required=True)
    parser.add_argument("--filtering", help="Remove from training dataset sentences whose embeddings are <= eps close.",
                        type=bool, default=False)
    # parser.add_argument("--filter_eps", help="Thresholds for filtering.",
    #                     type=float, default=0.05)
    parser.add_argument("--clustering", help="Clustering algorithm to test.",
                        type=str, required=True)
    parser.add_argument("--metric", help="Metric to measure the distance between embeddings.",
                        type=str, choices=["cosine", "euclidean", "cityblock", "chebyshev"], default="cosine")
    parser.add_argument("--aggr_score", help="Scoring function to assign score to labels.",
                        type=str, choices=["sum", "sqrt_sum", "inverse", "gaussian",], default="sum")
    parser.add_argument("--thresholds", help="Thresholds to distinguish between informative and non-informative content.",
                        type=float, nargs='+', default=0.5)
    parser.add_argument("--knn", help="Number of nearest neighbors to consider for the final score.",
                        type=int, default=5)
    parser.add_argument("--run", help="What main operation the script will perform",
                        type=str, choices=["voxel", "quantity", "correlation", "distance", "accuracy", "auroc", "ckp_distance", "ckp_correlation", "ckp_accuracy"], required=True)
    return parser