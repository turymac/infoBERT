import argparse
import ast

def add_base_args(parser):
    parser.add_argument("--model", help="Nome del modello di embedding da usare.",
                        type=str, required=True)
    parser.add_argument("--clustering-alg", help="Algoritmo di clustering da testare (può essere una lista separata da virgole o 'all').",
                        type=str, required=True,)
    parser.add_argument("--metric", help="Metrica per misurare la distanza tra embedding.",
                        type=str, choices=["cosine", "euclidean", "cityblock", "chebyshev"], default="cosine", )
    parser.add_argument("--threshold", help="Threshold per discriminare l'informazione.",
                        type=float, default=0.5)
    parser.add_argument("--knn", help="Numero di vicini da considerare per lo score finale.",
                        type=int, default=5)

    return parser