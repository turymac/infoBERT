import numpy as np
from sklearn.metrics import accuracy_score

from utils.utils import get_metric

def compute_accuracy(args, model, centroids, sentences_df):
    compute_distance = get_metric(args.metric)

    model_name = args.model.replace("/", "_")
    threshold = args.thresholds  # Assumo che la soglia sia definita in args

    print(f"Computing accuracy on test set for {model_name}")

    accuracies = []

    for cat in set(sentences_df["Category"].tolist()):
        cat_df = sentences_df.loc[sentences_df["Category"] == cat].copy()
        cat_embeddings = model.encode(cat_df["Sentence"].tolist())  # Calcola gli embeddings con model.encode()
        cat_labels = np.array(cat_df["Environment_info"].tolist())

        model_labels = []
        for embedding in cat_embeddings:
            # Calcola la distanza tra l'embedding della frase e ogni centroide
            distances = [compute_distance(embedding, centroid) for centroid in centroids]
            label = 1 if min(distances) < threshold else 0  # Assegna l'etichetta sulla base della soglia
            model_labels.append(label)

        # Calcolo dell'accuratezza per questa categoria
        acc = accuracy_score(cat_labels, model_labels)
        accuracies.append(acc)

        print(f"> Accuracy for category '{cat}': {acc:.4f}")

    # Restituisce la media dell'accuratezza su tutte le categorie
    overall_accuracy = np.mean(accuracies)
    print(f"\n>> Overall Accuracy: {overall_accuracy:.4f}")

    return overall_accuracy
