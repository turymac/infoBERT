import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_curve, auc

from utils.utils import get_metric

def compute_auroc(args, model, centroids, sentences_df):

    def x(path, labels,  scores):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # baseline

        # Punto ottimale evidenziato
        plt.scatter(optimal_fpr, optimal_tpr, color='red', label=f"Best Threshold = {-optimal_threshold:.4f}")
        plt.annotate(f"{-optimal_threshold:.2f}", (optimal_fpr, optimal_tpr), textcoords="offset points", xytext=(10, -10), ha='center', fontsize=5, color='red')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(f"{path}.png", dpi=300)


    compute_distance = get_metric(args.metric)

    model_name = args.model.replace("/", "_")

    print(f"Computing AUROC on test set for {model_name}")

    tot_labels = []
    tot_scores = []
    categories = sorted(sentences_df["Category"].unique())
    for cat in categories:
        cat_df = sentences_df.loc[sentences_df["Category"] == cat].copy()
        cat_embeddings = model.encode(cat_df["Sentence"].tolist())  # Calcola gli embeddings con model.encode()
        cat_labels = np.array(cat_df["Environment_info"].tolist())
        tot_labels.append(cat_labels)

        cat_scores = []
        for embedding in cat_embeddings:
            # Calcola la distanza tra l'embedding della frase e ogni centroide
            distances = [compute_distance(embedding, centroid) for centroid in centroids]
            cat_scores.append(-min(distances))

        tot_scores.append(np.array(cat_scores))

        # Compute AUROC for Category
        x(f"plot/{model_name}/{cat}", cat_labels, cat_scores)

    x(f"plot/{model_name}/all", np.concatenate(tot_labels), np.concatenate(tot_scores))


    return 0

def compute_accuracy(args, model, centroids, sentences_df):
    compute_distance = get_metric(args.metric)

    model_name = args.model.replace("/", "_")
    threshold = args.thresholds  # Assumo che la soglia sia definita in args

    print(f"Computing accuracy on test set for {model_name}")

    accuracies = []
    categories = sorted(sentences_df["Category"].unique())
    for cat in categories:
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
