import numpy as np

from sklearn.metrics.pairwise import cosine_distances

from utils.utils import get_label_score, get_metric

def rbf_weights(X, C, gamma=10.0): #Make gamma a parameter in args
    distances = cosine_distances(X, C)
    similarities = np.exp(-gamma * distances**2)
    weights = similarities.sum(axis=1)
    return weights

def compute_information_quantity(args, embedded_sentences, embeddings, centroids, test_df, scores_df, get_avg=False, verbose=False):
    compute_distance = get_metric(args.metric)
    knn = args.knn
    thresholds = args.thresholds

    if knn != len(thresholds):
        raise ValueError('knn must be equal to len(thresholds)')  # Work on 2 thresholds and parameter alpha

    weights = rbf_weights(embeddings, centroids)

    cat_correlations = {}
    categories = sorted(test_df["Category"].unique())
    for cat in categories:
        cat_df = test_df.loc[test_df['Category'] == cat].copy()
        cat_df.sort_values(by="Name", inplace=True)

        model_cat_scores = []
        # Itera su ciascuna etichetta in `etichette`
        for product in cat_df.Name.tolist():
            partial_scores = []
            for _, emb_sentence in embedded_sentences[product]:
                sentence_score = []
                # Calcola la distanza tra l'embedding della frase e ogni centroide
                distances = [(compute_distance(emb_sentence, emb), weight) for emb, weight in zip(embeddings, weights)]
                knn_distances = sorted(distances)[:knn]
                # Calcola lo score parziale della frase
                sentence_score = [max(0, (thresholds[id] - distance) * weight) for id, (distance, weight) in enumerate(knn_distances)]

                partial_scores.append(sentence_score)
            partial_scores = np.array(partial_scores)
            punteggio_totale = get_label_score(args, partial_scores)

            # Aggiunge la riga per l'etichetta
            model_cat_scores.append(punteggio_totale)

        my_cat_scores_df = scores_df.loc[scores_df['Category'] == cat].copy()
        my_cat_scores_df.sort_values(by="Name", inplace=True)
        my_cat_scores = my_cat_scores_df.Norm_score.tolist()

        cat_correlation = np.corrcoef(model_cat_scores, my_cat_scores)[0, 1]
        if verbose == True:
            print(f"> {cat}: {cat_correlation:.3f}")
        cat_correlations[cat] = cat_correlation

    print(f" ----- Avg_correlation: {np.mean(list(cat_correlations.values())):.3f} -----")

    if get_avg == True:
        return np.mean(list(cat_correlations.values()))
    else:
        return cat_correlations
