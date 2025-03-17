from google.colab import files
from openpyxl import Workbook

from utils.utils import get_metric

def partial_distance_knn_to_excel(args, embedded_sentences, centroids, test_df):
    compute_distance = get_metric(args.metric)
    knn = args.knn

    model_name = args.model.replace("/","_")
    dataset_version = args.dataset

    # Crea un nuovo workbook e seleziona il foglio attivo
    wb = Workbook()
    ws = wb.active

    # Intestazioni
    headers = ["Name", "Category", "Frase"] + [f"Distanza Top {i + 1}" for i in range(knn)]
    ws.append(headers)

    for cat in set(test_df.Category.tolist()):
        cat_df = test_df.loc[test_df['Category'] == cat].copy()
        cat_df.sort_values(by="Name", inplace=True)

        for product in cat_df.Name.tolist():
            ws.append([product, cat])

            for original_sentence, emb_sentence in embedded_sentences[product]:
                # Calcola la distanza tra l'embedding della frase e ogni centroide
                distances = [compute_distance(emb_sentence, centroid) for centroid in centroids]
                top_knn_distances = sorted(distances)[:knn]

                # Aggiunge una riga con la frase e le distanze rispetto ai centroidi più vicini
                ws.append(["", "", original_sentence] + [f"{d:.3f}" for d in top_knn_distances])

            ws.append(["", "", ""])  # Riga vuota per separazione

    # Salva il workbook in un file .xlsx
    wb.save(f"distance-{model_name}-{args.metric}-{dataset_version}.xlsx")
    # Scarica il workbook
    files.download(f"distance-{model_name}-{args.metric}-{dataset_version}.xlsx")