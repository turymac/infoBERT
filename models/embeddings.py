import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def compute_embeddings(model_name, dataset_version, save_path="datasets/training_set/precomputed_embeddings"):
    os.makedirs(save_path, exist_ok=True)
    embedding_file = os.path.join(save_path, f"{dataset_version}_{model_name}_embeddings.npy")

    if os.path.exists(embedding_file):
        print(f"Loading precomputed embeddings from {embedding_file}...")
        return np.load(embedding_file)

    dataset_version = "v3"
    df = pd.read_excel(f"/datasets/training_set/environmental_sentences_filtered_{dataset_version}.xlsx")
    df['Label'] = df['Environmental Sentences']
    sentences = df.Label.tolist()
    print(f"{len(sentences)} sentences read")

    print(f"Computing {model_name} embeddings for dataset {dataset_version}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True)
    np.save(embedding_file, embeddings)
    return embeddings