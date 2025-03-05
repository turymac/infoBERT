from sklearn.decomposition import PCA

def apply_pca(embeddings):
    pca = PCA(n_components=0.95)  # Keep 95% variance
    return pca.fit_transform(embeddings)