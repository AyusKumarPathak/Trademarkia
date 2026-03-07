import numpy as np
import pickle
from sklearn.mixture import GaussianMixture


"""
Fuzzy Clustering using Gaussian Mixture Model.

Why GMM?

- Produces probability distribution over clusters
- Supports overlapping topics
- Fits the assignment requirement:
  "A document should belong to multiple clusters with probabilities"
"""


def perform_clustering():

    print("Loading embeddings...")

    embeddings = np.load("models/embeddings.npy")

    print("Embedding shape:", embeddings.shape)

    # Number of clusters
    # Dataset originally has 20 topics, so we start with 20 clusters
    n_clusters = 20

    print("Running Gaussian Mixture clustering...")

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=42
    )

    gmm.fit(embeddings)

    # Probability distribution for each document
    cluster_probs = gmm.predict_proba(embeddings)

    print("Cluster probability shape:", cluster_probs.shape)

    # Save model
    with open("models/gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)

    # Save cluster probabilities
    np.save("models/cluster_probs.npy", cluster_probs)

    print("Clustering complete.")
    print("Files saved:")
    print("models/gmm_model.pkl")
    print("models/cluster_probs.npy")


if __name__ == "__main__":
    perform_clustering()