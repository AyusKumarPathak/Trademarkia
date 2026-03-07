import numpy as np
from sklearn.mixture import GaussianMixture

embeddings = np.load("models/embeddings.npy")

scores = []

for k in range(10, 40, 5):

    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(embeddings)

    bic = gmm.bic(embeddings)

    scores.append((k, bic))
    print(k, bic)

print("Best cluster size:", min(scores, key=lambda x: x[1]))