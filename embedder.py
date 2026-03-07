import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from data_pipeline import load_dataset

"""
Embedding Model Choice Justification

We use 'all-MiniLM-L6-v2' because:

- 384 dimensional embeddings
- small (~90MB)
- fast inference
- strong semantic similarity performance

This model is widely used in semantic search systems.
"""

MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings():

    print("Loading dataset...")

    documents, labels = load_dataset()

    print("Total documents:", len(documents))

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings)

    print("Embedding shape:", embeddings.shape)

    # Save embeddings
    np.save("models/embeddings.npy", embeddings)

    # Save documents
    with open("models/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    # Save labels
    with open("models/labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    print("Embeddings saved.")


if __name__ == "__main__":
    generate_embeddings()
    