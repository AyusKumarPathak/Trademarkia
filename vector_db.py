import faiss
import numpy as np
import pickle


class VectorStore:
    """
    FAISS Vector Database for Semantic Search.

    FAISS is used because:
    - Extremely fast similarity search
    - Widely used in production ML systems
    - Efficient for large embedding collections
    """

    def __init__(self):

        print("Loading embeddings...")

        # Load embeddings generated earlier
        self.embeddings = np.load("models/embeddings.npy")

        # Load original documents
        with open("models/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        # Get embedding dimension
        dim = self.embeddings.shape[1]

        """
        IndexFlatIP performs inner product search.

        After L2 normalization:
            inner product == cosine similarity
        """

        self.index = faiss.IndexFlatIP(dim)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)

        print("Building FAISS index...")

        # Add embeddings to FAISS index
        self.index.add(self.embeddings)

        print("Total vectors indexed:", self.index.ntotal)

    def search(self, query_embedding, k=5):
        """
        Search top k semantically similar documents.
        """

        query_embedding = np.array(query_embedding)

        # Normalize query vector
        faiss.normalize_L2(query_embedding)

        # Perform vector search
        D, I = self.index.search(query_embedding, k)

        results = []

        for idx, score in zip(I[0], D[0]):

            results.append({
                "doc_id": int(idx),
                "score": float(score),
                "text": self.documents[idx][:300]
            })

        return results


# -------------------------
# TEST BLOCK
# -------------------------

if __name__ == "__main__":

    from sentence_transformers import SentenceTransformer

    print("Starting vector store test...\n")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize vector database
    db = VectorStore()

    # Example query
    query = "space shuttle launch"

    print("\nQuery:", query)

    # Generate query embedding
    query_embedding = model.encode([query])

    # Perform search
    results = db.search(query_embedding, k=5)

    print("\nTop Results:\n")

    for r in results:
        print("Score:", r["score"])
        print(r["text"])
        print("-" * 50)