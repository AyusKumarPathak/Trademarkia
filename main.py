import time
import numpy as np
import pickle

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from vector_db import VectorStore
from semantic_cache import SemanticCache


# ---------------------------------------------------
# FASTAPI INITIALIZATION
# ---------------------------------------------------

app = FastAPI(
    title="Semantic Search System",
    description="Semantic search with fuzzy clustering and semantic caching",
    version="1.0"
)


# ---------------------------------------------------
# LOAD SYSTEM COMPONENTS
# ---------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading vector database...")
vector_db = VectorStore()

print("Loading clustering model...")
with open("models/gmm_model.pkl", "rb") as f:
    gmm_model = pickle.load(f)

print("Initializing semantic cache...")
cache = SemanticCache(similarity_threshold=0.8)

print("System ready.")


# ---------------------------------------------------
# REQUEST SCHEMA
# ---------------------------------------------------

class QueryRequest(BaseModel):
    query: str


# ---------------------------------------------------
# QUERY ENDPOINT
# ---------------------------------------------------

@app.post("/query")
def query_system(request: QueryRequest):

    query = request.query.strip()

    if not query:
        return {"error": "Query cannot be empty"}

    start_time = time.time()

    # --------------------------------------------
    # Step 1 — Embed Query
    # --------------------------------------------

    query_embedding = model.encode(query)

    # --------------------------------------------
    # Step 2 — Predict Cluster (Fuzzy clustering)
    # --------------------------------------------

    cluster_probs = gmm_model.predict_proba([query_embedding])[0]

    dominant_cluster = int(np.argmax(cluster_probs))

    # --------------------------------------------
    # Step 3 — Check Semantic Cache
    # --------------------------------------------

    hit, match, score = cache.lookup(query_embedding, dominant_cluster)

    if hit:

        latency = time.time() - start_time

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": match["query"],
            "similarity_score": float(score),
            "result": match["result"],
            "dominant_cluster": dominant_cluster,
            "latency_ms": latency * 1000
        }

    # --------------------------------------------
    # Step 4 — Perform Vector Search
    # --------------------------------------------

    results = vector_db.search([query_embedding], k=5)

    # --------------------------------------------
    # Step 5 — Store in Cache
    # --------------------------------------------

    cache.add(query, query_embedding, results, dominant_cluster)

    latency = time.time() - start_time

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": dominant_cluster,
        "latency_ms": latency * 1000
    }


# ---------------------------------------------------
# CACHE STATS ENDPOINT
# ---------------------------------------------------

@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


# ---------------------------------------------------
# CACHE CLEAR ENDPOINT
# ---------------------------------------------------

@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {
        "message": "Cache cleared successfully"
    }