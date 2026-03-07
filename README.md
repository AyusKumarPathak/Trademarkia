# **Semantic Search System with Fuzzy Clustering and Semantic Cache**

## **Overview**

This project implements a lightweight **semantic search system** built on the **20 Newsgroups dataset (~20,000 Usenet posts across 20 topics)**.

The system supports **natural language queries** and retrieves **semantically similar documents** using **vector embeddings and a vector database**.

In addition to search, the system introduces a **semantic caching layer** capable of recognizing **paraphrased queries** and avoiding redundant computation.

The architecture combines:

- **Transformer-based text embeddings**
- **Vector similarity search**
- **Fuzzy clustering for semantic topic modeling**
- **A custom-built semantic cache**
- **A FastAPI service for real-time query interaction**

---

# **System Architecture**

```
User Query
    │
    ▼
Embedding Model (Sentence Transformers)
    │
    ▼
Cluster Prediction (Gaussian Mixture Model)
    │
    ▼
Semantic Cache Lookup
    │        │
    │        └── Cache Hit → Return cached result
    │
    ▼
Vector Database Search (FAISS)
    │
    ▼
Top-k Similar Documents
    │
    ▼
Cache Storage
```

---

# **Dataset**

**Source**

https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

The dataset contains approximately **20,000 Usenet posts across 20 topic categories**, including subjects such as:

- computer hardware  
- politics  
- religion  
- sports  
- science  
- space exploration  

---

# **Data Cleaning**

Raw Usenet posts contain substantial noise such as:

- email headers  
- quoted replies  
- signatures  
- network metadata  

To improve semantic representation, the pipeline removes:

- message headers  
- quoted reply blocks  
- PGP signatures  
- email addresses  
- special characters  

Documents shorter than **20 words** are also filtered out to prevent unstable embeddings.

---

# **Part 1 — Embeddings and Vector Database**

## **Embedding Model**

The system uses:

```
sentence-transformers/all-MiniLM-L6-v2
```

### **Reasons for choosing this model**

- **384-dimensional embeddings**
- **fast inference**
- **small model size (~90MB)**
- **strong semantic similarity performance**
- **widely used in production semantic search systems**

Each document is converted into a **dense vector representation**.

---

## **Vector Database**

Embeddings are stored in a **FAISS vector index**.

FAISS enables:

- **extremely fast similarity search**
- **efficient nearest-neighbor retrieval**
- **scalable vector indexing**

The system uses:

```
IndexFlatIP
```

After **L2 normalization**, inner product similarity becomes equivalent to **cosine similarity**.

This allows efficient retrieval of **semantically similar documents**.

---

# **Part 2 — Fuzzy Clustering**

The **20 Newsgroups dataset contains overlapping topics**.

For example:

A document discussing **gun legislation** may relate to both:

- politics  
- firearms  

To capture this semantic overlap, the system uses **Gaussian Mixture Models (GMM)**.

Unlike hard clustering methods (e.g., **k-means**), GMM produces **probabilistic cluster memberships**.

### **Example**

**Document cluster distribution**

```
politics: 0.55
firearms: 0.32
law: 0.13
```

This allows documents to belong to **multiple topics simultaneously**.

---

# **Choosing the Number of Clusters**

Cluster count was explored using **BIC (Bayesian Information Criterion)** across multiple configurations.

Results indicated that **~20 clusters** provides a strong balance between:

- topic granularity  
- model stability  
- dataset structure  

The final model uses:

```
20 clusters
```

Cluster probabilities are stored for all documents.

---

# **Part 3 — Semantic Cache**

Traditional caches rely on **exact query matching**.

### **Example**

```
Query 1: space shuttle launch
Query 2: when does the shuttle launch
```

These queries are **semantically identical** but would miss a traditional cache.

To solve this, a **semantic cache was implemented from scratch**.

---

## **Cache Mechanism**

1. Incoming queries are **embedded**
2. The query cluster is predicted using the **GMM model**
3. Cache lookup is restricted to that cluster
4. **Cosine similarity** is computed between query embeddings
5. If similarity exceeds a threshold → **cache hit**

---

## **Tunable Parameter**

```
similarity_threshold = 0.8
```

- Lower thresholds increase cache hits but risk incorrect matches  
- Higher thresholds reduce false matches but lower hit rate  

---

# **Part 4 — FastAPI Service**

The system exposes a **REST API built using FastAPI**.

FastAPI provides:

- **automatic documentation**
- **asynchronous request handling**
- **high performance**

---

# **API Endpoints**

## **POST /query**

Request:

```json
{
  "query": "when does the shuttle launch"
}
```

Response:

```json
{
  "query": "when does the shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    {
      "doc_id": 13919,
      "score": 0.5612902641296387,
      "text": "sorry for asking a question that s not entirely based on the technical aspects of space but i couldn t find the answer on the faqs i m currently in the uk which makes seeing a space shuttle launch a little difficult however i have been selected to be an exchange student at louisiana state uni from a"
    },
    {
      "doc_id": 13396,
      "score": 0.5409904718399048,
      "text": "archive name space schedule last modified date 93 04 01 14 39 23 space shuttle answers launch schedules tv coverage shuttle launchings and landings schedules and how to see them shuttle operations are discussed in the usenet group sci space shuttle and ken hollis posts a compressed version of the sh"
    },
    {
      "doc_id": 14244,
      "score": 0.5206694602966309,
      "text": "well you better not get the shuttle as your launch vehicle and most elv s have too far of a backlog for political messages if during the campaign season the candidates for president had launched one right around now we d be getting a launch for perot 92 and if they had used the shuttle we d be seein"
    },
    {
      "doc_id": 13854,
      "score": 0.5049104690551758,
      "text": "comet commercial experiment transport is to launch from wallops island virginia and orbit earth for about 30 days it is scheduled to come down in the utah test training range west of salt lake city utah i saw a message in this group toward the end of march that it was to launch on march 27 does anyo"
    },
    {
      "doc_id": 14041,
      "score": 0.4960457980632782,
      "text": "hello out there if your familiar with the comet program then this concerns you comet is scheduled to be launched from wallops island sometime in june does anyone know if an official launch date has been set thanks rob"
    }
  ],
  "dominant_cluster": 19,
  "latency_ms": 105.03458976745605
}
```

---

## **GET /cache/stats**

Response:

```json
{
  "total_entries": 1,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 0.5
}
```

---

## **DELETE /cache**

Response:

```json
{
  "message": "Cache cleared successfully"
}
```

---

# **Running the Project**

## **1. Create Virtual Environment**

```
python -m venv venv
```

Activate:

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

---

## **2. Install Dependencies**

```
pip install -r requirements.txt
```

---

## **3. Generate Embeddings**

```
python embedder.py
```

---

## **4. Run Clustering**

```
python clustering.py
```

---

## **5. Start API Server**

```
uvicorn main:app --reload
```

API documentation will be available at:

```
http://127.0.0.1:8000/docs
```

---

# **Project Structure**

```
semantic-search-system
│
├── dataset/
│
├── models/
│   ├── embeddings.npy
│   ├── documents.pkl
│   ├── labels.pkl
│   ├── gmm_model.pkl
│   └── cluster_probs.npy
│
├── data_pipeline.py
├── embedder.py
├── clustering.py
├── cluster_analysis.py
├── vector_db.py
├── semantic_cache.py
├── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

`cluster_analysis.py` was used to determine the **best number of clusters**.

---

# **AI Assistance Disclosure**

This project was developed primarily through **independent implementation and experimentation**.

During development, **AI-assisted tools** were occasionally used for:

- clarifying library usage and documentation  
- discussing architectural approaches  
- debugging implementation issues  
- improving documentation quality  

All core design decisions, system integration, and implementation of the following components were performed and validated manually:

- data preprocessing pipeline  
- embedding generation workflow  
- FAISS vector search integration  
- fuzzy clustering using Gaussian Mixture Models  
- semantic cache design and similarity logic  
- FastAPI service architecture  

AI tools were used similarly to consulting **technical documentation or developer forums**, and the final system reflects the author's understanding of the underlying concepts.
