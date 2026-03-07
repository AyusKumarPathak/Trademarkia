import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, similarity_threshold=0.85):

        self.cache = {}  # cluster -> list of entries
        self.threshold = similarity_threshold

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding, cluster):

        if cluster not in self.cache:
            self.miss_count += 1
            return False, None, 0

        best_match = None
        best_score = 0

        for entry in self.cache[cluster]:

            score = cosine_similarity(
                [query_embedding],
                [entry["embedding"]]
            )[0][0]

            if score > best_score and score > self.threshold:

                best_match = entry
                best_score = score

        if best_match:
            self.hit_count += 1
            return True, best_match, best_score

        self.miss_count += 1
        return False, None, 0

    def add(self, query, embedding, result, cluster):

        if cluster not in self.cache:
            self.cache[cluster] = []

        self.cache[cluster].append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = self.hit_count + self.miss_count

        entries = sum(len(v) for v in self.cache.values())

        return {
            "total_entries": entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total else 0
        }

    def clear(self):

        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0