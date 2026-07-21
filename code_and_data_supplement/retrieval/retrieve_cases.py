"""
Retrieval system for CMR. Section 5.3.
Uses all-MiniLM-L6-v2 to encode serialized CPO evidence.
Retrieves top-K cases from the training case base.
Excludes query itself and same-session samples.
"""

import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


class CaseRetriever:
    """
    Case-base retrieval with same-session exclusion.

    Encoding: all-MiniLM-L6-v2 over serialized CPO evidence.
    Retrieval: cosine similarity (L2-normalized), K=3, brute-force.
    """

    def __init__(
        self,
        encoder_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
    ):
        self.encoder = SentenceTransformer(encoder_model)
        self.top_k = top_k
        self.case_base = []          # list of {sample_id, session_id, evidence_text, label, rationale}
        self.embeddings = None        # (N, D) normalized
        self.index = None             # NearestNeighbors

    def build_index(self, cases: list):
        """
        Build retrieval index from training cases.

        Args:
            cases: list of dicts with keys:
                sample_id, session_id, evidence_text, label, rationale
        """
        self.case_base = cases
        texts = [c["evidence_text"] for c in cases]
        self.embeddings = self.encoder.encode(
            texts, batch_size=32, show_progress_bar=True,
        )

        # L2-normalize for cosine distance
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = self.embeddings / norms

        self.index = NearestNeighbors(
            n_neighbors=min(self.top_k + 10, len(cases)),
            metric="cosine",
            algorithm="brute",
        )
        self.index.fit(normalized)

    def retrieve(self, query_evidence_text: str, query_sample_id: str, query_session_id: str):
        """
        Retrieve top-K cases, excluding the query itself and same-session cases.

        Returns list of {sample_id, label, rationale, similarity}.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self.encoder.encode([query_evidence_text])
        query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-8)

        # Retrieve extra candidates to allow for exclusions
        distances, indices = self.index.kneighbors(
            query_vec, n_neighbors=min(self.top_k + 10, len(self.case_base)),
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            case = self.case_base[idx]
            # Exclude self and same-session samples
            if case["sample_id"] == query_sample_id:
                continue
            if case["session_id"] == query_session_id:
                continue
            results.append({
                "sample_id": case["sample_id"],
                "label": case["label"],
                "rationale": case.get("rationale", ""),
                "similarity": 1.0 - float(dist),
            })
            if len(results) >= self.top_k:
                break

        return results
