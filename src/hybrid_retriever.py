import re
import numpy as np
from rank_bm25 import BM25Okapi

from .retriever import DenseRetriever
from .config import CHUNKS_FILE, TOP_K
from .io_utils import read_jsonl


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def minmax_normalize(score_dict):
    if not score_dict:
        return {}

    values = list(score_dict.values())
    min_score = min(values)
    max_score = max(values)

    if max_score == min_score:
        return {
            key: 1.0
            for key in score_dict
        }

    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in score_dict.items()
    }


class HybridRetriever:
    def __init__(
        self,
        alpha: float = 0.5,
        candidate_k: int = 15,
    ):
        """
        alpha:
            weight for dense retrieval

        1 - alpha:
            weight for BM25 retrieval
        """

        self.alpha = alpha
        self.candidate_k = candidate_k

        self.dense = DenseRetriever()

        self.rows = read_jsonl(CHUNKS_FILE)

        tokenized_corpus = [
            tokenize(row["text"])
            for row in self.rows
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

        self.row_by_chunk = {
            row["chunk_id"]: row
            for row in self.rows
        }

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ):
        # ---------------------------------
        # Dense retrieval
        # ---------------------------------

        dense_results = self.dense.search(
            query,
            top_k=self.candidate_k
        )

        dense_raw = {
            item["chunk_id"]:
                item["retrieval_score"]
            for item in dense_results
        }

        # ---------------------------------
        # BM25 retrieval
        # ---------------------------------

        query_tokens = tokenize(query)

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        top_bm25_indices = np.argsort(
            bm25_scores
        )[::-1][:self.candidate_k]

        bm25_raw = {}

        for idx in top_bm25_indices:
            row = self.rows[idx]

            bm25_raw[
                row["chunk_id"]
            ] = float(
                bm25_scores[idx]
            )

        # ---------------------------------
        # Score normalization
        # ---------------------------------

        dense_norm = minmax_normalize(
            dense_raw
        )

        bm25_norm = minmax_normalize(
            bm25_raw
        )

        # ---------------------------------
        # Candidate union
        # ---------------------------------

        candidate_ids = (
            set(dense_raw.keys())
            |
            set(bm25_raw.keys())
        )

        results = []

        for chunk_id in candidate_ids:

            dense_score = dense_norm.get(
                chunk_id,
                0.0
            )

            bm25_score = bm25_norm.get(
                chunk_id,
                0.0
            )

            hybrid_score = (
                self.alpha * dense_score
                +
                (1 - self.alpha)
                * bm25_score
            )

            row = dict(
                self.row_by_chunk[
                    chunk_id
                ]
            )

            row["dense_score"] = (
                dense_score
            )

            row["bm25_score"] = (
                bm25_score
            )

            row["hybrid_score"] = (
                hybrid_score
            )

            row["retrieval_score"] = (
                hybrid_score
            )

            results.append(row)

        results.sort(
            key=lambda x:
                x["hybrid_score"],
            reverse=True,
        )

        return results[:top_k]