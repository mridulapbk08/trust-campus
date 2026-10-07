from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 5,
    ):
        if not candidates:
            return []

        pairs = [
            [query, item["text"]]
            for item in candidates
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for item, score in zip(candidates, scores):
            row = dict(item)
            row["rerank_score"] = float(score)
            reranked.append(row)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]