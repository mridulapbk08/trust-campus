import math
import re
from collections import Counter


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on",
    "for", "and", "or", "that", "this", "with",
    "what", "how", "does", "do", "did", "can",
    "could", "should", "would", "if", "from",
    "at", "by", "as", "it", "its", "their",
}


def tokenize(text: str):
    tokens = re.findall(
        r"\b[a-z0-9]+\b",
        text.lower()
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def lexical_overlap(
    question: str,
    evidence_text: str,
):
    q_tokens = set(tokenize(question))
    e_tokens = set(tokenize(evidence_text))

    if not q_tokens:
        return 0.0

    overlap = q_tokens.intersection(
        e_tokens
    )

    return len(overlap) / len(q_tokens)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def normalize_rerank_score(score: float):
    """
    Cross-encoder scores are not naturally
    probabilities.

    We map them into [0, 1] only as a
    comparable evidence signal.
    """
    return sigmoid(score)


def source_agreement(evidence):
    """
    Measures whether multiple highly ranked
    chunks point to the same document.

    High agreement suggests that the retrieval
    system repeatedly found evidence from the
    same source.
    """

    if not evidence:
        return 0.0

    document_ids = [
        item["document_id"]
        for item in evidence
    ]

    counts = Counter(document_ids)

    most_common_count = max(
        counts.values()
    )

    return (
        most_common_count
        / len(document_ids)
    )


def score_margin(evidence):
    """
    Measures how strongly the top-ranked
    evidence separates from the next result.
    """

    if not evidence:
        return 0.0

    scores = [
        normalize_rerank_score(
            item.get(
                "rerank_score",
                0.0
            )
        )
        for item in evidence
    ]

    scores.sort(
        reverse=True
    )

    if len(scores) == 1:
        return scores[0]

    margin = (
        scores[0]
        - scores[1]
    )

    # Scale small margins into a useful range
    return min(
        max(
            margin * 5,
            0.0
        ),
        1.0
    )


class EvidenceScorer:

    def __init__(
        self,
        relevance_weight=0.40,
        overlap_weight=0.30,
        agreement_weight=0.20,
        margin_weight=0.10,
    ):

        self.relevance_weight = (
            relevance_weight
        )

        self.overlap_weight = (
            overlap_weight
        )

        self.agreement_weight = (
            agreement_weight
        )

        self.margin_weight = (
            margin_weight
        )

    def score(
        self,
        question: str,
        evidence: list,
    ):

        if not evidence:
            return {
                "evidence_score": 0.0,
                "relevance": 0.0,
                "overlap": 0.0,
                "agreement": 0.0,
                "margin": 0.0,
            }

        # ----------------------------
        # Relevance
        # ----------------------------

        rerank_scores = [
            normalize_rerank_score(
                item.get(
                    "rerank_score",
                    0.0
                )
            )
            for item in evidence
        ]

        relevance = (
            sum(rerank_scores)
            / len(rerank_scores)
        )

        # ----------------------------
        # Query-evidence overlap
        # ----------------------------

        top_text = " ".join(
            item["text"]
            for item in evidence[:3]
        )

        overlap = lexical_overlap(
            question,
            top_text
        )

        # ----------------------------
        # Source agreement
        # ----------------------------

        agreement = source_agreement(
            evidence
        )

        # ----------------------------
        # Rank confidence margin
        # ----------------------------

        margin = score_margin(
            evidence
        )

        # ----------------------------
        # Combined score
        # ----------------------------

        evidence_score = (
            self.relevance_weight
            * relevance
            +
            self.overlap_weight
            * overlap
            +
            self.agreement_weight
            * agreement
            +
            self.margin_weight
            * margin
        )

        return {
            "evidence_score":
                float(evidence_score),

            "relevance":
                float(relevance),

            "overlap":
                float(overlap),

            "agreement":
                float(agreement),

            "margin":
                float(margin),
        }