from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker
from .evidence_scorer import EvidenceScorer
from .semantic_sufficiency import semantic_sufficiency
from .llm import generate


class EvidenceAwareRAGV4:

    def __init__(
        self,
        semantic_confidence_threshold=0.70,
        candidate_k=20,
        top_k=5,
    ):
        self.semantic_confidence_threshold = (
            semantic_confidence_threshold
        )

        self.candidate_k = candidate_k
        self.top_k = top_k

        self.retriever = HybridRetriever(
            alpha=0.5,
            candidate_k=self.candidate_k,
        )

        self.reranker = CrossEncoderReranker()

        self.evidence_scorer = EvidenceScorer()

    def _retrieve_and_rerank(
        self,
        question,
    ):
        candidates = self.retriever.search(
            question,
            top_k=self.candidate_k,
        )

        evidence = self.reranker.rerank(
            question,
            candidates,
            top_k=self.top_k,
        )

        return evidence

    def _build_prompt(
        self,
        question,
        evidence,
    ):
        evidence_blocks = []

        for i, item in enumerate(
            evidence,
            start=1,
        ):
            evidence_blocks.append(
                (
                    f"[Evidence {i}]\n"
                    f"Document: "
                    f"{item.get('document_id', 'unknown')}\n"
                    f"Chunk: "
                    f"{item.get('chunk_id', 'unknown')}\n"
                    f"{item.get('text', '')}"
                )
            )

        evidence_text = "\n\n".join(
            evidence_blocks
        )

        return f"""
Answer the university-policy question using
only the retrieved evidence.

Do not use outside knowledge.

Do not invent missing facts.

If the question contains a false premise and
the evidence disproves it, clearly correct
the premise.

Answer directly and concisely.

Question:
{question}

Retrieved evidence:
{evidence_text}

Answer:
"""

    def answer(
        self,
        question,
    ):
        # ---------------------------------
        # 1. Retrieve and rerank evidence
        # ---------------------------------

        evidence = (
            self._retrieve_and_rerank(
                question
            )
        )

        # ---------------------------------
        # 2. Evidence diagnostics
        # ---------------------------------

        evidence_result = (
            self.evidence_scorer.score(
                question,
                evidence,
            )
        )

        evidence_score = (
            evidence_result[
                "evidence_score"
            ]
        )

        score_components = {
            "relevance":
                evidence_result.get(
                    "relevance",
                    0.0,
                ),

            "overlap":
                evidence_result.get(
                    "overlap",
                    0.0,
                ),

            "agreement":
                evidence_result.get(
                    "agreement",
                    0.0,
                ),

            "margin":
                evidence_result.get(
                    "margin",
                    0.0,
                ),
        }

        # ---------------------------------
        # 3. Semantic evidence sufficiency
        # ---------------------------------

        semantic_result = (
            semantic_sufficiency(
                question,
                evidence,
            )
        )

        semantic_label = (
            semantic_result[
                "label"
            ]
        )

        semantic_confidence = (
            semantic_result[
                "confidence"
            ]
        )

        semantic_reason = (
            semantic_result[
                "reason"
            ]
        )

        # ---------------------------------
        # 4. Decision policy
        #
        # Evidence score is diagnostic.
        # Semantic sufficiency determines
        # whether answering is justified.
        # ---------------------------------

        should_answer = (
            semantic_label
            == "sufficient"
            and semantic_confidence
            >= self.semantic_confidence_threshold
        )

        if not should_answer:

            return {
                "answer":
                    "INSUFFICIENT EVIDENCE",

                "decision":
                    "abstain",

                "evidence":
                    evidence,

                "evidence_score":
                    evidence_score,

                "score_components":
                    score_components,

                "semantic_label":
                    semantic_label,

                "semantic_confidence":
                    semantic_confidence,

                "semantic_reason":
                    semantic_reason,
            }

        # ---------------------------------
        # 5. Generate grounded answer
        # ---------------------------------

        prompt = self._build_prompt(
            question,
            evidence,
        )

        generated_answer = generate(
            prompt
        )

        return {
            "answer":
                generated_answer.strip(),

            "decision":
                "answer",

            "evidence":
                evidence,

            "evidence_score":
                evidence_score,

            "score_components":
                score_components,

            "semantic_label":
                semantic_label,

            "semantic_confidence":
                semantic_confidence,

            "semantic_reason":
                semantic_reason,
        }