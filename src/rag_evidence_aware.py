from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker
from .evidence_scorer import EvidenceScorer
from .llm import generate


SYSTEM_INSTRUCTION = """
You are a university information assistant.

Answer ONLY from the evidence provided.

Rules:
- Use only information explicitly supported by the evidence.
- Give a concise direct answer.
- Cite supporting evidence using [1], [2], etc.
- Do not invent policies, dates, requirements, or exceptions.
- Do not repeat these instructions.
""".strip()


class EvidenceAwareRAGV3:

    def __init__(
        self,
        threshold: float = 0.55,
    ):

        self.retriever = HybridRetriever(
            alpha=0.5,
            candidate_k=20,
        )

        self.reranker = (
            CrossEncoderReranker()
        )

        self.scorer = (
            EvidenceScorer()
        )

        self.threshold = threshold

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):

        # ----------------------------
        # Retrieve
        # ----------------------------

        candidates = (
            self.retriever.search(
                question,
                top_k=20,
            )
        )

        # ----------------------------
        # Rerank
        # ----------------------------

        evidence = (
            self.reranker.rerank(
                question,
                candidates,
                top_k=top_k,
            )
        )

        # ----------------------------
        # Score evidence sufficiency
        # ----------------------------

        score_info = (
            self.scorer.score(
                question,
                evidence,
            )
        )

        evidence_score = (
            score_info[
                "evidence_score"
            ]
        )

        # ----------------------------
        # Selective abstention
        # ----------------------------

        if (
            evidence_score
            < self.threshold
        ):

            return {
                "question":
                    question,

                "answer":
                    "INSUFFICIENT EVIDENCE",

                "evidence":
                    evidence,

                "evidence_score":
                    evidence_score,

                "score_components":
                    score_info,

                "decision":
                    "abstain",
            }

        # ----------------------------
        # Generation
        # ----------------------------

        evidence_blocks = []

        for i, item in enumerate(
            evidence,
            start=1,
        ):

            evidence_blocks.append(
                f"[{i}]\n"
                f"Title: "
                f"{item['title']}\n"
                f"Section: "
                f"{item.get('section', '')}\n"
                f"Text: "
                f"{item['text']}"
            )

        evidence_text = "\n\n".join(
            evidence_blocks
        )

        prompt = f"""
{SYSTEM_INSTRUCTION}

QUESTION:
{question}

EVIDENCE:
{evidence_text}

Provide only the final answer with inline citations.
""".strip()

        answer = generate(
            prompt
        )

        return {
            "question":
                question,

            "answer":
                answer,

            "evidence":
                evidence,

            "evidence_score":
                evidence_score,

            "score_components":
                score_info,

            "decision":
                "answer",
        }