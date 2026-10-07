from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker
from .evidence_scorer import EvidenceScorer
from .context_sufficiency import context_sufficiency_score
from .llm import generate


SYSTEM_INSTRUCTION = """
You are a university information assistant.

Answer ONLY from the evidence provided.

Rules:
- Use only information explicitly supported by the evidence.
- Give a concise direct answer.
- Cite supporting evidence using [1], [2], etc.
- Do not invent policies, dates, requirements, or exceptions.
""".strip()


class EvidenceAwareRAGV2:

    def __init__(
        self,
        evidence_threshold: float = 0.55,
        context_threshold: float = 0.70,
    ):
        self.retriever = HybridRetriever(
            alpha=0.5,
            candidate_k=20,
        )

        self.reranker = CrossEncoderReranker()
        self.scorer = EvidenceScorer()

        self.evidence_threshold = evidence_threshold
        self.context_threshold = context_threshold

    def answer(self, question: str, top_k: int = 5):

        candidates = self.retriever.search(
            question,
            top_k=20,
        )

        evidence = self.reranker.rerank(
            question,
            candidates,
            top_k=top_k,
        )

        score_info = self.scorer.score(
            question,
            evidence,
        )

        evidence_score = score_info["evidence_score"]

        context_info = context_sufficiency_score(
            question
        )

        context_score = context_info["context_score"]

        if (
            evidence_score < self.evidence_threshold
            or context_score < self.context_threshold
        ):
            return {
                "question": question,
                "answer": "INSUFFICIENT EVIDENCE",
                "evidence": evidence,
                "evidence_score": evidence_score,
                "score_components": score_info,
                "context_score": context_score,
                "context_info": context_info,
                "decision": "abstain",
            }

        evidence_blocks = []

        for i, item in enumerate(evidence, start=1):
            evidence_blocks.append(
                f"[{i}]\n"
                f"Title: {item['title']}\n"
                f"Section: {item.get('section', '')}\n"
                f"Text: {item['text']}"
            )

        evidence_text = "\n\n".join(evidence_blocks)

        prompt = f"""
{SYSTEM_INSTRUCTION}

QUESTION:
{question}

EVIDENCE:
{evidence_text}

Provide only the final answer with inline citations.
""".strip()

        answer = generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "evidence_score": evidence_score,
            "score_components": score_info,
            "context_score": context_score,
            "context_info": context_info,
            "decision": "answer",
        }