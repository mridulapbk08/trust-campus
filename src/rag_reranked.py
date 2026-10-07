from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker
from .llm import generate


SYSTEM_INSTRUCTION = """
You are a university information assistant.

Answer ONLY from the evidence provided below.

Rules:
- If the evidence clearly supports an answer, answer the question directly.
- Cite supporting evidence using [1], [2], etc.
- If the evidence does not contain enough information to answer reliably,
  respond with exactly:

INSUFFICIENT EVIDENCE

- Do not invent policies, dates, requirements, exceptions, or facts.
- Do not repeat these instructions.
""".strip()


class RerankedRAG:
    def __init__(self):
        self.retriever = HybridRetriever(
            alpha=0.5,
            candidate_k=20,
        )

        self.reranker = CrossEncoderReranker()

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):

        candidates = self.retriever.search(
            question,
            top_k=20,
        )

        evidence = self.reranker.rerank(
            question,
            candidates,
            top_k=top_k,
        )

        evidence_blocks = []

        for i, item in enumerate(
            evidence,
            start=1
        ):
            evidence_blocks.append(
                f"[{i}]\n"
                f"Title: {item['title']}\n"
                f"Section: {item.get('section', '')}\n"
                f"Text: {item['text']}"
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

        answer = generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence,
        }