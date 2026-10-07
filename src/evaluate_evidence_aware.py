from .config import BENCHMARK, ROOT
from .io_utils import (
    read_jsonl,
    write_jsonl,
)

from .rag_evidence_aware import (
    EvidenceAwareRAG,
)

from .metrics import (
    retrieval_hit,
    abstained,
    token_f1,
    answer_decision_correct,
)


def safe_divide(a, b):
    return a / b if b else 0.0


def main():

    questions = read_jsonl(
        BENCHMARK
    )

    if not questions:
        raise SystemExit(
            f"No benchmark questions "
            f"found at {BENCHMARK}."
        )

    rag = EvidenceAwareRAG(
        threshold=0.55
    )

    rows = []

    for q in questions:

        print(
            f"Evaluating "
            f"{q['question_id']}: "
            f"{q['question']}"
        )

        out = rag.answer(
            q["question"]
        )

        did_abstain = abstained(
            out["answer"]
        )

        row = {
            **q,

            "model_answer":
                out["answer"],

            "abstained":
                did_abstain,

            "evidence_score":
                out[
                    "evidence_score"
                ],

            "score_components":
                out[
                    "score_components"
                ],

            "system_decision":
                out["decision"],

            "retrieval_hit":
                retrieval_hit(
                    out["evidence"],
                    q.get(
                        "gold_document_ids",
                        []
                    )
                ),

            "retrieved_document_ids":
                [
                    item[
                        "document_id"
                    ]
                    for item
                    in out["evidence"]
                ],

            "retrieved_chunk_ids":
                [
                    item[
                        "chunk_id"
                    ]
                    for item
                    in out["evidence"]
                ],

            "token_f1":
                (
                    token_f1(
                        out["answer"],
                        q[
                            "gold_answer"
                        ],
                    )
                    if q[
                        "answerable"
                    ]
                    else None
                ),

            "decision_correct":
                answer_decision_correct(
                    q["answerable"],
                    did_abstain,
                ),
        }

        rows.append(row)

        print(
            f"  Evidence score: "
            f"{out['evidence_score']:.3f}"
        )

        print(
            f"  Decision: "
            f"{out['decision']}"
        )

    results_path = (
        ROOT
        / "results"
        / "evidence_aware_rag.jsonl"
    )

    write_jsonl(
        results_path,
        rows,
    )

    retrieval_rows = [
        r for r in rows
        if r.get(
            "gold_document_ids"
        )
    ]

    answerable_rows = [
        r for r in rows
        if r["answerable"]
    ]

    unanswerable_rows = [
        r for r in rows
        if not r["answerable"]
    ]

    answered_rows = [
        r for r in rows
        if not r["abstained"]
    ]

    false_rejections = [
        r for r in answerable_rows
        if r["abstained"]
    ]

    false_acceptances = [
        r for r
        in unanswerable_rows
        if not r["abstained"]
    ]

    retrieval_accuracy = safe_divide(
        sum(
            r["retrieval_hit"]
            for r
            in retrieval_rows
        ),
        len(retrieval_rows),
    )

    coverage = safe_divide(
        len(answered_rows),
        len(rows),
    )

    decision_accuracy = safe_divide(
        sum(
            r["decision_correct"]
            for r in rows
        ),
        len(rows),
    )

    false_rejection_rate = (
        safe_divide(
            len(false_rejections),
            len(answerable_rows),
        )
    )

    false_acceptance_rate = (
        safe_divide(
            len(false_acceptances),
            len(unanswerable_rows),
        )
    )

    answerable_f1 = [
        r["token_f1"]
        for r in answerable_rows
        if (
            not r["abstained"]
            and
            r["token_f1"]
            is not None
        )
    ]

    avg_f1 = (
        safe_divide(
            sum(answerable_f1),
            len(answerable_f1),
        )
        if answerable_f1
        else 0.0
    )

    print()
    print("=" * 60)

    print(
        "EVIDENCE-AWARE RAG RESULTS"
    )

    print("=" * 60)

    print(
        f"Document Retrieval Hit Rate: "
        f"{retrieval_accuracy:.3f}"
    )

    print(
        f"Coverage: "
        f"{coverage:.3f}"
    )

    print(
        f"Answer/Abstain Decision Accuracy: "
        f"{decision_accuracy:.3f}"
    )

    print(
        f"False Rejection Rate: "
        f"{false_rejection_rate:.3f}"
    )

    print(
        f"False Acceptance Rate: "
        f"{false_acceptance_rate:.3f}"
    )

    print(
        f"Average Token F1: "
        f"{avg_f1:.3f}"
    )

    print()

    print(
        "False Rejections:",
        [
            r["question_id"]
            for r
            in false_rejections
        ]
    )

    print(
        "False Acceptances:",
        [
            r["question_id"]
            for r
            in false_acceptances
        ]
    )

    print()

    print(
        "Evidence Scores:"
    )

    for row in rows:
        print(
            f"  "
            f"{row['question_id']}: "
            f"{row['evidence_score']:.3f} "
            f"({'answerable' if row['answerable'] else 'unanswerable'})"
        )

    print()

    print(
        f"Saved {len(rows)} rows "
        f"to {results_path}"
    )


if __name__ == "__main__":
    main()