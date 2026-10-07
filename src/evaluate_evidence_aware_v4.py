import json
import re
import sys
from pathlib import Path

from .rag_evidence_aware_v4 import (
    EvidenceAwareRAGV4,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def read_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


def write_jsonl(
    path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def tokenize(text):
    return re.findall(
        r"\b[a-z0-9]+\b",
        str(text).lower(),
    )


def token_f1(
    prediction,
    gold,
):
    pred_tokens = tokenize(
        prediction
    )

    gold_tokens = tokenize(
        gold
    )

    if (
        not pred_tokens
        and not gold_tokens
    ):
        return 1.0

    if (
        not pred_tokens
        or not gold_tokens
    ):
        return 0.0

    pred_counts = {}

    for token in pred_tokens:
        pred_counts[token] = (
            pred_counts.get(
                token,
                0,
            )
            + 1
        )

    gold_counts = {}

    for token in gold_tokens:
        gold_counts[token] = (
            gold_counts.get(
                token,
                0,
            )
            + 1
        )

    overlap = 0

    for token, pred_count in (
        pred_counts.items()
    ):
        overlap += min(
            pred_count,
            gold_counts.get(
                token,
                0,
            ),
        )

    if overlap == 0:
        return 0.0

    precision = (
        overlap
        / len(pred_tokens)
    )

    recall = (
        overlap
        / len(gold_tokens)
    )

    if (
        precision + recall
        == 0
    ):
        return 0.0

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def document_retrieval_hit(
    retrieved_document_ids,
    gold_document_ids,
):
    if not gold_document_ids:
        return 1

    retrieved = set(
        retrieved_document_ids
    )

    gold = set(
        gold_document_ids
    )

    return int(
        bool(
            retrieved.intersection(
                gold
            )
        )
    )


def get_output_path(
    benchmark_path,
):
    benchmark_stem = (
        Path(
            benchmark_path
        ).stem
    )

    return (
        RESULTS_DIR
        / (
            "evidence_aware_rag_v4_"
            f"{benchmark_stem}.jsonl"
        )
    )


def main():
    if len(sys.argv) > 1:
        benchmark_path = Path(
            sys.argv[1]
        )

    else:
        benchmark_path = (
            ROOT
            / "data"
            / "benchmark"
            / "questions_test_30_frozen.jsonl"
        )

    if not benchmark_path.is_absolute():
        benchmark_path = (
            ROOT
            / benchmark_path
        )

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: "
            f"{benchmark_path}"
        )

    questions = read_jsonl(
        benchmark_path
    )

    rag = EvidenceAwareRAGV4(
        semantic_confidence_threshold=0.70,
    )

    rows = []

    retrieval_hits = 0
    answered_count = 0
    correct_decisions = 0

    answerable_count = 0
    unanswerable_count = 0

    false_rejections = []
    false_acceptances = []

    answered_answerable_f1 = []

    for q in questions:

        question_id = q.get(
            "question_id",
            q.get(
                "id",
                "unknown",
            ),
        )

        question = q[
            "question"
        ]

        answerable = bool(
            q.get(
                "answerable",
                False,
            )
        )

        gold_answer = q.get(
            "gold_answer",
            "",
        )

        gold_document_ids = (
            q.get(
                "gold_document_ids",
                [],
            )
        )

        category = q.get(
            "category",
            "",
        )

        print(
            f"Evaluating "
            f"{question_id}: "
            f"{question}"
        )

        out = rag.answer(
            question
        )

        model_answer = out.get(
            "answer",
            "",
        )

        decision = out.get(
            "decision",
            "abstain",
        )

        abstained = (
            decision != "answer"
        )

        evidence_score = float(
            out.get(
                "evidence_score",
                0.0,
            )
        )

        score_components = (
            out.get(
                "score_components",
                {},
            )
        )

        semantic_label = (
            out.get(
                "semantic_label",
                "unsupported",
            )
        )

        semantic_confidence = float(
            out.get(
                "semantic_confidence",
                0.0,
            )
        )

        semantic_reason = (
            out.get(
                "semantic_reason",
                "",
            )
        )

        evidence = out.get(
            "evidence",
            [],
        )

        retrieved_document_ids = [
            item.get(
                "document_id"
            )
            for item in evidence
            if item.get(
                "document_id"
            )
        ]

        retrieved_chunk_ids = [
            item.get(
                "chunk_id"
            )
            for item in evidence
            if item.get(
                "chunk_id"
            )
        ]

        retrieval_hit = (
            document_retrieval_hit(
                retrieved_document_ids,
                gold_document_ids,
            )
        )

        retrieval_hits += (
            retrieval_hit
        )

        if not abstained:
            answered_count += 1

        expected_answer = (
            answerable
        )

        system_answered = (
            not abstained
        )

        decision_correct = int(
            system_answered
            == expected_answer
        )

        correct_decisions += (
            decision_correct
        )

        if answerable:
            answerable_count += 1

            if abstained:
                false_rejections.append(
                    question_id
                )

        else:
            unanswerable_count += 1

            if not abstained:
                false_acceptances.append(
                    question_id
                )

        row_token_f1 = 0.0

        if (
            answerable
            and not abstained
        ):
            row_token_f1 = token_f1(
                model_answer,
                gold_answer,
            )

            answered_answerable_f1.append(
                row_token_f1
            )

        row = {
            "question_id":
                question_id,

            "question":
                question,

            "category":
                category,

            "answerable":
                answerable,

            "gold_answer":
                gold_answer,

            "gold_document_ids":
                gold_document_ids,

            "model_answer":
                model_answer,

            "abstained":
                abstained,

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

            "system_decision":
                decision,

            "retrieval_hit":
                retrieval_hit,

            "retrieved_document_ids":
                retrieved_document_ids,

            "retrieved_chunk_ids":
                retrieved_chunk_ids,

            "token_f1":
                row_token_f1,

            "decision_correct":
                decision_correct,
        }

        rows.append(
            row
        )

        print(
            f"  Evidence score: "
            f"{evidence_score:.3f}"
        )

        print(
            f"  Semantic label: "
            f"{semantic_label}"
        )

        print(
            f"  Semantic confidence: "
            f"{semantic_confidence:.3f}"
        )

        print(
            f"  Decision: "
            f"{decision}"
        )

    total = len(
        questions
    )

    retrieval_hit_rate = (
        retrieval_hits
        / total
        if total
        else 0.0
    )

    coverage = (
        answered_count
        / total
        if total
        else 0.0
    )

    decision_accuracy = (
        correct_decisions
        / total
        if total
        else 0.0
    )

    false_rejection_rate = (
        len(
            false_rejections
        )
        / answerable_count
        if answerable_count
        else 0.0
    )

    false_acceptance_rate = (
        len(
            false_acceptances
        )
        / unanswerable_count
        if unanswerable_count
        else 0.0
    )

    average_token_f1 = (
        sum(
            answered_answerable_f1
        )
        / len(
            answered_answerable_f1
        )
        if answered_answerable_f1
        else 0.0
    )

    print()
    print(
        "=" * 60
    )

    print(
        "EVIDENCE-AWARE RAG V4 RESULTS"
    )

    print(
        "=" * 60
    )

    try:
        display_benchmark = str(
            benchmark_path.relative_to(
                ROOT
            )
        )

    except ValueError:
        display_benchmark = str(
            benchmark_path
        )

    print(
        f"Benchmark: "
        f"{display_benchmark}"
    )

    print(
        "Document Retrieval Hit Rate: "
        f"{retrieval_hit_rate:.3f}"
    )

    print(
        "Coverage: "
        f"{coverage:.3f}"
    )

    print(
        "Answer/Abstain Decision Accuracy: "
        f"{decision_accuracy:.3f}"
    )

    print(
        "False Rejection Rate: "
        f"{false_rejection_rate:.3f}"
    )

    print(
        "False Acceptance Rate: "
        f"{false_acceptance_rate:.3f}"
    )

    print(
        "Average Token F1: "
        f"{average_token_f1:.3f}"
    )

    print()

    print(
        "False Rejections:",
        false_rejections,
    )

    print(
        "False Acceptances:",
        false_acceptances,
    )

    print()

    print(
        "Scores:"
    )

    for row in rows:

        status = (
            "answerable"
            if row[
                "answerable"
            ]
            else "unanswerable"
        )

        print(
            f"  "
            f"{row['question_id']}: "
            f"evidence="
            f"{row['evidence_score']:.3f}, "
            f"semantic="
            f"{row['semantic_label']}, "
            f"confidence="
            f"{row['semantic_confidence']:.3f}, "
            f"{status}"
        )

    output_path = get_output_path(
        benchmark_path
    )

    write_jsonl(
        output_path,
        rows,
    )

    print()

    print(
        f"Saved "
        f"{len(rows)} rows to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()