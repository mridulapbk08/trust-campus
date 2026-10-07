import re


def retrieval_hit(retrieved, gold_document_ids):
    retrieved_ids = {r["document_id"] for r in retrieved}

    if not gold_document_ids:
        return 0

    return int(any(g in retrieved_ids for g in gold_document_ids))


def abstained(answer: str):
    return "INSUFFICIENT EVIDENCE" in answer.upper()


def normalize(text: str):
    text = text.lower()
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"[^a-z0-9\s.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(prediction: str, gold: str):
    prediction_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()

    if not gold_tokens:
        return None

    if not prediction_tokens:
        return 0.0

    pred_counts = {}
    gold_counts = {}

    for token in prediction_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    common = 0

    for token in pred_counts:
        if token in gold_counts:
            common += min(
                pred_counts[token],
                gold_counts[token]
            )

    if common == 0:
        return 0.0

    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def answer_decision_correct(answerable: bool, did_abstain: bool):
    if answerable:
        return int(not did_abstain)

    return int(did_abstain)