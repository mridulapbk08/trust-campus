import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

RESULT_FILES = {
    "Dense": "results/final_benchmark_dense_50q.jsonl",
    "Hybrid": "results/final_benchmark_hybrid_50q.jsonl",
    "Reranked": "results/final_benchmark_reranked_50q.jsonl",
    "Evidence-Aware V4": "results/final_benchmark_v4_50q.jsonl",
}

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

def load_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]

def safe_div(a, b):
    return a / b if b else 0.0

all_rows = {
    name: load_rows(path)
    for name, path in RESULT_FILES.items()
}

# Figure 1: Overall decision accuracy
systems = []
accuracies = []

for name, rows in all_rows.items():
    acc = safe_div(
        sum(int(r["decision_correct"]) for r in rows),
        len(rows)
    )
    systems.append(name)
    accuracies.append(acc)

plt.figure(figsize=(8, 5))
bars = plt.bar(systems, accuracies)
plt.ylabel("Decision Accuracy")
plt.title("Answer/Abstain Decision Accuracy by System")
plt.ylim(0, 1.0)
plt.xticks(rotation=15)

for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    OUT_DIR / "figure1_decision_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# Figure 2: Coverage vs False Acceptance Rate
coverage_values = []
far_values = []
labels = []

for name, rows in all_rows.items():
    unanswerable = [
        r for r in rows
        if not r["answerable"]
    ]

    answered = [
        r for r in rows
        if not r["abstained"]
    ]

    false_acceptances = [
        r for r in unanswerable
        if not r["abstained"]
    ]

    coverage = safe_div(
        len(answered),
        len(rows)
    )

    far = safe_div(
        len(false_acceptances),
        len(unanswerable)
    )

    coverage_values.append(coverage)
    far_values.append(far)
    labels.append(name)

plt.figure(figsize=(8, 6))
plt.scatter(far_values, coverage_values, s=100)

for x, y, label in zip(
    far_values,
    coverage_values,
    labels
):
    plt.annotate(
        label,
        (x, y),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("False Acceptance Rate")
plt.ylabel("Coverage")
plt.title("Coverage–False Acceptance Tradeoff")
plt.xlim(0, max(far_values) + 0.04)
plt.ylim(0, max(coverage_values) + 0.08)

plt.tight_layout()
plt.savefig(
    OUT_DIR / "figure2_coverage_far_tradeoff.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# Figure 3: Category-wise decision accuracy
category_order = [
    "paraphrase",
    "compositional",
    "negation",
    "needs_context",
    "unsupported",
    "future",
    "false_premise",
]

category_scores = {}

for name, rows in all_rows.items():
    grouped = defaultdict(list)

    for r in rows:
        grouped[r["category"]].append(
            int(r["decision_correct"])
        )

    category_scores[name] = [
        safe_div(
            sum(grouped[c]),
            len(grouped[c])
        )
        if grouped[c]
        else 0.0
        for c in category_order
    ]

x = list(range(len(category_order)))
width = 0.19

plt.figure(figsize=(12, 6))

for i, (name, scores) in enumerate(category_scores.items()):
    positions = [
        p + (i - 1.5) * width
        for p in x
    ]

    plt.bar(
        positions,
        scores,
        width=width,
        label=name
    )

plt.xticks(
    x,
    [
        "Paraphrase",
        "Compositional",
        "Negation",
        "Needs Context",
        "Unsupported",
        "Future",
        "False Premise",
    ],
    rotation=20
)

plt.ylabel("Decision Accuracy")
plt.title("Decision Accuracy by Question Category")
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
plt.savefig(
    OUT_DIR / "figure3_category_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Generated:")
print(" figures/figure1_decision_accuracy.png")
print(" figures/figure2_coverage_far_tradeoff.png")
print(" figures/figure3_category_accuracy.png")
