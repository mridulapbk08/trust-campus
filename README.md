# TRUST-Campus

**TRUST-Campus: Evidence-Aware Retrieval-Augmented Generation for Reliable University Information Access**

TRUST-Campus investigates a central reliability problem in retrieval-augmented generation (RAG):

> Retrieved evidence may be relevant without being sufficient to justify an answer.

The project evaluates whether an explicit evidence-sufficiency decision layer can improve answer/abstain behavior in university information systems.

We compare the proposed **Evidence-Aware RAG V4** system against three baselines:

- Dense RAG
- Hybrid RAG
- Reranked RAG

---

## Research Question

Can a RAG system distinguish between:

1. evidence that is merely relevant to a question, and
2. evidence that is sufficient to support answering that question?

The system is designed to abstain when the retrieved evidence does not adequately support a grounded answer.

---

## Final Evaluation

The final frozen evaluation set contains **50 questions** across seven categories:

| Category | Questions |
|---|---:|
| Paraphrase | 8 |
| Compositional | 6 |
| Negation | 6 |
| Needs Context | 10 |
| Unsupported | 10 |
| Future | 5 |
| False Premise | 5 |
| **Total** | **50** |

The set contains:

- **20 answerable questions**
- **30 unanswerable questions**

The final benchmark is treated as a **frozen final evaluation set**, not as an independently collected external benchmark.

---

## Final Results

| System | Retrieval Hit | Coverage | Decision Accuracy | False Rejection Rate | False Acceptance Rate | Token F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dense RAG | 1.000 | 0.220 | 0.660 | 0.650 | 0.133 | 0.507 |
| Hybrid RAG | 1.000 | 0.260 | 0.660 | 0.600 | 0.167 | 0.480 |
| Reranked RAG | 1.000 | 0.180 | 0.740 | 0.600 | **0.033** | **0.523** |
| **Evidence-Aware RAG V4** | **1.000** | **0.420** | **0.820** | **0.200** | 0.167 | 0.443 |

### Main Finding

Evidence-Aware RAG V4 achieved:

- **82% answer/abstain decision accuracy**
- **42% coverage**
- **20% false rejection rate**

Compared with the strongest conventional retrieval baseline, Reranked RAG:

- Decision accuracy improved from **74% to 82%**
- Coverage increased from **18% to 42%**
- False rejection decreased from **60% to 20%**

Reranked RAG remained the most conservative system, with the lowest false acceptance rate of **3.3%**.

---

## Category-Level Results

| Category | Dense | Hybrid | Reranked | Evidence-Aware V4 |
|---|---:|---:|---:|---:|
| Paraphrase | 0.750 | 0.750 | 0.750 | **1.000** |
| Compositional | 0.000 | 0.167 | 0.333 | **0.667** |
| Negation | 0.167 | 0.167 | 0.000 | **0.667** |
| Needs Context | 0.600 | 0.500 | **0.900** | 0.600 |
| Unsupported | **1.000** | **1.000** | **1.000** | **1.000** |
| Future | **1.000** | **1.000** | **1.000** | **1.000** |
| False Premise | **1.000** | **1.000** | **1.000** | 0.800 |

Evidence-Aware V4 shows the largest gains on:

- paraphrased questions
- compositional questions
- negated questions

The primary remaining weakness is **context-dependent questioning**, where a general policy statement may be relevant but insufficient to answer a user-specific question.

---

## Figures

### Decision Accuracy

![Decision Accuracy](figures/figure1_decision_accuracy.png)

### Coverage vs False Acceptance

![Coverage vs False Acceptance](figures/figure2_coverage_far_tradeoff.png)

### Category-Level Accuracy

![Category Accuracy](figures/figure3_category_accuracy.png)

---

## Repository Structure

trust-campus/
├── data/
├── experiments/
├── figures/
│   ├── figure1_decision_accuracy.png
│   ├── figure2_coverage_far_tradeoff.png
│   └── figure3_category_accuracy.png
├── paper/
│   ├── TRUST_Campus_Research_Paper.docx
│   └── TRUST_Campus_Research_Paper.pdf
├── results/
│   ├── final_benchmark_dense_50q.jsonl
│   ├── final_benchmark_hybrid_50q.jsonl
│   ├── final_benchmark_reranked_50q.jsonl
│   └── final_benchmark_v4_50q.jsonl
├── scripts/
│   └── generate_final_figures.py
├── src/
├── README.md
└── requirements.txt

---

## Reproducibility

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

If needed for figure generation:

pip install matplotlib

---

## Final Benchmark Files

Frozen final evaluation set:

data/benchmark/questions_final_50_frozen.jsonl

Baseline-compatible copy:

data/benchmark/questions_final_50_baseline.jsonl

Final result files:

results/final_benchmark_dense_50q.jsonl
results/final_benchmark_hybrid_50q.jsonl
results/final_benchmark_reranked_50q.jsonl
results/final_benchmark_v4_50q.jsonl

---

## Running the Baselines

The baseline evaluators read the benchmark path defined in:

src/config.py

Dense RAG:
python -m src.evaluate

Hybrid RAG:
python -m src.evaluate_hybrid

Reranked RAG:
python -m src.evaluate_reranked

Evidence-Aware RAG V4:
python -m src.evaluate_evidence_aware_v4 data/benchmark/questions_final_50_frozen.jsonl

---

## Generate Final Figures

python scripts/generate_final_figures.py

Generated outputs:

figures/figure1_decision_accuracy.png
figures/figure2_coverage_far_tradeoff.png
figures/figure3_category_accuracy.png

---

## Frozen V4 Verification

The final V4 implementation was verified against the frozen copies.

cmp -s src/rag_evidence_aware_v4.py src/rag_evidence_aware_v4_frozen.py && echo "RAG SAME" || echo "RAG DIFFERENT"

cmp -s src/semantic_sufficiency.py src/semantic_sufficiency_v4_frozen.py && echo "SEMANTIC SAME" || echo "SEMANTIC DIFFERENT"

Expected output:

RAG SAME
SEMANTIC SAME

---

## Metrics

### Retrieval Hit Rate
Whether the retrieved evidence contains the gold supporting document.

### Coverage
Fraction of all questions for which the system produces an answer rather than abstaining.

### Answer/Abstain Decision Accuracy
Whether the system correctly answers an answerable question or abstains from an unanswerable question.

### False Rejection Rate
Fraction of answerable questions incorrectly rejected.

### False Acceptance Rate
Fraction of unanswerable questions incorrectly answered.

### Token F1
Lexical overlap between the generated answer and the gold answer for answered, answerable questions.

---

## Key Observation

All four systems achieved a retrieval hit rate of **1.000** on questions with gold supporting documents.

This means the major reliability differences were not caused by failure to retrieve the correct document.

Instead, the central challenge was deciding whether the retrieved evidence was actually sufficient to support answering.

This motivates the core principle of TRUST-Campus:

> **Relevant evidence is not necessarily sufficient evidence.**

---

## Limitations

- The final evaluation contains only 50 questions.
- The benchmark focuses on a constrained university-information domain.
- Category sizes are small.
- The evaluation set was constructed for this project rather than independently collected.
- Context-dependent questions remain a significant failure mode.
- Token F1 measures lexical overlap and should not be interpreted as a complete measure of answer quality.

---

## Paper

Full paper:

paper/TRUST_Campus_Research_Paper.pdf

Editable version:

paper/TRUST_Campus_Research_Paper.docx

---

## Author

**Mridula Prabhakar**

Research interests include trustworthy AI systems, reliable software systems, retrieval-augmented generation, AI for software engineering, verification, and autonomous systems.