# TRUST-Campus

**Evidence-Aware Retrieval-Augmented Generation for Reliable University Information Access**

Mridula Prabhakar · Independent research preprint, 2026 · [https://doi.org/10.5281/zenodo.23440215](https://doi.org/10.5281/zenodo.23440215)

> **Research question:** When is retrieved information *sufficient* to support an answer, rather than merely relevant to the question?

TRUST-Campus is an experimental retrieval-augmented generation (RAG) system for university-policy questions. It separates **finding relevant source passages** from **deciding whether those passages justify an answer**. When the system judges the evidence insufficient, it abstains instead of generating a potentially unsupported conclusion.

The project includes an Evidence-Aware RAG V4 pipeline, three retrieval-based baselines, a frozen 50-question evaluation, and a **Streamlit research interface** for inspecting live answer/abstain decisions and retrieved evidence.

**Status:** Independent research prototype and publicly archived preprint. Not an official Northeastern University service. Do not rely on its answers for academic, employment, or immigration decisions; verify current requirements with the responsible university office.

## Contents

- [At a glance](#at-a-glance)
- [Research contributions](#research-contributions)
- [Pipeline overview](#pipeline-overview)
- [Results](#results)
- [Interactive research demo](#interactive-research-demo)
- [Running the system](#running-the-system)
- [Reproducing the recorded evaluation](#reproducing-the-recorded-evaluation)
- [Repository map](#repository-map)
- [Error analysis and future directions](#error-analysis-and-future-directions)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Paper and citation](#paper-and-citation)

## At a glance

| Component | Description |
| --- | --- |
| Research focus | Evidence sufficiency, selective answering, abstention, and trustworthy RAG |
| Application domain | Course drop and withdrawal rules, graduate co-op policy, and curricular practical training (CPT) guidance |
| Proposed system | Evidence-Aware RAG V4 |
| Comparisons | Dense RAG, Hybrid RAG, and Reranked RAG |
| Evaluation | 50 frozen, project-constructed questions: 20 answerable and 30 unanswerable |
| Interface | Streamlit (`app.py`), using the existing V4 backend |
| Paper | [TRUST-Campus preprint](https://doi.org/10.5281/zenodo.23440215) |

## Research contributions

The paper makes three bounded, testable contributions:

1. **An explicit semantic sufficiency stage.** After retrieval, Evidence-Aware V4 assesses whether the available passages contain the conditions, scope, and policy relationships needed to justify answering the *specific* question. A high retrieval or reranking score alone does not trigger an answer.
2. **A structured answerability evaluation.** A frozen, researcher-constructed 50-question set includes paraphrase, composition, negation, missing personal context, unsupported requests, future questions, and false premises. The design targets both over-answering and excessive abstention.
3. **A comparative analysis of selective answering.** Dense, hybrid, reranked, and evidence-aware systems are assessed under the same evaluation definitions, emphasizing decision accuracy, coverage, false rejection, and false acceptance alongside token F1.

These are contributions **within this experimental setting**, not claims of a universal verification algorithm or general-purpose policy advisor.

### How the proposed approach differs

| System | Main distinguishing feature | Where answerability is handled |
| --- | --- | --- |
| Dense RAG | Dense retrieval followed by generation | Short abstention instruction to the generator |
| Hybrid RAG | Combines dense and BM25 retrieval signals | Short abstention instruction to the generator |
| Reranked RAG | Reranks candidate passages before generation | Short abstention instruction to the generator |
| **Evidence-Aware V4** | Adds a task-specific semantic sufficiency decision after retrieval | **Separate sufficiency judge with an extended prompt, before generation** |

The key research distinction is **semantic support versus topical relevance**: a policy paragraph can discuss CPT yet still lack the individual facts needed to decide whether a particular student is eligible. Because V4 uses a separate judge with a longer prompt while the baselines rely on a short instruction, the comparison also spans a difference in prompt-engineering effort.

## Pipeline overview

A retrieved policy document can be on-topic without answering the *specific* question asked. For example, a document explaining CPT eligibility requirements does not establish whether an individual student meets all those requirements.

TRUST-Campus introduces a **task-specific semantic evidence-sufficiency decision** between retrieval and response generation:

```text
User question
      |
Hybrid retrieval and cross-encoder reranking
      |
Candidate evidence passages
      |
Evidence-sufficiency decision
      |
      +---- Sufficient --------> Generate a grounded answer
      |
      +---- Insufficient ------> Abstain
```

The sufficiency layer checks whether the retrieved evidence contains the conditions, scope, and policy relationships needed for the requested conclusion. It is intentionally conservative for unsupported, future, and context-dependent questions, while allowing an answer when the evidence supports a direct, compositional, or negated conclusion. It is a *task-specific decision mechanism*, **not** a learned universal verifier or a guarantee that every returned answer is correct. The implementation is in `src/rag_evidence_aware_v4.py` and `src/semantic_sufficiency.py`.

## Results

### Evaluation set

| Category | Questions | Expected answerability |
| --- | ---: | --- |
| Paraphrase | 8 | Answerable |
| Compositional | 6 | Answerable |
| Negation | 6 | Answerable |
| Needs context | 10 | Unanswerable |
| Unsupported | 10 | Unanswerable |
| Future | 5 | Unanswerable |
| False premise | 5 | Unanswerable |
| **Total** | **50** | **20 answerable / 30 unanswerable** |

This is a **frozen internal evaluation set**, not an independently collected external benchmark. It was frozen before the final comparison, and V4 was not tuned after seeing these results. The sufficiency prompt contains worked examples drafted against an earlier question set; no final question reproduces one verbatim, but three are close paraphrases.

### Overall comparison

Values below are from the archived experiment described in the preprint and can be recomputed from the result files in `results/`; they are **not continuously recomputed by the UI**.

| System | Retrieval hit | Coverage | Decision accuracy | False rejection | False acceptance | Token F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense RAG | 1.000 | 0.220 | 0.660 | 0.650 | 0.133 | 0.507 |
| Hybrid RAG | 1.000 | 0.260 | 0.660 | 0.600 | 0.167 | 0.480 |
| Reranked RAG | 1.000 | 0.180 | 0.740 | 0.600 | **0.033** | **0.523** |
| **Evidence-Aware RAG V4** | **1.000** | **0.420** | **0.820** | **0.200** | 0.167 | 0.443 |

Compared with Reranked RAG, the strongest conventional baseline, Evidence-Aware V4 improved **decision accuracy (74% → 82%)** and **answer coverage (18% → 42%)**, while reducing **false rejection (60% → 20%)**.

**Paired significance (exact McNemar tests, V4 against Reranked RAG):**

| Comparison | Discordant questions | p-value |
| --- | --- | ---: |
| Coverage | 12 answered only by V4, 0 answered only by Reranked | 0.0005 |
| Decision accuracy | 8 correct only for V4, 4 correct only for Reranked | 0.39 |
| False acceptance | 4 answered wrongly only by V4, 0 only by Reranked | 0.13 |

The coverage gain is significant; the accuracy and false-acceptance differences are not. Excluding the three questions that closely paraphrase the prompt's worked examples, coverage remains significant (p = 0.002) and accuracy remains non-significant (p = 0.75).

**Critical trade-off:** V4's false-acceptance rate was **16.7%**, compared with **3.3%** for Reranked RAG. Higher overall decision accuracy does not imply that V4 was safer on every metric. Personalized, context-dependent questions were the central failure mode.

All compared systems had a retrieval hit rate of **1.000 on questions with gold supporting documents**. In this setting, identifying the correct document was not the same as deciding whether the document justified an answer.

Token F1 is implementation-dependent: recomputing V4's answers with the baseline tokenizer gives 0.398 rather than 0.443. The decision metrics are unaffected.

### Category-level decision accuracy

| Category | Dense | Hybrid | Reranked | Evidence-Aware V4 |
| --- | ---: | ---: | ---: | ---: |
| Paraphrase | 0.750 | 0.750 | 0.750 | **1.000** |
| Compositional | 0.000 | 0.167 | 0.333 | **0.667** |
| Negation | 0.167 | 0.167 | 0.000 | **0.667** |
| Needs context | 0.600 | 0.500 | **0.900** | 0.600 |
| Unsupported | 1.000 | 1.000 | 1.000 | 1.000 |
| Future | 1.000 | 1.000 | 1.000 | 1.000 |
| False premise | **1.000** | **1.000** | **1.000** | 0.800 |

### Figures

![Decision accuracy](figures/figure1_decision_accuracy.png)

![Coverage versus false acceptance](figures/figure2_coverage_far_tradeoff.png)

![Category-level accuracy](figures/figure3_category_accuracy.png)

### Metrics

| Metric | Definition |
| --- | --- |
| Retrieval hit rate | Gold supporting document appears among retrieved passages, when gold document IDs exist |
| Coverage | Fraction of questions for which the system answers rather than abstains |
| Decision accuracy | Fraction where the system makes the correct answer-versus-abstain decision |
| False rejection rate | Fraction of answerable questions incorrectly rejected |
| False acceptance rate | Fraction of unanswerable questions incorrectly answered |
| Token F1 | Lexical overlap with the gold answer for answered, answerable questions; not a semantic correctness guarantee |

## Interactive research demo

The Streamlit UI offers two modes:

**Live research demo**
- Ask a university-policy question or choose an example.
- Run the existing Evidence-Aware RAG V4 pipeline.
- See the returned **ANSWER** or **ABSTAIN** decision.
- Inspect retrieved text, document/chunk identifiers, and retrieval/reranking scores.
- View research interpretation separately from the pipeline's actual output. Retrieval scores indicate *relevance*, not answer correctness probabilities.

**Recorded evaluation**
- Review metrics from the project's **previously completed**, frozen 50-question experiment.
- Compare Evidence-Aware V4 with the strongest conventional baseline.
- Inspect saved questions, responses, decision labels, and retrieved document IDs.

The recorded evaluation is a display of prior results from `results/final_benchmark_v4_50q.jsonl`, **not** a fresh benchmark run.

### Example questions

| Question | Example observed outcome |
| --- | --- |
| Does unpaid experiential learning require CPT authorization for an F-1 student? | **ANSWER** in a local demo run |
| Am I eligible for co-op? | **ABSTAIN** in a local demo run, because individual eligibility information is missing |

These are illustrative local outputs, not proof of general accuracy. Individual model responses can vary by configuration.

### Interface screenshots

The following screenshots show the **locally running Streamlit research interface**. It is not a publicly hosted service. The university-inspired red-and-white appearance does not imply Northeastern University endorsement.

**Research dashboard — live question input and exploration modes**

![TRUST-Campus dashboard with research-mode sidebar, question selector, and input field](screenshots/dashboard.png)

**Answer decision — generated response and inspectable retrieved passages**

![TRUST-Campus ANSWER decision for an unpaid experiential-learning CPT question, with five retrieved evidence passages](screenshots/answer-decision.png)

**Abstain decision — retrieved evidence is insufficient for individual eligibility**

![TRUST-Campus ABSTAIN decision for an individual co-op eligibility question, with evidence inspector](screenshots/abstain-decision.png)

**Recorded evaluation — aggregate metrics and baseline comparison**

![TRUST-Campus recorded evaluation showing decision accuracy, coverage, rejection, acceptance, and grouped baseline chart](screenshots/evaluation-overview.png)

**Recorded benchmark case — retrieved documents, decision labels, and response text**

![TRUST-Campus recorded benchmark case showing retrieved documents, decision labels, and response text](screenshots/recorded-case.png)

## Running the system

### Configuration

The settings are in `src/config.py`, `src/llm.py` and `src/reranker.py`:

| Setting | Value |
| --- | --- |
| Generator and sufficiency judge | `qwen2.5:7b` through a local Ollama server at `http://localhost:11434`, temperature 0, seed 42 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector index | FAISS inner-product index |
| Chunking | 900 characters with 150-character overlap |
| Passages used per question | 5 (Reranked RAG and V4 rerank 20 candidates down to 5) |

The model is referenced by its Ollama tag; the exact model digest used for the reported runs is not recorded in the repository.

### Corpus and index

The repository includes the collected corpus and the index used for the evaluation: `data/raw/documents.jsonl` (five Northeastern University policy pages), `data/processed/chunks.jsonl` (43 chunks) and `data/processed/faiss.index`. They can be rebuilt with:

```bash
python -m src.collect_sources
python -m src.build_index
```

Rebuilding fetches the current versions of the pages, which may differ from the versions used in the paper. To keep the reported setup, use the committed files.

### Local demo

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen2.5:7b
python -m streamlit run app.py
```

Open **http://localhost:8501** if the browser does not open automatically.

The UI imports `src.rag_evidence_aware_v4` and calls the V4 pipeline's `answer(question)` method. It does not change the frozen research model or evaluation data, and it reports failures rather than producing an answer. If a live question fails, check that the virtual environment is active and that the Ollama server is running with `qwen2.5:7b` available.

## Reproducing the recorded evaluation

The frozen question sets and the reported result records are:

```text
data/benchmark/questions_final_50_frozen.jsonl      final 50 questions (V4 evaluator input)
data/benchmark/questions_final_50_baseline.jsonl    the same 50 questions in the baseline evaluators' format
results/final_benchmark_dense_50q.jsonl
results/final_benchmark_hybrid_50q.jsonl
results/final_benchmark_reranked_50q.jsonl
results/final_benchmark_v4_50q.jsonl
```

**Baselines.** The three baseline evaluators read the file set as `BENCHMARK` in `src/config.py`. That setting currently points to `data/benchmark/questions.jsonl`, an earlier development set. To rerun the reported comparison, set it to `data/benchmark/questions_final_50_baseline.jsonl` first, then run:

```bash
python -m src.evaluate            # Dense RAG   -> results/baseline_dense_rag_v2.jsonl
python -m src.evaluate_hybrid     # Hybrid RAG  -> results/hybrid_rag.jsonl
python -m src.evaluate_reranked   # Reranked RAG -> results/reranked_rag.jsonl
```

**Evidence-Aware V4.** Pass the final question set explicitly; without an argument the evaluator uses the 30-question test set.

```bash
python -m src.evaluate_evidence_aware_v4 data/benchmark/questions_final_50_frozen.jsonl
```

This writes `results/evidence_aware_rag_v4_questions_final_50_frozen.jsonl`. The `final_benchmark_*` files are the preserved copies of the reported runs. Re-running the evaluations calls the language model again, so outputs may not match them exactly.

**Figures.** Regenerate Figures 1 to 3 from the preserved results with:

```bash
python scripts/generate_final_figures.py
```

**Frozen V4 source.** The V4 implementation was frozen for the reported evaluation. To confirm that the active and frozen copies match:

```bash
cmp -s src/rag_evidence_aware_v4.py src/rag_evidence_aware_v4_frozen.py && echo "RAG SAME" || echo "RAG DIFFERENT"
cmp -s src/semantic_sufficiency.py src/semantic_sufficiency_v4_frozen.py && echo "SEMANTIC SAME" || echo "SEMANTIC DIFFERENT"
```

Both report SAME in the published repository.

## Repository map

```text
trust-campus/
├── app.py                      Streamlit research interface
├── src/                        retrievers, reranker, sufficiency judge, RAG pipelines and evaluators
├── data/
│   ├── raw/                    collected policy pages
│   ├── processed/              chunks and FAISS index
│   └── benchmark/              question sets, including the frozen final 50
├── results/                    preserved result records
├── figures/                    Figures 1 to 3 of the paper
├── scripts/
│   └── generate_final_figures.py
├── screenshots/                interface screenshots
├── experiments/                experiment plan
├── paper/                      earlier paper version
├── requirements.txt
└── README.md
```

`src/` also contains the earlier V1–V3 evidence-aware pipelines and their evaluators. `results/generalization_probe_v4_final_30q.jsonl` is an additional V4 run on the 30-question test set and is not part of the paper's reported results. The final paper is the [Zenodo preprint](https://doi.org/10.5281/zenodo.23440215).

## Error analysis and future directions

The paper reports four false rejections by Evidence-Aware V4 (`f009`, `f011`, `f019`, `f020`), concentrated in compositional and negated policy questions. These cases involve recognizing when multiple retrieved statements jointly support a conclusion.

False acceptances are especially important: V4 answered five unanswerable questions (`f024`, `f026`, `f028`, `f030`, `f049`). **Four occur in the needs-context category**, where relevant general policy information does not determine a student's personal eligibility, and **one in the false-premise category**. This shows why high retrieval quality must not be equated with reliable personalized conclusions.

Future work suggested by these results:

- Represent **missing user context** explicitly and ask targeted clarification questions instead of guessing eligibility.
- Develop **calibrated sufficiency scoring** whose threshold can be tuned to a target false-acceptance rate.
- Check **premises and contradictions** so that false assumptions are rejected before answering.
- Evaluate on larger, independently collected, cross-institution question sets, with retrieval-corruption tests and human evaluation.

## Limitations and responsible use

- The evaluation uses **50 questions** from a narrowly scoped, researcher-constructed university-policy benchmark.
- Category sizes are small (compositional and negation have six questions each, false premise five), so a single error changes category accuracy substantially.
- All systems achieved perfect retrieval on this set, so the study does not characterize behaviour under retrieval failure.
- **False acceptance remains a concern:** V4 performed worse on this measure than the reranked baseline.
- Questions requiring personal context remain challenging.
- Retrieval relevance scores are **not** confidence or correctness probabilities.
- Token F1 is a lexical measure and does not establish semantic correctness.
- The policy corpus may become outdated. This prototype should never be used as a source of official CPT, co-op, immigration, or academic advising decisions.

## Paper and citation

**Mridula Prabhakar.** *TRUST-Campus: Evidence-Aware Retrieval-Augmented Generation for Reliable University Information Access.* Independent research preprint, 2026. Zenodo. https://doi.org/10.5281/zenodo.23440215

```bibtex
@misc{prabhakar2026trustcampus,
  author    = {Prabhakar, Mridula},
  title     = {TRUST-Campus: Evidence-Aware Retrieval-Augmented Generation for Reliable University Information Access},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23440215},
  url       = {https://doi.org/10.5281/zenodo.23440215},
  note      = {Independent research preprint}
}
```

**Publication status:** Public preprint; not described here as peer-reviewed.

## Repository reuse and licensing

The repository does not include a `LICENSE` file, so public visibility on GitHub does not by itself grant reuse permissions. Model and university-document licenses may also constrain redistribution.

## Author

**Mridula Prabhakar**
Independent research in trustworthy AI, RAG, evidence sufficiency, reliable software systems, and verification.

[Research paper](https://doi.org/10.5281/zenodo.23440215) · [Portfolio](https://mridulapbk08.github.io/)
