# Experiment Plan

## Baselines
- B0: LLM without retrieval
- B1: Dense vector RAG
- B2: Hybrid retrieval + reranking
- Proposed: Evidence-aware selective RAG

## Core metrics
- Retrieval Recall@K / hit rate
- Answer correctness
- Citation precision
- Citation coverage
- Unsupported Answer Rate
- False Acceptance Rate
- False Rejection Rate
- Selective accuracy
- Coverage

## Ablations
Remove one component at a time:
- reranker
- contradiction detector
- citation support checker
- evidence sufficiency score
- abstention threshold

## Stress tests
Evaluate with:
- missing evidence
- conflicting evidence
- outdated evidence
- irrelevant retrieval
- partial evidence
