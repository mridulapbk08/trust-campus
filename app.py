from __future__ import annotations

import html
import importlib
import inspect
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "final_benchmark_v4_50q.jsonl"
PAPER_URL = "https://doi.org/10.5281/zenodo.23221797"

st.set_page_config(page_title="TRUST-Campus · Evidence-Aware RAG", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {color-scheme: light;}
.stApp {background:#FFFFFF;color:#171717;}
.block-container {max-width:1320px;padding-top:2.1rem;padding-bottom:3rem;}
section[data-testid="stSidebar"] {background:#171717;border-right:5px solid #C8102E;}
section[data-testid="stSidebar"] * {color:#FFFFFF !important;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {background:transparent;}
section[data-testid="stSidebar"] [data-testid="stRadio"] input {accent-color:#C8102E;}
section[data-testid="stSidebar"] hr {border-color:#404040;}
section[data-testid="stSidebar"] a {color:#FFFFFF !important;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#C2C2C2 !important;}
h1,h2,h3 {color:#252525;letter-spacing:-.025em;}
h2 {font-size:1.5rem !important;}
.hero {background:#C8102E;color:#fff;border:0;border-radius:4px;padding:2.35rem 2.4rem;margin-bottom:1.8rem;box-shadow:none;}
.eyebrow {color:#FFFFFF;font-size:.72rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;}
.hero-title {font-size:2.6rem;line-height:1.15;letter-spacing:-.055em;font-weight:800;color:#fff;margin:.4rem 0 .65rem;}
.hero-sub {color:#FFFFFF;line-height:1.7;font-size:.98rem;max-width:850px;}
.hero-tag {display:inline-block;margin-top:1rem;padding:.42rem .75rem;border-radius:3px;background:#FFFFFF;color:#B20D29;font-size:.75rem;font-weight:650;}
.section-kicker {font-size:.73rem;text-transform:uppercase;letter-spacing:.14em;font-weight:800;color:#C8102E;margin:0 0 .4rem;}
.soft-card {padding:1.2rem 1.25rem;background:#fff;border:1px solid #E2E3E5;border-radius:6px;margin-bottom:.8rem;}
.card-label {font-size:.77rem;color:#636363;text-transform:uppercase;font-weight:700;letter-spacing:.07em;}
.card-title {font-size:1.05rem;font-weight:750;color:#292929;margin:.25rem 0;}
.card-value {font-size:1.8rem;color:#252525;font-weight:800;margin:.2rem 0;}
.answer-block {border:1px solid #cadfda;border-left:5px solid #27937d;border-radius:6px;background:#fff;padding:1.3rem 1.5rem;margin:.75rem 0;}
.abstain-block {border:1px solid #ecd9b8;border-left:5px solid #c78b23;border-radius:6px;background:#fff;padding:1.3rem 1.5rem;margin:.75rem 0;}
.decision-title {font-size:1.08rem;font-weight:800;color:#292929;margin-bottom:.5rem;}
.decision-body {font-size:.96rem;line-height:1.75;color:#383838;white-space:pre-wrap;overflow-wrap:anywhere;}
.source-label {font-size:.8rem;color:#777777;}
.stButton>button[kind="primary"]:not(:disabled) {background:#C8102E !important;border:1px solid #C8102E !important;color:#fff !important;border-radius:5px;padding:.68rem 1rem;font-weight:750;box-shadow:0 3px 10px rgba(200,16,46,.16);}
.stButton>button[kind="primary"]:not(:disabled) * {color:#fff !important;}
.stButton>button[kind="primary"]:not(:disabled):hover {background:#A10D26 !important;border-color:#A10D26 !important;color:#fff !important;}
.stButton>button[kind="primary"]:disabled {background:#E4E6E8 !important;border:1px solid #D6D8DB !important;color:#6E747A !important;border-radius:5px;opacity:1;}
.stButton>button[kind="primary"]:disabled * {color:#6E747A !important;}
.stButton>button[kind="primary"]:focus-visible {outline:3px solid #171717;outline-offset:2px;}
.evidence-text {line-height:1.8;font-size:.94rem;color:#383838;white-space:pre-wrap;overflow-wrap:anywhere;max-height:330px;overflow-y:auto;padding:.4rem .2rem .6rem 0;}
div[data-testid="stMetric"] {background:#fff;border:1px solid #E2E3E5;border-radius:6px;padding:1rem 1.1rem;}
div[data-testid="stMetricValue"] {font-weight:800;color:#262626;}
div[data-testid="stExpander"] {background:white;border:1px solid #E2E3E5;border-radius:5px;}
.footer {color:#767676;font-size:.8rem;padding-top:.7rem;line-height:1.7;}
@media(max-width:700px){.block-container{padding-top:1rem}.hero{padding:1.4rem}.hero-title{font-size:2rem}}
</style>
""", unsafe_allow_html=True)


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@st.cache_resource(show_spinner="Loading Evidence-Aware V4 research pipeline…")
def load_rag():
    module = importlib.import_module("src.rag_evidence_aware_v4")
    classes = [
        (name, cls) for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__ and callable(getattr(cls, "answer", None))
        and not name.startswith("_")
    ]
    if not classes:
        raise RuntimeError("No public class implementing answer(question) found in src.rag_evidence_aware_v4")
    classes.sort(key=lambda entry: ("v4" not in entry[0].lower(), "evidence" not in entry[0].lower()))
    return classes[0][1]()


def is_abstention(out: dict) -> bool:
    if isinstance(out.get("abstained"), bool):
        return out["abstained"]
    value = str(out.get("decision", out.get("label", ""))).strip().lower()
    if value in {"abstain", "insufficient", "insufficient_evidence", "reject"}:
        return True
    answer = str(out.get("answer", "")).lower()
    return any(marker in answer for marker in (
        "insufficient evidence", "cannot determine from", "not enough information", "i cannot determine"
    ))


def normalize_evidence(out: dict) -> list[dict]:
    raw = out.get("evidence", out.get("retrieved_evidence", [])) or []
    if isinstance(raw, dict):
        raw = list(raw.values())
    normalized = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            normalized.append({"source": f"Passage {i+1}", "chunk": "", "text": item, "score": None})
        elif isinstance(item, dict):
            normalized.append({
                "source": str(item.get("document_id") or item.get("source") or f"Passage {i+1}"),
                "chunk": str(item.get("chunk_id") or ""),
                "text": str(item.get("text") or item.get("content") or item.get("passage") or ""),
                "score": item.get("rerank_score", item.get("score")),
            })
    return normalized


st.markdown("""
<div class="hero">
  <div class="eyebrow">Independent research · Trustworthy retrieval-augmented generation</div>
  <div class="hero-title">TRUST-Campus <span style="font-weight:400;color:#FFFFFF">/ Research Lab</span></div>
  <div class="hero-sub">A relevant document does not necessarily justify an answer. Investigate how the Evidence-Aware V4 pipeline retrieves university-policy evidence, assesses sufficiency, and chooses to <b>answer or abstain</b>.</div>
  <span class="hero-tag">INDEPENDENT RESEARCH DEMONSTRATION · NOT OFFICIAL UNIVERSITY GUIDANCE</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ◈ TRUST-Campus")
    st.caption("Evidence-aware question answering")
    st.markdown("**Research modes**")
    mode = st.radio("Research modes", ["Live research demo", "Recorded evaluation"], label_visibility="collapsed")
    st.divider()
    st.markdown("**System**  ")
    st.write("Evidence-Aware RAG V4")
    st.markdown(f"[Read the research paper ↗]({PAPER_URL})")
    st.divider()
    st.caption("Independent research demonstration, not official university guidance. Verify CPT, co-op, and enrollment policies with official university sources.")

if mode == "Live research demo":
    st.markdown('<div class="section-kicker">01 / Explore the pipeline</div>', unsafe_allow_html=True)
    st.subheader("Ask a university-policy question")
    examples = [
        "Choose a research question…",
        "Does unpaid experiential learning require CPT authorization for an F-1 student?",
        "Can I begin working immediately after receiving a co-op offer, before CPT approval?",
        "Am I eligible for CPT at Northeastern University?",
        "Am I eligible for co-op?",
    ]
    chosen = st.selectbox("Try a test question", examples)
    if "question_text" not in st.session_state:
        st.session_state.question_text = ""
    if chosen != examples[0] and st.session_state.get("selected_example") != chosen:
        st.session_state.question_text = chosen
    st.session_state.selected_example = chosen
    question = st.text_area("Your question", key="question_text", height=110,
                            placeholder="Ask about course policies, co-op, or CPT guidance…")
    st.caption("Live mode uses your existing locally configured Evidence-Aware V4 backend and indexed corpus.")
    if st.button("Evaluate evidence →", type="primary", use_container_width=True, disabled=not question.strip()):
        with st.spinner("Retrieving passages and assessing evidence sufficiency…"):
            try:
                pipeline = load_rag()
                result = pipeline.answer(question.strip())
                if not isinstance(result, dict):
                    raise TypeError(f"Expected dict, received {type(result).__name__}")
                st.session_state.last_result = (question.strip(), result)
            except Exception as exc:
                st.session_state.pop("last_result", None)
                st.error(f"Live pipeline could not run: {exc}")
                st.info("Check the local model service, installed model, and indexed corpus. No demo answer has been substituted.")

    if st.session_state.get("last_result"):
        asked, out = st.session_state.last_result
        st.divider()
        st.markdown('<div class="section-kicker">02 / Review the decision</div>', unsafe_allow_html=True)
        st.caption(f"Evaluated question: {asked}")
        abstained = is_abstention(out)
        evidence = normalize_evidence(out)
        left, right = st.columns([1.45, 1], gap="large")
        with left:
            st.subheader("Answerability decision")
            text = str(out.get("answer") or "No answer text returned by the pipeline.")
            style = "abstain-block" if abstained else "answer-block"
            name = "ABSTAIN · Evidence judged insufficient" if abstained else "ANSWER · Pipeline returned an answer"
            st.markdown(f'<div class="{style}"><div class="decision-title">{escape(name)}</div>'
                        f'<div class="decision-body">{escape(text)}</div></div>', unsafe_allow_html=True)
            why = out.get("reason") or out.get("sufficiency_reason")
            if why:
                st.info(f"Pipeline-reported explanation: {why}")
            else:
                st.caption("No explicit sufficiency explanation was exposed by this pipeline response. The decision above is the reported output, not a reconstructed explanation.")
            interpretation = (
                "Relevant policy passages may not establish an individual student's eligibility without their specific circumstances. "
                "The pipeline abstained; this is a research interpretation of the outcome, not a model-provided reason."
                if abstained else
                "The pipeline returned an answer to this policy question. Inspect the retrieved passages to assess "
                "whether they support that answer; a high retrieval score does not establish correctness."
            )
            st.markdown('<div class="soft-card"><div class="card-label">Research interpretation · Not model-generated</div>'
                        '<div class="card-title">Relevance ≠ Sufficiency</div>'
                        f'<div class="source-label">{escape(interpretation)}</div></div>',
                        unsafe_allow_html=True)
        with right:
            st.subheader("Evidence inspector")
            st.metric("Retrieved passages", len(evidence))
            st.caption("Ranking scores measure passage relevance; they are **not correctness probabilities**.")
            if not evidence:
                st.info("The pipeline did not return accessible evidence passages.")
            for i, item in enumerate(evidence):
                heading = f"{i+1:02d} · {item['source']}"
                with st.expander(heading):
                    if item["chunk"]:
                        st.caption(f"Passage identifier: {item['chunk']}")
                    st.markdown(f'<div class="evidence-text">{escape(item["text"] or "No passage text provided")}</div>', unsafe_allow_html=True)
                    if item["score"] is not None:
                        try:
                            formatted_score = f"{float(item['score']):.2f}"
                        except (TypeError, ValueError):
                            formatted_score = str(item["score"])
                        st.caption(f"Retrieval/reranking score (not confidence): {formatted_score}")
            with st.expander("Research debugging · raw pipeline output"):
                st.json(out)

else:
    st.markdown('<div class="section-kicker">01 / Frozen experiment</div>', unsafe_allow_html=True)
    st.subheader("Recorded 50-question evaluation")
    st.caption("Results from the frozen internal evaluation in the published preprint. These are not live predictions, independent replications, or external validation.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision accuracy", "82%")
    m2.metric("Answer coverage", "42%")
    m3.metric("False rejection", "20%")
    m4.metric("False acceptance", "16.7%")
    st.warning("**Important limitation:** The reranked baseline had lower false acceptance (3.3%) than Evidence-Aware V4 (16.7%). Higher decision accuracy does not mean the system is safer on every metric.")
    st.markdown("#### Comparison with the strongest conventional baseline")
    comparison = pd.DataFrame(
        {"Evidence-Aware V4": [82, 42, 16.7], "Reranked baseline": [74, 18, 3.3]},
        index=["Decision accuracy", "Answer coverage", "False acceptance"],
    )
    chart_data = comparison.rename_axis("Research metric").reset_index().melt(
        id_vars="Research metric", var_name="System", value_name="Percent"
    )
    chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Research metric:N", sort=["Decision accuracy", "Answer coverage", "False acceptance"],
                axis=alt.Axis(labelAngle=0, title=None)),
        xOffset=alt.XOffset("System:N"),
        y=alt.Y("Percent:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(title="Percent")),
        color=alt.Color("System:N", scale=alt.Scale(domain=["Evidence-Aware V4", "Reranked baseline"], range=["#C8102E", "#969696"]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Research metric:N", "System:N", alt.Tooltip("Percent:Q", format=".1f")],
    ).properties(height=330)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Metric order: decision accuracy, answer coverage, false acceptance. Values come from the frozen internal evaluation.")
    st.divider()
    st.markdown('<div class="section-kicker">02 / Inspect a recorded case</div>', unsafe_allow_html=True)
    rows = read_jsonl(RESULTS)
    if rows:
        choices = [f"{row.get('question_id', row.get('id', i+1))} · {str(row.get('question', ''))[:100]}" for i, row in enumerate(rows)]
        index = st.selectbox("Choose a benchmark question", range(len(rows)), format_func=lambda i: choices[i])
        item = rows[index]
        a, b = st.columns([1.45, 1], gap="large")
        with a:
            st.markdown("#### Recorded answer")
            st.write(item.get("model_answer") or item.get("answer") or "No stored answer")
            st.caption("Recorded output from a previous run; not a new model prediction.")
        with b:
            st.markdown("#### Evaluation labels")
            st.write("**Expected answerable:**", item.get("answerable", "Not recorded"))
            st.write("**Model abstained:**", item.get("abstained", "Not recorded"))
            st.write("**Decision correct:**", item.get("decision_correct", "Not recorded"))
            ids = item.get("retrieved_document_ids") or []
            if ids:
                st.write("**Retrieved documents:**", ", ".join(map(str, ids)))
    else:
        st.info(f"Recorded cases not found. Expected: {RESULTS.relative_to(ROOT)}")
        st.caption("Aggregate metrics above are published historical results, not recalculated values.")

st.divider()
st.markdown('<div class="footer">TRUST-Campus · Independent research demonstration · '
            'Not an institutional advising service · Verify any policy with official university sources · '
            f'<a href="{PAPER_URL}" target="_blank">Read research preprint ↗</a></div>', unsafe_allow_html=True)
