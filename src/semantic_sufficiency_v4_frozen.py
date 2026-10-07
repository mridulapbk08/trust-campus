import json
import re

from .llm import generate


VALID_LABELS = {
    "sufficient",
    "needs_context",
    "unsupported",
}


def _clean_json_text(text: str) -> str:
    """
    Extract a JSON object from an LLM response.
    """

    text = text.strip()

    # Remove markdown code fences if present.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    # Extract the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):
        text = text[start:end + 1]

    return text.strip()


def _fallback_result():
    """
    Conservative fallback if the semantic
    judge produces invalid output.
    """

    return {
        "label": "unsupported",
        "sufficient": False,
        "confidence": 0.0,
        "reason": (
            "Semantic sufficiency judge "
            "returned an invalid response."
        ),
    }


def _parse_confidence(value):
    """
    Parse confidence robustly.

    Accepts:
    - 0.95
    - "0.95"
    - 95
    - "95%"
    """

    try:
        if isinstance(
            value,
            str,
        ):
            value = value.strip()

            if value.endswith("%"):
                number = float(
                    value[:-1]
                )

                return max(
                    0.0,
                    min(
                        1.0,
                        number / 100.0,
                    ),
                )

        confidence = float(
            value
        )

        # Handle accidental 0-100 scale.
        if confidence > 1.0:
            confidence = (
                confidence / 100.0
            )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    except Exception:
        return 0.0


def semantic_sufficiency(
    question: str,
    evidence: list,
) -> dict:
    """
    Determine whether retrieved evidence is
    sufficient to determine an answer.

    Labels:

    sufficient:
        Evidence directly determines the
        answer.

    needs_context:
        Relevant rules may be present, but
        required facts about a specific user,
        job, course, or situation are missing.

    unsupported:
        The requested information is not
        established by the retrieved evidence.
    """

    if not evidence:
        return {
            "label": "unsupported",
            "sufficient": False,
            "confidence": 1.0,
            "reason":
                "No evidence was retrieved.",
        }

    evidence_blocks = []

    for i, item in enumerate(
        evidence,
        start=1,
    ):
        document_id = item.get(
            "document_id",
            "unknown",
        )

        chunk_id = item.get(
            "chunk_id",
            "unknown",
        )

        text = item.get(
            "text",
            "",
        )

        evidence_blocks.append(
            (
                f"[Evidence {i}]\n"
                f"Document: {document_id}\n"
                f"Chunk: {chunk_id}\n"
                f"{text}"
            )
        )

    evidence_text = "\n\n".join(
        evidence_blocks
    )

    prompt = f"""
You are a semantic evidence-sufficiency judge
for a retrieval-augmented generation system.

You are NOT answering the user's question.

Your only task is to determine whether the
retrieved evidence contains enough information
to determine the answer.

Use exactly ONE of these labels:

"sufficient"
"needs_context"
"unsupported"


==================================================
LABEL 1: SUFFICIENT
==================================================

Use "sufficient" when the retrieved evidence
directly determines the answer.

This includes:

- direct factual policy questions
- questions asking what a policy says
- questions asking for a number
- questions asking for a duration
- questions asking for a threshold
- questions asking for a requirement
- questions asking for a prerequisite
- questions asking for a deadline
- questions asking for a general rule
- yes/no questions whose truth can be
  determined from the evidence
- conditional questions when all conditions
  needed to determine the answer are already
  stated in the question
- questions containing a false premise when
  the evidence directly disproves that premise


GENERAL POLICY QUESTIONS

A general question asking WHAT THE RULE,
REQUIREMENT, THRESHOLD, OR PREREQUISITE IS
does NOT require personal context.

Examples:

"How many semester hours does a graduate
student generally need before co-op?"

This asks for the general requirement.

If the evidence states the requirement,
label:

"sufficient"


"What should an F-1 graduate student have
in place before starting co-op?"

This asks for general prerequisites.

If the evidence states those prerequisites,
label:

"sufficient"


Do NOT confuse:

"What are the requirements?"

with:

"Do I personally meet the requirements?"


A reference to:

- a student
- an F-1 student
- a graduate student
- a co-op student
- an employee

does NOT automatically make the question
personal or context-dependent.


==================================================
CONTRADICTION / FALSE-PREMISE RULE
==================================================

Evidence is sufficient when it directly
CONTRADICTS the statement made in the
question.

The evidence does NOT need to repeat the
question using identical wording.

If the evidence logically entails or
contradicts the claim strongly enough to
determine the answer, label:

"sufficient"


Example 1:

Question:

"A student working a full-time co-op is
always prohibited from taking any course at
the same time, correct?"

Evidence:

A course may be permitted while the student
is on full-time co-op, subject to policy.

The evidence disproves "always prohibited".

Therefore:

label = "sufficient"


Example 2:

Question:

"If an experiential learning position is
unpaid, an eligible F-1 student does not
need CPT authorization, correct?"

Evidence:

Eligible F-1 students participating in
experiential learning in the United States
must obtain CPT authorization, including
paid or unpaid experiences.

The evidence disproves the claim.

Therefore:

label = "sufficient"


Do NOT invent a different population such as
"ineligible F-1 students" when the question
explicitly asks about an eligible F-1
student.

Judge the question exactly as written.


Example 3:

Question:

"If my CPT employer changes, can I assume my
existing CPT authorization automatically
covers the new employer?"

Evidence:

CPT authorization is employer-specific and
employer changes must be reported and
approved.

The evidence directly determines that the
existing authorization cannot simply be
assumed to cover the new employer.

Therefore:

label = "sufficient"


Example 4:

Question:

"Does missing all remaining classes
automatically count as formally withdrawing?"

Evidence:

Nonattendance does not constitute formal
withdrawal.

Therefore:

label = "sufficient"


Example 5:

Question:

"Is a 2.8 cumulative GPA enough when the
stated requirement is 3.000?"

Evidence:

The minimum GPA is 3.000.

Therefore:

label = "sufficient"


==================================================
LABEL 2: NEEDS_CONTEXT
==================================================

Use "needs_context" only when answering the
specific question requires missing facts
about the user's particular situation.

The central test is:

Would additional facts about the specific
person, job, course, employment situation,
immigration situation, or eligibility be
required before the policy can be applied?

If YES:

label = "needs_context"


Examples:

"Do I qualify for graduate co-op?"

This requires facts such as:

- completed credits
- GPA
- enrollment status
- program requirements
- pre-co-op requirements

If those facts are not provided:

label = "needs_context"


"How many days do I have to drop my class?"

If different course lengths or term types
have different deadlines and the course type
is unknown:

label = "needs_context"


"Can I take a course while I am working?"

If the answer depends on whether the work is
co-op, CPT, another employment type, or other
circumstances that are not stated:

label = "needs_context"


==================================================
SPECIFIC JOB / PERSONAL-SITUATION RULE
==================================================

Questions referring to a SPECIFIC job,
position, or personal eligibility require
context when facts necessary to apply the
policy are missing.

Examples of phrases that may signal this:

- "this job"
- "my job"
- "this position"
- "my position"
- "am I eligible"
- "do I qualify"
- "can I use CPT for this job"


Example:

"Can I use CPT for this job?"

General CPT policy evidence is NOT enough by
itself.

Answering this requires information about the
specific job and the student's situation,
such as whether the employment is eligible,
academically related or integral as required,
and properly authorized.

If those facts are absent:

label = "needs_context"


IMPORTANT:

Do NOT infer that a specific job qualifies
for CPT merely because the retrieved evidence
contains general CPT rules.


==================================================
LABEL 3: UNSUPPORTED
==================================================

Use "unsupported" when the requested
information is not established by the
retrieved evidence.

This includes:

- unsupported statistics
- unsupported averages
- unsupported percentages
- unsupported salaries
- unsupported rankings
- unsupported employer counts
- unsupported participation counts
- future predictions
- future policy values
- future tuition values
- future processing times
- future employment outcomes
- facts not present in the supplied evidence


Examples:

"What will the graduate co-op GPA requirement
be in 2035?"

If the evidence only contains the current
requirement:

label = "unsupported"


"What percentage of F-1 students eventually
receive OPT after CPT?"

If the evidence contains no such statistic:

label = "unsupported"


"Which employer currently hires the most
graduate co-op students?"

If the evidence does not provide employer
ranking data:

label = "unsupported"


IMPORTANT:

A question is NOT unsupported just because
the statement in the question is false.

If the evidence directly disproves the
statement:

label = "sufficient"


==================================================
DECISION PROCEDURE
==================================================

Follow this order carefully.

STEP 1:

Determine whether the question asks about a
SPECIFIC person's eligibility, a SPECIFIC job,
a SPECIFIC position, or another specific
situation.

Ask:

Are required facts about that situation
missing?

If YES:

label = "needs_context"

Stop.


STEP 2:

Otherwise determine whether the retrieved
evidence directly provides the requested
fact, rule, requirement, threshold, deadline,
or prerequisite.

Also determine whether the evidence directly
ENTAILS or CONTRADICTS a claim made in the
question.

If YES:

label = "sufficient"

Stop.


STEP 3:

If the information requested is not
established by the evidence:

label = "unsupported"


==================================================
IMPORTANT GUARDRAILS
==================================================

- Relevant evidence is NOT automatically
  sufficient.

- Topic similarity is NOT enough.

- Do not use outside knowledge.

- Do not infer facts absent from the supplied
  evidence.

- Do not assume personal eligibility.

- Do not assume a specific job qualifies.

- Do not predict future policies or outcomes.

- Do not invent additional conditions that
  are not part of the question.

- Do not introduce populations that the
  question did not ask about.

- General policy questions are not personal
  eligibility questions.

- Evidence that disproves a false premise is
  sufficient evidence.

- Judge only from the supplied evidence.


==================================================
CONFIDENCE
==================================================

Return a confidence value between 0 and 1.

Confidence represents how certain you are
that the LABEL is correct.

Typical values:

0.95 = very clear classification

0.85 = strong classification

0.70 = somewhat uncertain

0.55 = ambiguous classification


Do NOT automatically return zero.

Do NOT copy a fixed confidence value simply
because it appears in these instructions.

Select confidence based on how clearly the
question and evidence support the chosen
label.


==================================================
QUESTION
==================================================

{question}


==================================================
RETRIEVED EVIDENCE
==================================================

{evidence_text}


==================================================
OUTPUT
==================================================

Return ONLY one valid JSON object.

Required keys:

label
confidence
reason

The label must be exactly one of:

sufficient
needs_context
unsupported

The confidence must be a number between
0 and 1.

The reason must be one short sentence.

Do not include markdown.

Do not include any text before the JSON.

Do not include any text after the JSON.
"""

    raw_response = generate(
        prompt
    )

    cleaned = _clean_json_text(
        raw_response
    )

    try:
        result = json.loads(
            cleaned
        )

    except Exception:
        return _fallback_result()

    label = str(
        result.get(
            "label",
            "",
        )
    ).strip().lower()

    if label not in VALID_LABELS:
        return _fallback_result()

    confidence = _parse_confidence(
        result.get(
            "confidence",
            0.0,
        )
    )

    reason = str(
        result.get(
            "reason",
            "",
        )
    ).strip()

    if not reason:
        reason = (
            "No explanation was provided."
        )

    return {
        "label":
            label,

        "sufficient":
            label == "sufficient",

        "confidence":
            confidence,

        "reason":
            reason,
    }