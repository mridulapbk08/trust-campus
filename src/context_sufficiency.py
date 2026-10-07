import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_context_requirements(question: str) -> dict:
    q = normalize(question)

    missing_context = []
    future_or_unknown = False

    # -------------------------------------------------
    # Ambiguous course-drop questions
    # -------------------------------------------------
    # "How many days do I have to drop it?"
    # Missing whether this is full-semester vs half/third term.
    if (
        "drop it" in q
        or "days do i have to drop" in q
    ):
        missing_context.append(
            "course_type_or_term_length"
        )

    # -------------------------------------------------
    # Personal co-op eligibility
    # -------------------------------------------------
    # "Am I eligible for co-op?"
    # Requires user-specific GPA, completed credits,
    # enrollment/program status, etc.
    if re.search(
        r"\bam i eligible (for )?co-?op\b",
        q,
    ):
        missing_context.extend(
            [
                "completed_credits",
                "gpa",
                "enrollment_status",
                "program_requirements",
            ]
        )

    # -------------------------------------------------
    # Generic remote-work question
    # -------------------------------------------------
    # "Can I work remotely?"
    # Missing whether question concerns co-op/CPT,
    # required academic term, modality, etc.
    if re.fullmatch(
        r"can i work remotely\??",
        q,
    ):
        missing_context.extend(
            [
                "employment_context",
                "academic_term_status",
            ]
        )

    # -------------------------------------------------
    # Generic work-duration question
    # -------------------------------------------------
    # "How long can I work?"
    # Could refer to CPT, co-op, part-time/full-time, etc.
    if re.fullmatch(
        r"how long can i work\??",
        q,
    ):
        missing_context.append(
            "employment_type"
        )

    # -------------------------------------------------
    # Generic class + work question
    # -------------------------------------------------
    # "Can I take classes while working?"
    # Missing whether this is full-time co-op,
    # part-time co-op, CPT, etc.
    if re.fullmatch(
        r"can i take classes while working\??",
        q,
    ):
        missing_context.append(
            "work_or_coop_status"
        )

    # -------------------------------------------------
    # Future / unknowable information
    # -------------------------------------------------

    explicit_future_year = re.search(
        r"\b20(3[0-9]|4[0-9]|5[0-9])\b",
        q,
    )

    if explicit_future_year:
        future_or_unknown = True

    future_phrases = [
        "next year",
        "in the future",
        "will be",
        "will receive",
        "will hire",
        "will cost",
    ]

    if any(
        phrase in q
        for phrase in future_phrases
    ):
        future_or_unknown = True

    # -------------------------------------------------
    # IMPORTANT:
    # Do NOT mark policy-specific "Can I" questions
    # as ambiguous just because they are first person.
    #
    # Examples that should remain answerable:
    #
    # "Can I work on CPT before OGS approves it?"
    # "If I change employer or work location during CPT..."
    # -------------------------------------------------

    context_complete = (
        len(missing_context) == 0
        and not future_or_unknown
    )

    context_score = (
        1.0
        if context_complete
        else 0.50
    )

    return {
        "context_score":
            context_score,

        "context_complete":
            context_complete,

        "missing_context":
            missing_context,

        "future_or_unknown":
            future_or_unknown,
    }


def context_sufficiency_score(
    question: str,
) -> dict:
    return detect_context_requirements(
        question
    )