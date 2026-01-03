# src/rag_llm.py
# Strict triage-style RAG generation for Ollama (best local version)
# Goals:
# - Reduce repetition + templated outputs
# - Stay grounded in retrieved context
# - Avoid diagnosis + disease-name listing unless user explicitly asks
# - Ask up to 3 targeted questions when key details are missing
# - Robust headings parsing + deterministic formatting
# - More natural, friendly tone without losing safety

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import re
import textwrap
import ollama


# -----------------------------
# Config
# -----------------------------
DEFAULT_MAX_CHUNKS = 6
DEFAULT_MAX_CHARS = 3500

HEADINGS = [
    "Home care steps (next 24 hours)",
    "What to monitor (warning signs)",
    "When to contact a veterinarian",
    "Questions to clarify (max 3)",
]

GENERIC_FOLLOWUP_PATTERNS = [
    r"^what should i do\??$",
    r"^what do i do\??$",
    r"^what now\??$",
    r"^next steps\??$",
    r"^help\??$",
    r"^now what\??$",
    r"^i am in a rural area.*$",
    r"^i can(?:not|'t) reach (?:a )?vet.*$",
    r"^can't reach (?:a )?vet.*$",
]

URGENT_TERMS = [
    "immediately", "emergency", "urgent",
    "difficulty breathing", "trouble breathing", "labored breathing",
    "open-mouth breathing", "blue gums", "pale gums",
    "collapse", "seizure", "unconscious",
    "not breathing", "bleeding", "blood"
]


# -----------------------------
# Helpers
# -----------------------------
def _build_context(chunks: List[str], max_chars: int = DEFAULT_MAX_CHARS, max_chunks: int = DEFAULT_MAX_CHUNKS) -> str:
    clean = [c.strip() for c in (chunks or []) if c and c.strip()]
    joined = "\n\n---\n\n".join(clean[:max_chunks])
    return joined[:max_chars]


def _extract_species_from_prompt(text: str) -> Optional[str]:
    """
    Extract species from the prompt if present, e.g. 'Species: dog' or 'Species: cat'.
    This works because chat_app includes 'Pet profile: - Species: ...'
    """
    t = (text or "").lower()
    m = re.search(r"\bspecies\s*:\s*(cat|dog)\b", t)
    if m:
        return m.group(1)
    return None


def _user_asked_for_diagnosis(query: str) -> bool:
    q = (query or "").lower()
    triggers = [
        "what is it",
        "what could it be",
        "what might it be",
        "possible causes",
        "possible reason",
        "what disease",
        "which disease",
        "diagnose",
        "diagnosis",
        "is this",
        "could this be",
        "what do you think it is",
    ]
    return any(t in q for t in triggers)


def _is_generic_followup(query: str) -> bool:
    q = (query or "").strip().lower()
    return any(re.match(p, q) for p in GENERIC_FOLLOWUP_PATTERNS)


def _normalize_bullets(text: str) -> str:
    lines = (text or "").splitlines()
    out = []
    for ln in lines:
        s = ln.rstrip()
        if re.match(r"^\s*[-*]\s+", s):
            s = re.sub(r"^\s*[-*]\s+", "• ", s)
        elif re.match(r"^\s*\d+\)\s+", s):
            s = re.sub(r"^\s*\d+\)\s+", "• ", s)
        out.append(s)
    return "\n".join(out).strip()


def _force_one_bullet_per_line(text: str) -> str:
    t = (text or "").replace("\r\n", "\n")

    # If model prints "Heading • bullet" on one line, split it
    for h in HEADINGS:
        t = re.sub(rf"(?m)^\s*{re.escape(h)}\s*•\s*", f"{h}\n• ", t)

    # Ensure every bullet starts on a new line
    t = re.sub(r"[ \t]*•[ \t]*", "\n• ", t)

    # Clean excessive newlines
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _ensure_disclaimer(text: str) -> str:
    t = (text or "").strip()

    # Collapse duplicate disclaimers at end
    t = re.sub(r"(?is)(\n\s*This is not medical advice\.\s*){2,}$", "\nThis is not medical advice.", t).strip()

    if "not medical advice" not in t.lower():
        t = t.rstrip() + "\n\nThis is not medical advice."
    if not t.endswith("This is not medical advice."):
        t = t.rstrip() + "\nThis is not medical advice."
    return t


def _strip_sources_or_extras(text: str) -> str:
    t = text or ""
    # Remove "Sources:" blocks if the model prints them
    t = re.sub(r"(?is)\n\s*Sources\s*:\s*.*$", "", t).strip()

    # Remove common ChatGPT-ish prefaces
    t = re.sub(r"(?is)^\s*here['’]s (the )?(rewritten )?output.*?\n+", "", t).strip()
    t = re.sub(r"(?is)^\s*here['’]s the output.*?\n+", "", t).strip()
    t = re.sub(r"(?is)^\s*output\s*:\s*\n+", "", t).strip()
    return t


def _score_urgent(bullet: str) -> int:
    b = (bullet or "").lower()
    return sum(1 for term in URGENT_TERMS if term in b)


def _parse_or_coerce_headings(text: str) -> str:
    """
    Ensure output has all required headings in order.
    If headings are missing, coerce by bucketing bullet lines.
    """
    t = (text or "").strip()
    t = _strip_sources_or_extras(t)
    t = _normalize_bullets(t)
    t = _force_one_bullet_per_line(t)

    lower = t.lower()
    if all(h.lower() in lower for h in HEADINGS):
        # Canonicalize heading lines (best-effort)
        for h in HEADINGS:
            t = re.sub(rf"(?im)^\s*{re.escape(h)}\s*$", h, t)
        return t.strip()

    bullets = [ln.strip() for ln in t.splitlines() if ln.strip().startswith("• ")]

    # Prefer question-like bullets for the Questions section
    q_like = [b for b in bullets if b.endswith("?")]
    questions = q_like[:3] if q_like else []

    remaining = [b for b in bullets if b not in questions]

    # Urgent bullets should go to "When to contact a veterinarian"
    remaining_sorted = sorted(remaining, key=_score_urgent, reverse=True)
    urgent = [b for b in remaining_sorted if _score_urgent(b) > 0][:4]
    remaining2 = [b for b in remaining_sorted if b not in urgent]

    # Simple allocation
    home = remaining2[:4]
    monitor = remaining2[4:8]
    contact = urgent[:3] if urgent else remaining2[8:11]

    if not questions:
        questions = ["• What is your pet’s age and how long has this been happening?"]

    def _section(title: str, items: List[str]) -> str:
        if not items:
            return f"{title}\n• (Not enough information from the knowledge base.)"
        return title + "\n" + "\n".join(items[:4])

    out = "\n\n".join([
        _section(HEADINGS[0], home),
        _section(HEADINGS[1], monitor),
        _section(HEADINGS[2], contact),
        f"{HEADINGS[3]}\n" + "\n".join(questions[:3]),
    ]).strip()

    return out


def _missing_core_details(query: str) -> bool:
    q = (query or "").lower()

    has_species = bool(re.search(r"\bspecies\s*:\s*(cat|dog)\b", q)) or (" cat" in q) or (" dog" in q)
    has_time = any(x in q for x in ["today", "yesterday", "days", "hours", "week"])
    has_symptom = any(x in q for x in [
        "vomit", "vomiting", "diarr", "breath", "breathing", "cough", "sneez",
        "not eat", "not eating", "appetite", "not drinking", "letharg", "pain"
    ])

    return not (has_species and has_time and has_symptom)


def _build_prompts(query: str, context_text: str) -> Tuple[str, str]:
    asked_diagnosis = _user_asked_for_diagnosis(query)
    generic_followup = _is_generic_followup(query)
    missing = _missing_core_details(query)

    system_prompt = (
        "You are a careful, safety-first veterinary support assistant for pet owners. "
        "You must stay grounded in the retrieved context. "
        "You must be calm, friendly, and practical. "
        "You must avoid repeating the same phrases. "
        "You must not diagnose or speculate. "
        "You must not list diseases or conditions unless the user explicitly asks what it could be. "
        "Use bullet points only."
    )

    extra_rules: List[str] = []

    # Species consistency rule
    known_species = _extract_species_from_prompt(query)
    if known_species in ("cat", "dog"):
        extra_rules.append(
            f"- Species is known: {known_species}. Do NOT ask “cat or dog?”. "
            f"Do NOT refer to the pet as the other species."
        )

    # Tone
    extra_rules.append("- Start with ONE short acknowledgement sentence (friendly, not dramatic).")

    # Missing detail handling
    if missing:
        extra_rules.append("- Key details are missing. Use the 'Questions to clarify' section heavily and keep other sections to 1–2 bullets max.")

    # Diagnosis control
    if not asked_diagnosis:
        extra_rules.append("- Do NOT mention disease names or diagnoses.")
    else:
        extra_rules.append(
            "- If the user asked what it could be: you may mention up to 2 possible causes ONLY if explicitly present in the retrieved context. "
            "Label them as 'possible' and do not sound certain."
        )

    # Generic follow-up behavior
    if generic_followup:
        extra_rules.append("- The user question is generic. Do NOT repeat full generic advice. Ask 2–3 targeted clarifying questions instead.")

    user_prompt = textwrap.dedent(f"""
    User message:
    {query}

    Retrieved context (authoritative; do not go beyond it):
    {context_text}

    Output with exactly these headings (in this order) and bullet points only:

    {HEADINGS[0]}
    {HEADINGS[1]}
    {HEADINGS[2]}
    {HEADINGS[3]}

    Style rules:
    - Bullet points only, no paragraphs
    - The first bullet under Home care must be a short acknowledgement sentence
    - Avoid repeating the same wording across sections
    - Prefer short, concrete actions and monitoring
    - If key details are missing, ask questions instead of guessing
    - Limit to 4 bullets per section (questions max 3)

    Safety rules:
    {chr(10).join(extra_rules)}

    End with: This is not medical advice.
    """).strip()

    return system_prompt, user_prompt


# -----------------------------
# Public API
# -----------------------------
def generate_answer(
    query: str,
    retrieved_chunks: List[str],
    model: str = "llama3.1",
    temperature: float = 0.2,
) -> Tuple[str, Dict]:
    """
    Generate a grounded triage-style response using Ollama.
    Returns: (text, meta)
    """
    if not retrieved_chunks:
        fallback = (
            f"{HEADINGS[0]}\n"
            "• I couldn't find enough relevant information in the knowledge base.\n\n"
            f"{HEADINGS[1]}\n"
            "• Any worsening symptoms or new red flags.\n\n"
            f"{HEADINGS[2]}\n"
            "• If symptoms persist, worsen, or you are worried, contact a veterinarian.\n\n"
            f"{HEADINGS[3]}\n"
            "• What is your pet’s species, age, and weight (if known)?\n"
            "• How long has this been happening?\n"
            "• Any vomiting, diarrhea, or breathing changes?\n\n"
            "This is not medical advice."
        )
        return fallback, {"model": model, "total_tokens": None}

    context_text = _build_context(retrieved_chunks)
    system_prompt, user_prompt = _build_prompts(query, context_text)

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": float(temperature),
                "num_ctx": 2048,
                "num_predict": 320,
                "repeat_penalty": 1.18,
                "repeat_last_n": 256,
                "top_p": 0.9,
                "top_k": 40,
            },
        )

        text_raw = resp["message"]["content"]
        text_fmt = _parse_or_coerce_headings(text_raw)
        text_fmt = _ensure_disclaimer(text_fmt)

        meta = {"model": model, "total_tokens": resp.get("eval_count", None)}
        return text_fmt, meta

    except Exception as e:
        fallback = (
            f"(LLM error: {e})\n\n"
            f"{HEADINGS[0]}\n"
            "• I couldn't generate a response right now.\n\n"
            f"{HEADINGS[1]}\n"
            "• Any worsening symptoms.\n\n"
            f"{HEADINGS[2]}\n"
            "• If you are concerned or symptoms persist, contact a veterinarian.\n\n"
            f"{HEADINGS[3]}\n"
            "• How long has this been happening?\n"
            "• Any vomiting, diarrhea, or breathing changes?\n"
            "• Is your pet drinking normally?\n\n"
            "This is not medical advice."
        )
        return fallback, {"model": model, "total_tokens": None}
