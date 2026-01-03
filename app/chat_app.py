# app/chat_app.py
# Pet Symptom Chatbot, cleaned version
# Chunk level hybrid retrieval + reranker + Ollama generation
# English only
# Remembers user name and pet name from chat
# Optional nearby vet lookup via src/vet_locator.py

from __future__ import annotations

import os
import sys
import re
import pickle
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import streamlit as st
import faiss
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import linear_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_llm_gemma2 import generate_answer, HEADINGS

try:
    from src.vet_locator import geocode_location, find_vets_nearby
    VET_LOCATOR_AVAILABLE = True
except Exception:
    VET_LOCATOR_AVAILABLE = False


# -----------------------------
# Page + Theme
# -----------------------------
st.markdown(
    """
    <div style="
      display:flex;
      align-items:center;
      justify-content:center;
      gap:16px;
      margin-top:10px;
      margin-bottom:10px;
      position: relative;
      left: -40px;
    ">
      <img src="http://dl6.glitter-graphics.net/pub/1086/1086686ns3bdfgjh1.gif"
           width="58" height="52"
           style="image-rendering: pixelated;"/>
      <div>
        <div style="font-size:44px; font-weight:800; line-height:1.05; margin:0;">
          Pet Symptom Chatbot
        </div>
        <div style="opacity:0.75; margin-top:6px;">
          Evidence grounded support for your pet’s symptoms (not a diagnosis).
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


def load_css(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css(os.path.join(os.path.dirname(__file__), "styles.css"))


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    model_options = ["gemma2:9b", "llama3.1", "mistral", "llama3.2:3b"]
    model_name = st.selectbox("LLM model", model_options, index=0)
    use_llm = st.toggle("Use LLM", value=True)
    llm_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Top k chunks", 3, 10, 6, 1)
    pool = st.slider("Candidate pool", 30, 150, 80, 10)
    min_score = st.slider("Min relevance", 0.00, 0.30, 0.10, 0.01)

    st.divider()
    st.subheader("Reranker")
    use_reranker = st.toggle("Use reranker", value=True)
    rerank_pool = st.slider("Rerank pool size", 10, 80, 40, 5)
    rerank_pool = min(rerank_pool, pool)

    show_debug = st.toggle("Show debug retrieval", value=False)

    st.divider()
    st.subheader("Safety")
    show_alerts = st.toggle("Show safety alerts", value=True)

    st.divider()
    st.subheader("Nearby vets")
    if VET_LOCATOR_AVAILABLE:
        location_text = st.text_input(
            "Your location (city + district works best)",
            value="",
            help="Examples: 'Istanbul Kadikoy Turkey', 'Izmir Bornova Turkey', 'Ankara Cankaya Turkey'. You can also paste a full address."
        )
        radius_km = st.slider("Search radius (km)", 5, 80, 25, 5)
        show_vets = st.toggle("Show nearby veterinary clinics (every answer)", value=False)
        auto_vets_on_alert = st.toggle("Auto-show vets on urgent alerts", value=True)
    else:
        st.info("Vet locator is disabled (src/vet_locator.py not found).")
        location_text, radius_km, show_vets, auto_vets_on_alert = "", 25, False, False


# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    index = faiss.read_index("models/faiss_index.bin")
    titles = pickle.load(open("models/titles.pkl", "rb"))
    urls = pickle.load(open("models/urls.pkl", "rb"))
    chunks = pickle.load(open("models/chunks.pkl", "rb"))
    doc_emb = np.load("models/doc_embeddings.npy").astype("float32")

    tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    tfidf_mat = sp.load_npz("models/tfidf_matrix.npz")

    return embed_model, reranker, index, titles, urls, chunks, doc_emb, tfidf, tfidf_mat


embed_model, reranker, index, titles, urls, chunks, doc_emb, tfidf, tfidf_mat = load_artifacts()


# -----------------------------
# Rules + Helpers
# -----------------------------
RULES = [
    {"pattern": r"not breathing|unconscious|collapse|seizure|poison|toxin",
     "message": "⚠️ This may be an emergency. Seek veterinary help immediately."},
    {"pattern": r"blood in (his|her|the)?\s*(stool|poop)|bloody stool|blood in poop|vomit(ing)?\s*blood|bloody vomit|black (stool|poop)|tarry stool",
     "message": "⚠️ Blood in stool or vomit (or black tarry stool) can be urgent. Contact a veterinarian today."},
    {"pattern": r"difficulty breathing|trouble breathing|labored breathing|open-mouth breathing|blue gums|pale gums",
     "message": "⚠️ Breathing or gum color changes can be urgent. Contact a veterinarian now."},
    {"pattern": r"vomit|vomiting|diarrhea|lethargy|not eating|loss of appetite|fever|dehydration",
     "message": "If symptoms persist longer than 24 hours or worsen, contact a veterinarian."}
]

DERM_KW = {
    "red spot", "red spots", "spots", "rash", "hives", "welts",
    "itch", "itchy", "itching", "scratching", "chewing", "licking",
    "skin", "hot spot", "hair loss", "bald", "scab", "crust", "flaky", "dandruff",
    "ear", "ears", "ear discharge", "ear wax", "ear odor",
    "flea", "fleas", "tick", "ticks", "mite", "mites", "mange",
    "swelling"
}
GI_KW = {
    "vomit", "vomiting", "vomited", "puke", "nausea",
    "diarrhea", "diarrhoea", "loose stool", "stool", "poop",
    "constipation", "blood in stool", "bloody stool", "black stool", "tarry stool",
    "bloody vomit", "vomiting blood", "dehydration"
}


def classify_track(text: str) -> str:
    t = (text or "").lower()
    derm = any(k in t for k in DERM_KW)
    gi = any(k in t for k in GI_KW)
    if derm and not gi:
        return "derm"
    if gi and not derm:
        return "gi"
    if derm and gi:
        return "general"
    return "general"


def _set_track_from_context(ctx: str):
    if st.session_state.get("case_active", False) and st.session_state.get("symptom_track"):
        return
    st.session_state.symptom_track = classify_track(ctx)


def has_symptom_signal(text: str) -> bool:
    t = (text or "").lower()
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    symptom_patterns = [
        r"\bdoes\s+not\s+eat\b",
        r"\bdoesn['’]t\s+eat\b",
        r"\bwon['’]t\s+eat\b",
        r"\bnot\s+eating\b",
        r"\bnot\s+eat(ing)?\b",
        r"\beat\s+less\b",
        r"\bpoor\s+appetite\b",
        r"\bloss\s+of\s+appetite\b",
        r"\bletharg(ic|y)\b",
        r"\bdiarrh(ea|oea)\b",
        r"\bvomit(ing|ed)?\b",
        r"\bblood\b",
        r"\bskin\b",
        r"\brash\b",
        r"\bitch(y|ing)?\b",
        r"\bred\s+spots?\b",
        r"\bhives?\b",
        r"\bswelling\b",
        r"\bhair\s+loss\b",
        r"\bhot\s+spot(s)?\b",
        r"\bscab(s)?\b",
        r"\bcrust(y|ing)?\b",
        r"\bflak(e|y|ing)\b",
        r"\bdandruff\b",
        r"\bflea(s)?\b",
        r"\btick(s)?\b",
        r"\bmite(s)?\b",
        r"\bmange\b",
    ]
    if any(re.search(p, t) for p in symptom_patterns):
        return True

    symptom_keywords = [
        "symptom", "sick", "ill", "hurt", "pain", "weak", "tired", "low energy", "fatigue", "letharg",
        "appetite", "eat", "eats", "eating",
        "drink", "drinking", "water", "dehydration",
        "vomit", "vomiting", "puke", "nausea",
        "diarrhea", "diarrhoea", "loose stool", "stool", "poop", "constipation",
        "cough", "sneeze", "wheeze", "breath", "breathing", "pant",
        "fever", "blood", "bleeding", "seizure", "collapse", "unconscious",
        "itch", "itchy", "rash", "red spots", "spots", "hives", "swelling",
        "skin", "ear", "ears", "odor", "smell", "discharge",
        "hair loss", "flaky", "dandruff", "scab", "crust", "hot spot",
        "flea", "fleas", "tick", "ticks", "mite", "mites", "mange",
        "urine", "pee", "peeing", "straining", "can't pee",
        "limp", "limping", "can't walk", "won't walk",
        "runny nose", "nasal discharge", "sneezing",
    ]
    return any(k in t for k in symptom_keywords)


def _set_main_symptom_if_empty(text: str):
    if st.session_state.get("main_symptom"):
        return
    if has_symptom_signal(text):
        st.session_state.main_symptom = (text or "").strip()[:240]


def is_skin_case(text: str) -> bool:
    t = (text or "").lower()
    skin_hits = [
        "skin", "rash", "itch", "itchy", "red spot", "red spots", "spots",
        "ear", "ears", "hot spot", "hives", "swelling", "hair loss",
        "scab", "crust", "flaky", "dandruff",
        "flea", "fleas", "tick", "ticks", "mite", "mites", "mange",
    ]
    return any(k in t for k in skin_hits)


def scrub_unstated_diagnoses(answer: str, user_ctx: str) -> str:
    a = (answer or "")
    ctx = (user_ctx or "").lower()
    if any(k in ctx for k in ["diagnosed", "vet said", "heart murmur", "murmur", "grade i", "grade ii", "grade iii", "grade iv", "grade v", "grade vi"]):
        return a

    bad = re.compile(r"(?i)\b(diagnosed|has been diagnosed|heart murmur|murmur grade)\b")
    lines = a.splitlines()
    kept = [ln for ln in lines if not bad.search(ln)]
    out = "\n".join(kept).strip()

    if "this is not medical advice" not in out.lower():
        out += "\n\nThis is not medical advice."
    return out


def looks_like_profile_only(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if has_symptom_signal(t):
        return False
    profile_patterns = [
        r"^(hello|hi|hey)\b.*$",
        r"^my name is\b.*$",
        r"^i am\b.*$",
        r"^i'm\b.*$",
        r"^my (cat|dog|pet)('s)? name is\b.*$",
        r"^my (cat|dog) is named\b.*$",
        r"^my pet is a (cat|dog)\b.*$",
        r"^his name is\b.*$",
        r"^her name is\b.*$",
    ]
    return any(re.match(p, t) for p in profile_patterns)


def sanitize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w' -]", "", name, flags=re.UNICODE)
    name = re.sub(r"_", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    parts = name.split()
    if parts and parts[-1].lower() in {"and"}:
        parts = parts[:-1]
    name = " ".join(parts).strip()
    return name[:32]


def extract_user_name(text: str) -> Optional[str]:
    t = (text or "").strip()
    stop = r"(?=\s*(?:,|$|\band\b|\bhe\b|\bshe\b|\bit\b|\bmy\b|\bmy\s+(?:cat|dog|pet)\b))"
    name_pat = r"([^\d\W_][^\d\W_' -]{0,31}(?:[ ' -][^\d\W_][^\d\W_' -]{0,31}){0,2})"

    patterns = [
        rf"(?i)^\s*(?:hello|hi|hey)\s*(?:,|\s)*i\s*am\s+{name_pat}{stop}.*$",
        rf"(?i)^\s*(?:hello|hi|hey)\s*(?:,|\s)*my\s*name\s*is\s+{name_pat}{stop}.*$",
        rf"(?i)^\s*my\s*name\s*is\s+{name_pat}{stop}.*$",
        rf"(?i)^\s*i\s*am\s+{name_pat}{stop}.*$",
        rf"(?i)^\s*i'm\s+{name_pat}{stop}.*$",
    ]

    for p in patterns:
        m = re.match(p, t, flags=re.UNICODE)
        if m:
            return sanitize_name(m.group(1))
    return None


def extract_pet_info(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = (text or "").strip()

    species = None
    if re.search(r"(?i)\bdog\b", t):
        species = "dog"
    if re.search(r"(?i)\bcat\b", t):
        species = "cat"

    name_pat = r"([A-Za-zÀ-ÖØ-öø-ÿİıĞğŞşÇçÖöÜü][A-Za-zÀ-ÖØ-öø-ÿİıĞğŞşÇçÖöÜü' -]{0,31})"
    stop = r"(?=\s*(?:,|\.|!|\?|$|\bhe\b|\bshe\b|\bit\b|\bis\b|\band\b|\bwho\b|\bthat\b|\bwhich\b|\b(?:a|an|the)\b|\bcat\b|\bdog\b))"

    pet_patterns = [
        rf"(?i)\bmy\s+(?:cat|dog)\s+(?:is\s+named|named)\s+{name_pat}{stop}",
        rf"(?i)\bmy\s+(?:cat|dog)'?s\s+name\s+is\s+{name_pat}{stop}",
        rf"(?i)\bmy\s+pet'?s\s+name\s+is\s+{name_pat}{stop}",
        rf"(?i)\bhis\s+name\s+is\s+{name_pat}{stop}",
        rf"(?i)\bher\s+name\s+is\s+{name_pat}{stop}",
        rf"(?i)\bpet(?:'s)?\s+name\s+is\s+{name_pat}{stop}",
        rf"(?i)\bmy\s+(?:pet\s+)?(?:cat|dog)\s+is\s+{name_pat}{stop}",
        rf"(?i)\bmy\s+pet\s+(?:cat|dog)'?s\s+name\s+is\s+{name_pat}{stop}\b",
        rf"(?i)\bmy\s+(?:cat|dog)\s+{name_pat}{stop}\b",
    ]

    pet_name = None
    for p in pet_patterns:
        m = re.search(p, t, flags=re.UNICODE)
        if m:
            pet_name = sanitize_name(m.group(1))
            break

    return pet_name, species


def update_memory_from_text(text: str) -> dict:
    changed = {"user": False, "pet": False, "species": False}

    user_name = extract_user_name(text)
    if user_name and user_name != st.session_state.get("user_name", ""):
        st.session_state.user_name = user_name
        changed["user"] = True

    pet_name, pet_species = extract_pet_info(text)
    if pet_name and pet_name != st.session_state.get("pet_name", ""):
        st.session_state.pet_name = pet_name
        changed["pet"] = True

    if pet_species and pet_species != st.session_state.get("pet_species", ""):
        st.session_state.pet_species = pet_species
        changed["species"] = True

    return changed


def user_display() -> str:
    name = sanitize_name(st.session_state.get("user_name", ""))
    return name if name else "friend"


def pet_display() -> str:
    pet = sanitize_name(st.session_state.get("pet_name", ""))
    species = (st.session_state.get("pet_species", "") or "").strip().lower()

    if pet and species in ("cat", "dog"):
        return f"your {species} {pet}"
    if pet:
        return pet
    if species in ("cat", "dog"):
        return f"your {species}"
    return "your pet"


def build_ack(mem_changed: dict) -> str:
    if "memory_ack" not in st.session_state:
        st.session_state.memory_ack = {"pet_name": ""}

    lines = []
    current_pet = sanitize_name(st.session_state.get("pet_name", ""))

    if mem_changed.get("pet") and current_pet and st.session_state.memory_ack.get("pet_name", "") != current_pet:
        st.session_state.memory_ack["pet_name"] = current_pet
        lines.append(f"I’ll remember {current_pet}.")

    return ("\n\n".join(lines) + "\n\n") if lines else ""


def detect_intent(text: str) -> str:
    t = (text or "").lower().strip()
    if re.search(r"\b(hi|hello|hey)\b", t):
        return "greeting"
    if re.search(r"\b(thanks|thank you)\b", t):
        return "thanks"
    if re.search(r"(what (do i|should i) do next|what next|next steps|what should i do)", t):
        return "followup"
    return "symptom"


def check_rules(text: str) -> List[str]:
    alerts, seen = [], set()
    for r in RULES:
        if re.search(r["pattern"], text, flags=re.I) and r["message"] not in seen:
            alerts.append(r["message"])
            seen.add(r["message"])
    return alerts


def is_generic_followup(text: str) -> bool:
    t = (text or "").lower().strip()
    patterns = [
        r"^what should i do\??$",
        r"^what do i do\??$",
        r"^what now\??$",
        r"^next steps\??$",
        r"^help\??$",
        r"^now what\??$",
        r"^should i go to (the )?(vet|veterinary)\??$",
        r"^do i need to go to (the )?(vet|veterinary)\??$",
        r"^should i see a vet\??$",
        r"^do i need a vet\??$",
    ]
    return any(re.match(p, t) for p in patterns)


def get_symptom_context() -> str:
    msgs = [m for role, m in st.session_state.chat if role == "user"]
    symptom_msgs = [m for m in msgs if has_symptom_signal(m)]
    return ". ".join(symptom_msgs[-3:]).strip()


def unique_sources(docs: List[dict], limit: int = 3) -> List[dict]:
    seen, out = set(), []
    for d in docs:
        url = d.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(d)
        if len(out) >= limit:
            break
    return out


def _species_ok(url: str, species: str) -> bool:
    if not species:
        return True
    u = (url or "").lower()
    s = species.lower().strip()
    if s == "cat":
        return "/cat/" in u
    if s == "dog":
        return "/dog/" in u
    return True

def is_urgent_alert(alerts: List[str]) -> bool:
    if not alerts:
        return False
    t = " ".join(alerts).lower()
    return ("emergency" in t) or ("urgent" in t) or ("contact a veterinarian now" in t)

def asks_for_nearby_vets(text: str) -> bool:
    t = (text or "").lower().strip()
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t)

    patterns = [
        r"\bnearest\s+vet\b",
        r"\bnearby\s+vet(s)?\b",
        r"\bvet\s+near\s+me\b",
        r"\bveterinary\s+near\s+me\b",
        r"\bvet\s+nearby\b",
        r"\bfind\s+vet(s)?\b",
        r"\bvet\s+clinic(s)?\s+near\s+me\b",
        r"\bclosest\s+vet\b",
    ]
    return any(re.search(p, t) for p in patterns)


def extract_location_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"(?i)^\s*(i am in|i'm in|im in|we are in|we're in|located in)\s+", "", t).strip()
    t = t.replace("Türkiye", "Turkey").replace("TÜRKİYE", "Turkey")
    t = t.replace("türkiye", "Turkey").replace("turkiye", "Turkey")
    t = t.replace("İ", "I").replace("ı", "i")
    return t[:120]


def looks_like_location_reply(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False
    if has_symptom_signal(t):
        return False
    return any(k in t for k in [" turkey", " türkiye", " turkiye"]) or len(t.split()) >= 2


def google_maps_search_link(
    name: str,
    addr: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None
) -> str:
    if lat is not None and lon is not None:
        return f"https://www.google.com/maps?q={lat},{lon}"

    q = " ".join([x for x in [name, addr] if x]).strip()
    q = urllib.parse.quote_plus(q)
    return f"https://www.google.com/maps/search/?api=1&query={q}"


def build_vets_md(location_text: str, radius_km: int, limit: int = 5) -> str:
    loc = (location_text or "").strip()
    if not loc:
        return ""

    def _gmaps_fallback(loc_text: str) -> str:
        q = urllib.parse.quote_plus(f"veterinary clinic near {loc_text}")
        return f"https://www.google.com/maps/search/?api=1&query={q}"

    if not VET_LOCATOR_AVAILABLE:
        gmaps = _gmaps_fallback(loc)
        return f"1. Vet lookup is not enabled in this build.\n2. Google Maps fallback: [Search vets near {loc}]({gmaps})"

    try:
        latlon = geocode_location(loc)
        if not latlon:
            gmaps = _gmaps_fallback(loc)
            return (
                "1. Location lookup failed (geocoding may be rate limited).\n"
                "2. Try simpler: 'Kadikoy Istanbul Turkey'.\n"
                f"3. Google Maps fallback: [Search vets near {loc}]({gmaps})"
            )

        lat, lon = latlon
        vets = find_vets_nearby(lat, lon, radius_km=radius_km, limit=limit)

        if not vets:
            gmaps = _gmaps_fallback(loc)
            return (
                "1. Vet lookup returned no results (Overpass may be rate limited).\n"
                "2. Increase radius (try 50 to 80 km).\n"
                "3. Try simpler: 'Kadikoy Istanbul Turkey'.\n"
                f"4. Google Maps fallback: [Search vets near {loc}]({gmaps})"
            )

        lines = []
        for i, v in enumerate(vets, start=1):
            name = (v.get("name") or "Vet clinic").strip()
            addr = (v.get("address") or "").strip()
            dist = v.get("distance_km", None)
            phone = (v.get("phone") or "").strip()
            website = (v.get("website") or "").strip()

            vlat = v.get("lat", None)
            vlon = v.get("lon", None)
            map_link = google_maps_search_link(name, addr, vlat, vlon)

            extra = [f"[Map]({map_link})"]
            if phone:
                extra.append(f"☎️ {phone}")
            if website:
                extra.append(f"🌐 {website}")
            extra_txt = " | ".join(extra).strip()

            if dist is not None:
                base = f"{i}. {name} ({float(dist):.1f} km) {addr}".strip()
            else:
                base = f"{i}. {name} {addr}".strip()

            lines.append(f"{base} {extra_txt}".strip())

        return "\n".join(lines).strip()

    except Exception:
        gmaps = _gmaps_fallback(loc)
        return (
            "1. Vet lookup failed.\n"
            "2. Try a simpler location (example: 'Kadikoy Istanbul Turkey').\n"
            f"3. Google Maps fallback: [Search vets near {loc}]({gmaps})"
        )


def build_vet_offer_line() -> str:
    return (
        "\n\nIf you want, I can list nearby veterinary clinics. "
        "Say “nearest vet”, or tell me your city + district + country."
    )


# -----------------------------
# Track aware safe fallback answers
# -----------------------------
def _safe_fallback_answer_derm(u: str, p: str, alerts: List[str]) -> str:
    warning = ("\n".join(alerts) + "\n\n") if alerts else ""
    return (
        f"{warning}"
        f"Hi {u}. Thanks for telling me about {p}. I’m here with you.\n\n"
        f"{HEADINGS[0]}\n"
        "1. Keep your pet comfortable and prevent scratching (a cone can help if they are chewing or licking).\n"
        "2. Avoid new shampoos, sprays, or treats until things settle.\n"
        "3. If there is mild itch, a lukewarm oatmeal bath can help short term.\n\n"
        f"{HEADINGS[1]}\n"
        "1. Facial swelling, hives spreading quickly, trouble breathing, vomiting or diarrhea, collapse, or severe lethargy.\n"
        "2. Pus, foul odor, hot painful skin, or rapidly expanding redness.\n\n"
        f"{HEADINGS[2]}\n"
        "1. If spots are spreading, your pet is less active, not drinking, or appetite is reduced, contact a veterinarian within 24 hours (sooner if worsening).\n\n"
        f"{HEADINGS[3]}\n"
        "1. Are the spots raised like hives, or flat?\n"
        "2. Is your pet very itchy or chewing or licking a lot?\n\n"
        "This is not medical advice."
    )


def _safe_fallback_answer_gi(u: str, p: str, alerts: List[str]) -> str:
    warning = ("\n".join(alerts) + "\n\n") if alerts else ""
    return (
        f"{warning}"
        f"Hi {u}. Thanks for telling me about {p}. I’m here with you.\n\n"
        f"{HEADINGS[0]}\n"
        "1. Keep them comfortable and quiet.\n"
        "2. Offer small sips of water frequently.\n"
        "3. Avoid rich treats. Consider a bland diet in small amounts if they can keep water down.\n\n"
        f"{HEADINGS[1]}\n"
        "1. Repeated vomiting or diarrhea, worsening lethargy, refusal to drink, signs of pain, or dehydration.\n\n"
        f"{HEADINGS[2]}\n"
        "1. If symptoms worsen, persist beyond 24 hours, or you see blood in vomit or stool.\n\n"
        f"{HEADINGS[3]}\n"
        "1. About how many times in the last 24 hours?\n"
        "2. Can they keep water down?\n\n"
        "This is not medical advice."
    )


def _safe_fallback_answer(u: str, p: str, alerts: List[str]) -> str:
    track = st.session_state.get("symptom_track") or "general"
    if track == "derm":
        return _safe_fallback_answer_derm(u, p, alerts)
    if track == "gi":
        return _safe_fallback_answer_gi(u, p, alerts)
    return _safe_fallback_answer_gi(u, p, alerts)


# -----------------------------
# Output normalizer
# -----------------------------
def _renumber_bullets_by_section(text: str) -> str:
    lines = (text or "").splitlines()
    out = []
    in_section = False
    n = 0

    heading_set = set(HEADINGS)

    def is_heading(line: str) -> bool:
        return line.strip() in heading_set

    def strip_list_prefix(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^(?:\d+\.\s+|•\s+|\-\s+|\*\s+|\u2022\s+)", "", s).strip()
        return s

    def is_bullet(line: str) -> bool:
        return bool(re.match(r"^\s*(?:•|\-|\*|\u2022)\s+", line))

    def is_numbered(line: str) -> bool:
        return bool(re.match(r"^\s*\d+\.\s*", line))

    for raw in lines:
        line = raw.rstrip()

        if is_heading(line):
            out.append(line.strip())
            in_section = True
            n = 0
            continue

        if in_section and (is_bullet(line) or is_numbered(line)):
            item = strip_list_prefix(line)
            # skip empty items like "2." or "•"
            if not item:
                continue
            n += 1
            out.append(f"{n}. {item}")
            continue

        out.append(line)

    return "\n".join(out).strip()


def _force_headings_on_own_lines(t: str) -> str:
    if not t:
        return t

    for h in HEADINGS:
        h_esc = re.escape(h)

        # Split even if it appears mid line
        # Tolerate optional bold markers and optional trailing colon
        t = re.sub(
            rf"(?i)\s*\**\s*{h_esc}\s*\**\s*:?\s*",
            f"\n\n{h}\n",
            t
        )

    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _bulletize_sections(t: str) -> str:
    heading_set = set(HEADINGS)
    lines = (t or "").splitlines()
    out = []
    i = 0

    def split_sentences(s: str):
        s = re.sub(r"\s+", " ", (s or "")).strip()
        if not s:
            return []
        parts = re.split(r"(?<=[.!?])\s+", s)
        parts = [p.strip() for p in parts if p.strip()]
        # avoid turning stray numbering like "2." into bullets
        parts = [p for p in parts if not re.fullmatch(r"\d+\.?", p)]
        return parts

    def is_list_line(s: str) -> bool:
        s = s.strip()
        return bool(re.match(r"^(?:\d+\.\s+|•\s+|\-\s+|\*\s+)", s))

    while i < len(lines):
        line = lines[i].strip()

        if line in heading_set:
            out.append(line)
            i += 1

            body = []
            while i < len(lines) and (lines[i].strip() not in heading_set):
                body.append(lines[i].strip())
                i += 1

            items = []
            for b in body:
                if not b:
                    continue

                # skip pure numbering lines like "2." even if model outputs them
                if re.fullmatch(r"\d+\.\s*", b.strip()):
                    continue

                if is_list_line(b):
                    # skip empty numbered like "2. " or bullet like "• "
                    content = re.sub(r"^(?:\d+\.\s*|•\s*|\-\s*|\*\s*)", "", b).strip()
                    if not content:
                        continue
                    items.append(b)
                else:
                    for sent in split_sentences(b):
                        if sent.strip():
                            items.append("• " + sent)

            if not items:
                items = ["• Monitor closely and contact a veterinarian if symptoms worsen."]

            out.extend(items)
            out.append("")
            continue

        out.append(lines[i])
        i += 1

    return "\n".join(out).strip()


def _force_single_disclaimer_at_end(t: str) -> str:
    lines = (t or "").splitlines()
    kept = []
    for ln in lines:
        if re.search(r"(?i)\bthis is not medical advice\b", ln):
            continue
        kept.append(ln)
    t2 = "\n".join(kept).strip()
    return (t2 + "\n\nThis is not medical advice.").strip()


def _cap_questions_section(t: str, max_q: int = 3) -> str:
    parts = []
    blocks = re.split(r"(?m)^(?=" + "|".join(map(re.escape, HEADINGS)) + r")", t)
    for block in blocks:
        if not block.strip():
            continue
        lines = block.splitlines()
        if lines and lines[0].strip() == HEADINGS[3]:
            heading = lines[0].strip()
            body = []
            for ln in lines[1:]:
                # keep only non-empty numbered items
                m = re.match(r"^\s*\d+\.\s*(.*)$", ln.strip())
                if m:
                    content = (m.group(1) or "").strip()
                    if content:
                        body.append(f"1. {content}")  # temp, will renumber later
            body = body[:max_q]
            parts.append(heading + ("\n" + "\n".join(body) if body else ""))
        else:
            parts.append(block.strip())
    return "\n\n".join([p for p in parts if p.strip()]).strip()


def _remove_empty_list_items(t: str) -> str:
    t = re.sub(r"(?m)^\s*\d+\.\s*$", "", t)                 # "2."
    t = re.sub(r"(?m)^\s*(?:•|\-|\*|\u2022)\s*$", "", t)    # "•"
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _renumber_by_headings(t: str) -> str:
    heading_set = set(HEADINGS)
    lines = (t or "").splitlines()
    out = []
    n = 0
    in_section = False

    for raw in lines:
        line = raw.rstrip()
        s = line.strip()

        if s in heading_set:
            out.append(s)
            n = 0
            in_section = True
            continue

        if in_section:
            m = re.match(r"^\d+\.\s+(.*)$", s)
            if m:
                item = m.group(1).strip()
                if item:
                    n += 1
                    out.append(f"{n}. {item}")
                continue

            m = re.match(r"^(?:•|\-|\*|\u2022)\s+(.*)$", s)
            if m:
                item = m.group(1).strip()
                if item:
                    n += 1
                    out.append(f"{n}. {item}")
                continue

        out.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def scrub_diagnosis_language(answer: str) -> str:
    # removes lines that look like a diagnosis claim
    bad = re.compile(
        r"(?i)\b(might have|likely has|sounds like|suggests|consistent with)\b.*\b("
        r"infection|yeast|allergy|mites|mange|dermatitis)\b"
    )
    lines = []
    for ln in (answer or "").splitlines():
        if bad.search(ln):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def _enforce_questions_section(text: str, max_q: int = 3) -> str:
    hq = HEADINGS[3]
    heading_set = set(HEADINGS)

    lines = (text or "").splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.strip() != hq:
            out.append(line)
            i += 1
            continue

        out.append(hq)
        i += 1
        qs = []

        while i < len(lines) and lines[i].strip() not in heading_set:
            ln = lines[i].strip()
            m = re.match(r"^\d+\.\s+(.*)$", ln)
            if m:
                q = (m.group(1) or "").strip()
                if q.endswith("?"):
                    qs.append(q)
            i += 1

        qs = qs[:max_q]
        if qs:
            out.extend([f"{k+1}. {q}" for k, q in enumerate(qs)])
        else:
            out.append("1. When did this start (today, yesterday, how many hours or days)?")

        out.append("")
    return "\n".join(out).strip()


def clean_llm_output(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t

    # strip any model "Sources" section before your UI expander
    t = re.split(r"\n\s*Sources\s*\n", t, flags=re.I)[0].strip()
    t = re.split(r"\n\s*Source\s*\n", t, flags=re.I)[0].strip()

    # strip boilerplate
    t = re.sub(
        r"(?im)^[^\n]{0,160}\b(information based on|based on provided text)\b[^\n]*\n+",
        "",
        t
    ).strip()

    # normalize heading variants to your canonical HEADINGS
    heading_variants = {
        HEADINGS[0]: [r"home care steps.*24 hours.*", r"home care.*", r"care steps.*"],
        HEADINGS[1]: [r"what to monitor.*warning signs.*", r"warning signs.*", r"what to monitor.*"],
        HEADINGS[2]: [r"when to contact a veterinarian.*", r"when to contact.*vet.*", r"when to see a vet.*"],
        HEADINGS[3]: [r"questions to clarify.*", r"questions.*max.*", r"clarifying questions.*"],
    }

    lines = t.splitlines()
    norm_lines = []
    for line in lines:
        raw = line.strip()
        raw_unbold = re.sub(r"^\*{1,3}\s*", "", raw)
        raw_unbold = re.sub(r"\s*\*{1,3}$", "", raw_unbold).strip()

        replaced = None
        for canonical, patterns in heading_variants.items():
            for pat in patterns:
                if re.fullmatch(pat, raw_unbold, flags=re.I):
                    replaced = canonical
                    break
            if replaced:
                break

        norm_lines.append(replaced if replaced else line)

    t = "\n".join(norm_lines).strip()

    # main formatting fix
    t = _force_headings_on_own_lines(t)
    t = _bulletize_sections(t)

    # normalize bullets
    t = re.sub(r"(?m)^\s*[-*]\s+", "• ", t)
    t = re.sub(r"(?m)^\s*•\s+", "• ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # remove model empty items like "2." early
    t = _remove_empty_list_items(t)

    # questions: cap and enforce real question marks
    t = _cap_questions_section(t, max_q=3)
    t = _enforce_questions_section(t, max_q=3)

    # renumber once at the end so no blanks stay
    t = _renumber_bullets_by_section(t)

    # safety cleanup and disclaimer placement
    t = scrub_diagnosis_language(t)
    t = _force_single_disclaimer_at_end(t)

    return t

def clamp_answer_length(text: str, max_chars: int = 1400) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t

    parts = []
    for h in HEADINGS:
        m = re.search(
            rf"(?s)\n\n{re.escape(h)}\n(.*?)(?=\n\n(?:{re.escape(HEADINGS[0])}|{re.escape(HEADINGS[1])}|{re.escape(HEADINGS[2])}|{re.escape(HEADINGS[3])})\n|$)",
            "\n\n" + t
        )
        if m:
            body = (m.group(1) or "").strip()
            parts.append((h, body))
        else:
            parts.append((h, ""))

    out = []
    per_section_items = 4
    for h, body in parts:
        items = []
        for line in body.splitlines():
            s = line.strip()
            if re.match(r"^\d+\.\s+", s):
                items.append(s)
            elif s.startswith(("•", "*", "-")):
                items.append(s)
        items = items[:per_section_items]
        if not items:
            items = ["1. Keep this brief and monitor closely."]

        block = "\n".join(items)
        block = re.sub(r"(?m)^\s*[-*•]\s+", "1. ", block).strip()
        out.append(f"{h}\n{block}")

    t2 = "\n\n".join(out).strip()
    t2 = _renumber_bullets_by_section(t2)

    if len(t2) > max_chars:
        t2 = t2[:max_chars].rsplit("\n", 1)[0].rstrip() + "\n\nThis is not medical advice."
    return t2


def filter_to_evidence(answer: str, retrieved_chunks: List[str], sim_th: float = 0.22) -> str:
    if not retrieved_chunks:
        return answer

    chunks_ = [c.strip() for c in retrieved_chunks[:8] if (c or "").strip()]
    if not chunks_:
        return answer

    chunk_emb = embed_model.encode(chunks_, normalize_embeddings=True)

    def is_list_item(line: str) -> bool:
        s = re.sub(r"\s+", " ", line).strip()
        return bool(re.match(r"^(?:\d+\.\s+|•\s+|\-\s+|\*\s+)", s))

    def strip_item_prefix(line: str) -> str:
        s = line.strip()
        s = re.sub(r"^(?:\d+\.\s+|•\s+|\-\s+|\*\s+)", "", s).strip()
        return s

    def ok(line: str) -> bool:
        s = re.sub(r"\s+", " ", line).strip()
        if not s or not is_list_item(s):
            return True
        sent = strip_item_prefix(s)
        if len(sent) < 8:
            return True
        v = embed_model.encode([sent], normalize_embeddings=True)[0]
        sims = (chunk_emb @ v).ravel()
        return float(np.max(sims)) >= sim_th

    lines = answer.splitlines()
    kept = []
    for ln in lines:
        if ok(ln):
            kept.append(ln)

    t = "\n".join(kept).strip()

    for h in HEADINGS:
        if re.search(rf"(?m)^{re.escape(h)}\s*$", t) and not re.search(rf"(?s){re.escape(h)}\n[^\n]*(?:\d+\.\s+|•|\-|\*)", t):
            t = re.sub(
                rf"(?m)^{re.escape(h)}\s*$",
                f"{h}\n1. I’m not fully confident without more details. If symptoms worsen, contact a veterinarian.",
                t
            )

    t = _renumber_bullets_by_section(t)

    if "this is not medical advice" not in t.lower():
        t += "\n\nThis is not medical advice."
    return t


# -----------------------------
# Slot memory
# -----------------------------
NUM_WORDS = {
    "once": 1, "twice": 2, "thrice": 3,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
}


def _word_to_int(s: str) -> Optional[int]:
    if not s:
        return None
    s = s.strip().lower()
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    return NUM_WORDS.get(s)


def extract_slots(text: str, prev_slots: Optional[Dict] = None) -> Dict:
    t = (text or "").lower()
    prev = prev_slots or {}
    slots: Dict = {}

    m = re.search(r"\b(today|yesterday)\b", t)
    if m:
        slots["onset"] = m.group(1)

    m = re.search(r"\b(\d+)\s*(hour|hours|day|days|week|weeks)\b", t)
    if m:
        slots["duration"] = f"{m.group(1)} {m.group(2)}"

    m = re.search(r"\b(\d+|one|two|three|four|five)\s*(hour|hours|day|days|week|weeks)\s*ago\b", t)
    if m:
        n = _word_to_int(m.group(1))
        if n is not None:
            slots["duration"] = f"{n} {m.group(2)}"

    if re.search(r"\b(a|one)\s+week\b|\bweek\s+ago\b", t):
        slots["duration"] = "1 week"

    if re.search(r"blood in (his|her|the)?\s*(stool|poop)|bloody stool|blood in poop|black (stool|poop)|tarry stool", t):
        slots["blood_in_stool"] = "yes"
    if re.search(r"\bno blood\b|\bnot bloody\b", t):
        slots["blood_in_stool"] = "no"

    if re.search(r"\bvomit|vomited|vomiting\b", t):
        m = re.search(r"\bvomit(?:ed|ing)?\s+(once|twice|thrice|\d+)\b", t)
        if m:
            slots["vomit_count_24h"] = _word_to_int(m.group(1))
        else:
            m = re.search(r"\b(\d+|one|two|three|four|five)\s+times?\b", t)
            if m:
                slots["vomit_count_24h"] = _word_to_int(m.group(1))
            else:
                slots.setdefault("vomit_count_24h", None)
    else:
        m = re.search(r"\b(\d+|one|two|three|four|five)\s+times?\b", t)
        if m and ("vomit_count_24h" in prev):
            slots["vomit_count_24h"] = _word_to_int(m.group(1))

    if "diarr" in t or "loose stool" in t:
        slots["diarrhea"] = "yes"
        m = re.search(r"\b(\d+|one|two|three|four|five)\s+times?\b", t)
        if m:
            slots["diarrhea_count_24h"] = _word_to_int(m.group(1))
    if re.search(r"\b(no|not)\s+(diarrhea|diarrhoea|loose stool)\b", t):
        slots["diarrhea"] = "no"

    if re.search(r"\b(not eating|won't eat|doesn['’]t eat|does not eat|poor appetite|loss of appetite|eats less|reduced appetite)\b", t):
        slots["appetite"] = "reduced"
    elif re.search(r"\b(eats normally|eat normally|normal appetite|eats regular|eats regularly)\b", t):
        slots["appetite"] = "normal"

    if re.search(r"\b(not drinking|won't drink|doesn['’]t drink|does not drink)\b", t):
        slots["water_intake"] = "reduced"
    elif re.search(r"\b(drinks normally|drinking normally|normal water|drinking well|drinks water)\b", t):
        slots["water_intake"] = "normal"
    elif re.search(r"\b(had some water|drank some water|drank a little|a little water|some water)\b", t):
        slots["water_intake"] = "some"

    m = re.search(r"\b(only\s+)?drank\s+(once|twice|thrice|\d+)\b", t)
    if m:
        slots["water_times_today"] = _word_to_int(m.group(2))
        slots.setdefault("water_intake", "some")

    if re.search(r"\bletharg(ic|y)\b|\blow energy\b|\bweak\b|\btired\b", t):
        slots["energy"] = "reduced"
    elif re.search(r"\bnormal energy\b|\bactive\b", t):
        slots["energy"] = "normal"

    if re.search(r"\bkeep\s+water\s+down\b|\bkeeping\s+water\s+down\b", t):
        if re.search(r"\b(can|able to)\s+keep\s+water\s+down\b", t):
            slots["keep_water_down"] = "yes"
        if re.search(r"\b(can't|cannot|not able to)\s+keep\s+water\s+down\b", t):
            slots["keep_water_down"] = "no"

    if re.search(r"\bindoor\b", t):
        slots["indoor_outdoor"] = "indoor"
    if re.search(r"\boutdoor\b|goes outside|going outside", t):
        prev_io = slots.get("indoor_outdoor")
        slots["indoor_outdoor"] = "indoor/outdoor" if prev_io == "indoor" else "outdoor"

    if re.search(r"\bnever happened\b|\bnot sick like this before\b|\bno history\b|\bdid not have\b.*(issue|issues)\b", t):
        slots["history_similar"] = "no"
    if re.search(r"\b(happened before|had this before|similar episode)\b", t):
        slots["history_similar"] = "yes"

    m = re.search(r"\b(\d{1,2})\s*(years|year|yrs|yr)\s*old\b", t)
    if m:
        slots["age_years"] = int(m.group(1))

    if re.search(r"\bcat food\b|\bdry food\b|\bwet food\b|\bkibble\b|\bpuppy food\b|\bdog food\b|\bbland diet\b|\bchicken and rice\b", t):
        slots["diet"] = "mentioned"

    if re.search(r"\bitch(y|ing)?\b|\bscratching\b|\bchewing\b|\brubbing\b", t):
        slots["itching"] = "yes"
    if re.search(r"\bno itch\b|\bnot itchy\b|\bnot scratching\b", t):
        slots["itching"] = "no"

    if re.search(r"\bear(s)?\b", t):
        slots.setdefault("ear_involved", "yes")

    if re.search(r"\b(odor|smell|stink|stinky)\b", t):
        slots["ear_odor"] = "yes"
    if re.search(r"\b(no odor|no smell)\b", t):
        slots["ear_odor"] = "no"

    if re.search(r"\bdischarge\b|\bwax\b|\byellow\b|\bbrown\b|\bblack\b|\bpus\b", t):
        slots["ear_discharge"] = "yes"
    if re.search(r"\bno discharge\b", t):
        slots["ear_discharge"] = "no"

    if re.search(r"\bred spots?\b|\brash\b|\bhives?\b|\bwelts?\b", t):
        slots["skin_lesions"] = "yes"
    if re.search(r"\bspreading\b|\bgetting worse\b|\bmore spots\b|\bexpanding\b", t):
        slots["lesion_spread"] = "yes"
    if re.search(r"\bhair loss\b|\bpatchy hair\b|\bbald\b", t):
        slots["hair_loss"] = "yes"
    if re.search(r"\bflak(y|ing)\b|\bdandruff\b|\bscal(e|y)\b", t):
        slots["flaking"] = "yes"
    if re.search(r"\bpain\b|\bsore\b|\btender\b|\bcries\b", t):
        slots["pain"] = "yes"

    if re.search(r"\bflea(s)?\b", t):
        slots["parasite_flea_possible"] = "yes"
    if re.search(r"\btick(s)?\b", t):
        slots["parasite_tick_possible"] = "yes"
    if re.search(r"\bmite(s)?\b|\bmange\b", t):
        slots["parasite_mite_possible"] = "yes"

    if re.search(r"\b(flea|tick)\b.*\b(prevention|meds|medicine|treatment|pipette|spot[- ]on|collar)\b", t):
        slots["parasite_prevention_mentioned"] = "yes"

    if re.search(r"\bnew\b.*\b(food|treat|shampoo|soap|collar|harness|bedding|detergent|cleaner)\b", t):
        slots["recent_change"] = "yes"

    if re.search(r"\b(other pets|another dog|another cat|multi[- ]pet)\b", t):
        slots["other_pets"] = "yes"

    return slots


def merge_slots(new_slots: Dict):
    if "slots" not in st.session_state:
        st.session_state.slots = {}
    for k, v in (new_slots or {}).items():
        if v is None:
            st.session_state.slots.setdefault(k, None)
        else:
            st.session_state.slots[k] = v


def format_known_answers() -> str:
    s = st.session_state.get("slots", {}) or {}
    items = []
    if s.get("onset"):
        items.append(f"Onset: {s['onset']}")
    if s.get("duration"):
        items.append(f"Duration: {s['duration']}")
    if s.get("age_years"):
        items.append(f"Age: {s['age_years']} years")
    if "vomit_count_24h" in s and s.get("vomit_count_24h") is not None:
        items.append(f"Vomiting in last 24h: {s['vomit_count_24h']}")
    if s.get("diarrhea"):
        items.append(f"Diarrhea: {s['diarrhea']}")
    if s.get("diarrhea_count_24h"):
        items.append(f"Diarrhea episodes in last 24h: {s['diarrhea_count_24h']}")
    if s.get("blood_in_stool"):
        items.append(f"Blood in stool: {s['blood_in_stool']}")
    if s.get("appetite"):
        items.append(f"Appetite: {s['appetite']}")
    if s.get("water_intake"):
        items.append(f"Water intake: {s['water_intake']}")
    if s.get("water_times_today"):
        items.append(f"Water times today: {s['water_times_today']}")
    if s.get("energy"):
        items.append(f"Energy: {s['energy']}")
    if s.get("indoor_outdoor"):
        items.append(f"Indoor or outdoor: {s['indoor_outdoor']}")
    if s.get("history_similar"):
        items.append(f"Similar episode before: {s['history_similar']}")
    if s.get("diet"):
        items.append("Diet: mentioned")
    if s.get("keep_water_down"):
        items.append(f"Can keep water down: {s['keep_water_down']}")
    return "\n".join(items) if items else "None yet"


def missing_questions_from_slots() -> List[str]:
    s = st.session_state.get("slots", {}) or {}
    qs: List[str] = []

    track = st.session_state.get("symptom_track") or "general"

    if not (s.get("onset") or s.get("duration")):
        qs.append("When did this start (today, yesterday, how many hours or days)?")
    if not s.get("appetite"):
        qs.append("How is appetite right now (normal or reduced)?")
    if not s.get("water_intake"):
        qs.append("How is water intake right now (normal, reduced, or some)?")
    if not s.get("energy"):
        qs.append("How is energy right now (normal or reduced)?")

    gi_mentioned = ("vomit_count_24h" in s) or (s.get("diarrhea") in ("yes", "no")) or ("blood_in_stool" in s)
    allow_gi_questions = (track in ("gi", "general")) or gi_mentioned

    if allow_gi_questions and s.get("diarrhea") == "yes" and not s.get("diarrhea_count_24h"):
        qs.append("About how many diarrhea episodes in the last 24 hours?")

    if allow_gi_questions:
        if "vomit_count_24h" in s and s.get("vomit_count_24h") is None:
            qs.append("How many times has your pet vomited in the last 24 hours?")
        if ("vomit_count_24h" not in s) and ("diarrhea" not in s) and (track != "derm"):
            qs.append("Any vomiting or diarrhea in the last 24 hours (how many times)?")
        if ("vomit_count_24h" in s) and (s.get("keep_water_down") is None) and (s.get("vomit_count_24h") not in (0, None)):
            qs.append("Can they keep water down?")

    if not s.get("indoor_outdoor"):
        qs.append("Is your pet usually indoor, outdoor, or both?")

    ctx = (st.session_state.get("last_context", "") or "")
    recent_user_text = " ".join([m for role, m in st.session_state.chat if role == "user"])[-800:]
    skin_mode = (track == "derm") or is_skin_case(ctx) or is_skin_case(recent_user_text)

    if skin_mode:
        if s.get("itching") is None:
            qs.append("How itchy is it (none, mild, moderate, severe)? Is he scratching a lot?")
        if s.get("ear_involved") is None:
            qs.append("Are the red spots only around the ears, or also elsewhere (belly, paws, tail base)?")
        if s.get("ear_odor") is None:
            qs.append("Any ear odor or head shaking?")
        if s.get("ear_discharge") is None:
            qs.append("Any ear discharge or wax buildup (yellow, brown, black)?")
        if s.get("lesion_spread") is None:
            qs.append("Are the spots spreading or getting worse over hours or days?")
        if s.get("hair_loss") is None:
            qs.append("Any hair loss, flaking, or crusting on the skin?")
        if s.get("parasite_prevention_mentioned") is None:
            qs.append("When was the last flea or tick prevention, and what product was it?")
        if s.get("recent_change") is None:
            qs.append("Any recent changes in food, shampoo, collar, bedding, or cleaning products in the last 2 weeks?")
        if s.get("other_pets") is None:
            qs.append("Are there other pets at home, and do they have itching or skin spots too?")

    asked = st.session_state.get("asked_qs", set()) or set()
    qs = [q for q in qs if q not in asked]
    return qs


def rerank_docs(query: str, docs: List[dict], reranker, top_k: int) -> List[dict]:
    if not docs:
        return docs
    pairs = [(query, d["chunk"]) for d in docs]
    scores = reranker.predict(pairs)
    for d, s_ in zip(docs, scores):
        d["rerank_score"] = float(s_)
    docs = sorted(docs, key=lambda x: x.get("rerank_score", -1e9), reverse=True)
    return docs[:top_k]


def search_top_k_hybrid(
    query: str,
    k: int,
    pool: int,
    min_score: float,
    use_reranker: bool,
    rerank_pool: int,
) -> List[dict]:
    pool = int(max(1, pool))
    k = int(max(1, k))
    rerank_pool = int(max(1, rerank_pool))
    rerank_pool = min(rerank_pool, pool)

    track = st.session_state.get("symptom_track") or "general"
    query = f"[track={track}] {query}".strip()

    q = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, pool)

    q_vec = tfidf.transform([query])
    cos = linear_kernel(q_vec, tfidf_mat).ravel()

    cand: Dict[int, float] = {}
    species = (st.session_state.get("pet_species", "") or "").strip()

    for dense_score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        if not _species_ok(urls[idx], species):
            continue
        s_dense = float(max(dense_score, 0.0))
        s_sparse = float(cos[idx])
        score = 0.65 * s_dense + 0.35 * s_sparse
        cand[int(idx)] = max(cand.get(int(idx), -1e9), score)

    if pool < len(cos):
        top_sparse_idx = np.argpartition(-cos, pool)[:pool]
        top_sparse_idx = top_sparse_idx[np.argsort(-cos[top_sparse_idx])]
    else:
        top_sparse_idx = np.argsort(-cos)

    for idx in top_sparse_idx:
        if not _species_ok(urls[idx], species):
            continue
        s_sparse = float(cos[idx])
        s_dense = float((q @ doc_emb[idx].reshape(-1, 1)).ravel()[0])
        s_dense = max(s_dense, 0.0)
        score = 0.65 * s_dense + 0.35 * s_sparse
        cand[int(idx)] = max(cand.get(int(idx), -1e9), score)

    ranked = sorted(cand.items(), key=lambda x: x[1], reverse=True)

    out = []
    target = max(k, rerank_pool) if use_reranker else k
    for idx, score in ranked:
        if score < min_score:
            continue
        out.append({"title": titles[idx], "url": urls[idx], "chunk": chunks[idx], "score": float(score)})
        if len(out) >= target:
            break

    if use_reranker and out:
        rp = min(rerank_pool, len(out))
        out = rerank_docs(query, out[:rp], reranker, top_k=k)

    return out


def build_triage_queue(text: str) -> List[str]:
    merge_slots(extract_slots(text, st.session_state.get("slots", {})))
    qs = missing_questions_from_slots()

    seen = set()
    out = []
    for q in qs:
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out[:3]


def build_generation_prompt(user_ctx: str, retrieved_chunks: List[str]) -> str:
    u = user_display()
    p = pet_display()
    species = (st.session_state.get("pet_species", "") or "").strip().lower() or "unknown"
    track = st.session_state.get("symptom_track") or "general"
    chunks_text = "\n".join([f"• {c}" for c in retrieved_chunks[:8]])

    known = format_known_answers()
    allowed_qs = missing_questions_from_slots()[:2]
    allowed_block = "\n".join([f"• {q}" for q in allowed_qs]) if allowed_qs else "• (Ask no questions)"

    return f"""
You are a calm, warm, friendly veterinary support assistant for pet owners.

Hard rules:
- Do not switch topics away from the Symptom track. Stay consistent.
- Symptom track: {track} (derm=skin/ears, gi=vomiting/diarrhea, general=other/mixed)
- Use the retrieved context plus the user symptoms.
- Do not diagnose.
- Do not list diseases or conditions unless user explicitly asks “what could it be?”
- Never say “Not enough information from the knowledge base.”
- Do not ask for details already present in Known answers.
- If details are missing, ask up to 2 short questions.
- You may only ask questions from the Allowed questions list below.
- Use the exact headings listed. Do not add headings. Do not repeat headings.
- Keep it conversational and kind. Short sentences.
- If blood in stool or vomit is present, recommend contacting a vet today.

Start with:
Hi {u}. Thanks for telling me about {p}. I’m here with you.

Pet profile:
• Species: {species}

Known answers (do not ask again):
{known}

User symptoms:
{user_ctx}

Retrieved context:
{chunks_text}

Allowed questions (ask 0 to 2 only from this list):
{allowed_block}

Write in this structure:
{HEADINGS[0]}
{HEADINGS[1]}
{HEADINGS[2]}
{HEADINGS[3]}

Use numbered lists inside each heading (1., 2., 3.).
End with: This is not medical advice.
""".strip()


def build_followup_prompt(last_ctx: str, retrieved_chunks: List[str], user_question: str) -> str:
    u = user_display()
    p = pet_display()
    species = (st.session_state.get("pet_species", "") or "").strip().lower() or "unknown"
    track = st.session_state.get("symptom_track") or "general"
    chunks_text = "\n".join([f"• {c}" for c in retrieved_chunks[:8]])

    known = format_known_answers()
    allowed_qs = missing_questions_from_slots()[:2]
    allowed_block = "\n".join([f"• {q}" for q in allowed_qs]) if allowed_qs else "• (Ask no questions)"

    return f"""
You are a calm, warm, friendly veterinary support assistant for pet owners.

Rules:
- Do not switch topics away from the Symptom track. Stay consistent.
- Symptom track: {track} (derm=skin/ears, gi=vomiting/diarrhea, general=other/mixed)
- Use the retrieved context plus the known symptoms.
- Do not diagnose.
- Do not list diseases or conditions unless user explicitly asks “what could it be?”
- Do not ask questions that are already answered in Known answers.
- Ask up to 2 short questions only if needed.
- You may only ask questions from the Allowed questions list below.
- Use the exact headings listed. Do not add headings. Do not repeat headings.
- Keep it conversational and kind. Short sentences.
- If blood in stool or vomit is present, recommend contacting a vet today.

Start with:
Hi {u}. About {p}. I’m here with you.

Pet profile:
• Species: {species}

Known answers (do not ask again):
{known}

Known symptoms so far:
{last_ctx}

New user message:
{user_question}

Retrieved context:
{chunks_text}

Allowed questions (ask 0 to 2 only from this list):
{allowed_block}

Write in this structure:
{HEADINGS[0]}
{HEADINGS[1]}
{HEADINGS[2]}
{HEADINGS[3]}

Use numbered lists inside each heading (1., 2., 3.).
End with: This is not medical advice.
""".strip()


# -----------------------------
# Session state
# -----------------------------
def _ensure_state():
    defaults = {
        "chat": [],
        "last_context": "",
        "last_chunks": [],
        "last_docs": [],
        "user_name": "",
        "pet_name": "",
        "pet_species": "",
        "triage_queue": [],
        "triage_context": "",
        "triage_active": False,
        "case_active": False,
        "memory_ack": {"pet_name": ""},
        "slots": {},
        "asked_qs": set(),
        "vet_pending": False,
        "vet_chat_location": "",
        "symptom_track": "",
        "main_symptom": "",
        "intro_shown": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_ensure_state()


def clear_case_state():
    st.session_state.case_active = False
    st.session_state.triage_active = False
    st.session_state.triage_queue = []
    st.session_state.triage_context = ""
    st.session_state.last_context = ""
    st.session_state.last_chunks = []
    st.session_state.last_docs = []
    st.session_state.slots = {}
    st.session_state.asked_qs = set()
    st.session_state.vet_pending = False
    st.session_state.vet_chat_location = ""
    st.session_state.symptom_track = ""
    st.session_state.main_symptom = ""


if (not st.session_state.intro_shown) and (len(st.session_state.chat) == 0):
    st.session_state.intro_shown = True
    intro = (
        "Hi, I’m Pet Symptom Assistant. I can help you with your pet’s symptoms.\n\n"
        "Still, the best option is always getting help from a veterinary professional.\n\n"
        "If you want, I can also list nearby veterinary clinics. Just say “nearest vet”, or share your city, district, and country.\n\n"
        f"When you’re ready, tell me the main symptom you’re seeing in {pet_display()} and when it started."
    )
    st.session_state.chat.append(("assistant", intro))


# -----------------------------
# Chat input + Logic
# -----------------------------
user_msg = st.chat_input("Describe your pet’s symptoms...")

if user_msg:
    st.session_state.chat.append(("user", user_msg))
    mem_changed = update_memory_from_text(user_msg)
    ack = build_ack(mem_changed)

    if mem_changed.get("pet"):
        clear_case_state()

    if re.search(r"(?i)\b(new case|start over|reset|clear case)\b", user_msg):
        clear_case_state()
        assistant_payload = (
            f"Okay {user_display()}. Starting a new case.\n\n"
            "What’s the main symptom right now, and when did it start?"
        )
        st.session_state.chat.append(("assistant", assistant_payload))
        st.rerun()

    if st.session_state.get("vet_pending", False) and looks_like_location_reply(user_msg):
        loc = extract_location_text(user_msg)
        vets_md = build_vets_md(loc, radius_km, limit=5)

        low = vets_md.lower()
        is_failure = ("location lookup failed" in low) or ("vet lookup returned no results" in low) or ("vet lookup failed" in low)

        if is_failure:
            st.session_state.vet_pending = False
            assistant_payload = {
                "answer": (
                    f"I couldn’t fetch clinics right now for {loc} (OSM services may be rate limited).\n\n"
                    "You can try again in a minute, increase the radius, or use the Google Maps link below."
                ),
                "sources_md": "",
                "vets_md": vets_md,
            }
            st.session_state.chat.append(("assistant", assistant_payload))
            st.rerun()

        st.session_state.vet_pending = False
        st.session_state.vet_chat_location = loc
        assistant_payload = {
            "answer": f"Here are nearby veterinary clinics for {loc}:",
            "sources_md": "",
            "vets_md": vets_md,
        }
        st.session_state.chat.append(("assistant", assistant_payload))
        st.rerun()

    merge_slots(extract_slots(user_msg, st.session_state.get("slots", {})))
    merge_slots(extract_slots(st.session_state.get("last_context", ""), st.session_state.get("slots", {})))

    _set_main_symptom_if_empty(user_msg)
    _set_track_from_context(st.session_state.get("main_symptom") or user_msg)

    intent = detect_intent(user_msg)
    alerts = check_rules(user_msg)

    if st.session_state.case_active and (not st.session_state.triage_active) and intent not in ("greeting", "thanks") and (not asks_for_nearby_vets(user_msg)):
        intent = "followup"

    if asks_for_nearby_vets(user_msg):
        if not VET_LOCATOR_AVAILABLE:
            st.session_state.chat.append(("assistant", "Nearby vet lookup is not enabled in this build."))
            st.rerun()

        if not (location_text or "").strip():
            st.session_state.vet_pending = True
            st.session_state.chat.append(
                ("assistant", "Tell me your city + district + country (for example 'Istanbul Kadikoy Turkey') and I will list nearby veterinary clinics.")
            )
            st.rerun()

        vets_md = build_vets_md(location_text, radius_km, limit=5)
        assistant_payload = {"answer": "Here are nearby veterinary clinics:", "sources_md": "", "vets_md": vets_md}
        st.session_state.chat.append(("assistant", assistant_payload))
        st.rerun()

    if st.session_state.triage_active:
        st.session_state.triage_context = (st.session_state.triage_context + " " + user_msg).strip()
        merge_slots(extract_slots(st.session_state.triage_context, st.session_state.get("slots", {})))
        _set_track_from_context(st.session_state.triage_context)

        if len(st.session_state.triage_queue) > 0:
            next_q = st.session_state.triage_queue.pop(0)
            st.session_state.asked_qs.add(next_q)
            assistant_payload = f"Got it, {user_display()}. About {pet_display()}.\n\n{next_q}"
            st.session_state.chat.append(("assistant", assistant_payload))
            st.rerun()

    if intent == "greeting":
        assistant_payload = (
            f"Hi {user_display()}.\n\n"
            + ack +
            f"Tell me {pet_display()}'s main symptom and when it started.\n\n"
            "If you can, add: species (cat or dog), appetite, water intake, energy, and any vomiting or diarrhea.\n\n"
            "To find nearby clinics, you can say: 'nearest vet'."
        )
        st.session_state.chat.append(("assistant", assistant_payload))
        st.rerun()

    if intent == "thanks":
        st.session_state.chat.append(("assistant", f"You are welcome, {user_display()}."))
        st.rerun()

    if intent == "followup":
        if not st.session_state.last_context:
            assistant_payload = (
                f"Hi {user_display()}.\n\n"
                "I can help, but I need the symptom details first.\n\n"
                "Species, duration, appetite, water intake, energy, vomiting or diarrhea, and any breathing changes."
            )
            st.session_state.chat.append(("assistant", assistant_payload))
            st.rerun()

        if has_symptom_signal(user_msg):
            st.session_state.last_context = (st.session_state.last_context + " " + user_msg).strip()

        merge_slots(extract_slots(st.session_state.last_context, st.session_state.get("slots", {})))
        _set_track_from_context(st.session_state.get("main_symptom") or st.session_state.last_context)

        if is_generic_followup(user_msg):
            missing_qs = missing_questions_from_slots()
            if missing_qs:
                assistant_payload = (
                    f"Hi {user_display()}. About {pet_display()}.\n\n"
                    "Quick check:\n"
                    + "\n".join([f"{i+1}. {q}" for i, q in enumerate(missing_qs[:3])])
                )
                st.session_state.chat.append(("assistant", assistant_payload))
                st.rerun()

        docs = search_top_k_hybrid(
            st.session_state.last_context,
            k=top_k,
            pool=pool,
            min_score=min_score,
            use_reranker=use_reranker,
            rerank_pool=rerank_pool,
        )

        retrieved_chunks = [d["chunk"] for d in docs]
        st.session_state.last_chunks = retrieved_chunks
        st.session_state.last_docs = docs

        followup_prompt = build_followup_prompt(
            last_ctx=st.session_state.last_context,
            retrieved_chunks=retrieved_chunks,
            user_question=user_msg,
        )

        u = user_display()
        p = pet_display()

        if use_llm:
            if not retrieved_chunks:
                answer_text = _safe_fallback_answer(u, p, alerts)
                answer_text = clamp_answer_length(answer_text, max_chars=1400)
                answer_text = answer_text + build_vet_offer_line()
            else:
                with st.spinner("Thinking..."):
                    llm_text, _ = generate_answer(
                        followup_prompt,
                        retrieved_chunks[:8],
                        model=model_name,
                        temperature=llm_temp,
                    )
                answer_text = clean_llm_output(llm_text)
                answer_text = filter_to_evidence(answer_text, retrieved_chunks, sim_th=0.22)
                answer_text = clamp_answer_length(answer_text, max_chars=1400)
                answer_text = scrub_unstated_diagnoses(
                    answer_text,
                    (st.session_state.get("last_context", "") + " " + user_msg).strip()
                )
                answer_text = answer_text + build_vet_offer_line()
        else:
            answer_text = f"Hi {u}. About {p}.\n\n" + "\n\n".join(retrieved_chunks[:3]) + build_vet_offer_line()

        srcs = unique_sources(docs, limit=3)
        sources_md = "\n".join([f"• [{s['title']}]({s['url']})" for s in srcs]) if srcs else ""

        vets_md = ""
        if VET_LOCATOR_AVAILABLE and (location_text or "").strip():
            urgent_triggered = is_urgent_alert(alerts)
            if show_vets or (auto_vets_on_alert and urgent_triggered):
                vets_md = build_vets_md(location_text, radius_km, limit=5)

        assistant_payload = {"answer": answer_text, "sources_md": sources_md, "vets_md": vets_md}
        st.session_state.chat.append(("assistant", assistant_payload))
        st.rerun()

    if (not st.session_state.triage_active) and (not st.session_state.case_active):
        if intent == "symptom" and (not has_symptom_signal(user_msg)) and looks_like_profile_only(user_msg):
            assistant_payload = f"Got it, {user_display()}.\n\n" + ack + "What’s the main symptom right now, and when did it start?"
            st.session_state.chat.append(("assistant", assistant_payload))
            st.rerun()

        if intent == "symptom" and (not has_symptom_signal(user_msg)) and (not looks_like_profile_only(user_msg)):
            assistant_payload = f"Got it, {user_display()}.\n\n" + ack + "What’s the main symptom you’re seeing right now, and when did it start?"
            st.session_state.chat.append(("assistant", assistant_payload))
            st.rerun()

    species = (st.session_state.get("pet_species", "") or "").strip().lower()
    base_ctx = get_symptom_context()
    if not base_ctx and has_symptom_signal(user_msg):
        base_ctx = user_msg

    if not base_ctx and (not st.session_state.triage_active) and (not st.session_state.case_active):
        assistant_payload = f"Got it, {user_display()}.\n\n" + ack + "What’s the main symptom right now, and when did it start?"
        st.session_state.chat.append(("assistant", assistant_payload))
        st.rerun()

    context = f"Species: {species}. {base_ctx}".strip() if species else (base_ctx or "").strip()

    if (not st.session_state.triage_active) and (not st.session_state.case_active) and context:
        st.session_state.triage_active = True
        st.session_state.triage_context = context
        merge_slots(extract_slots(context, st.session_state.get("slots", {})))
        _set_main_symptom_if_empty(base_ctx)
        _set_track_from_context(st.session_state.get("main_symptom") or context)
        st.session_state.triage_queue = build_triage_queue(context)

    if st.session_state.triage_active and len(st.session_state.triage_queue) > 0:
        next_q = st.session_state.triage_queue.pop(0)
        st.session_state.asked_qs.add(next_q)
        assistant_payload = f"Got it, {user_display()}. About {pet_display()}.\n\n" + ack + f"{next_q}"
        st.session_state.chat.append(("assistant", assistant_payload))
        st.rerun()

    final_ctx = (st.session_state.triage_context or context).strip()
    merge_slots(extract_slots(final_ctx, st.session_state.get("slots", {})))
    _set_track_from_context(st.session_state.get("main_symptom") or final_ctx)

    docs = search_top_k_hybrid(
        final_ctx,
        k=top_k,
        pool=pool,
        min_score=min_score,
        use_reranker=use_reranker,
        rerank_pool=rerank_pool,
    )

    if show_debug:
        with st.expander("Debug retrieval (scores)"):
            for d in docs:
                st.write({"score": d.get("score"), "rerank": d.get("rerank_score"), "title": d.get("title")})

    retrieved_chunks = [d["chunk"] for d in docs]
    st.session_state.last_context = final_ctx
    st.session_state.last_chunks = retrieved_chunks
    st.session_state.last_docs = docs

    warning = ""
    if show_alerts and alerts:
        warning = "\n".join(alerts) + "\n\n"

    u = user_display()
    p = pet_display()

    if use_llm:
        if not retrieved_chunks:
            llm_text = _safe_fallback_answer(u, p, alerts)
            llm_text = clamp_answer_length(llm_text, max_chars=1400)
        else:
            generation_prompt = build_generation_prompt(user_ctx=final_ctx, retrieved_chunks=retrieved_chunks)
            with st.spinner("Thinking..."):
                llm_text, _ = generate_answer(
                    generation_prompt,
                    retrieved_chunks[:8],
                    model=model_name,
                    temperature=llm_temp,
                )
            llm_text = clean_llm_output(llm_text)
            llm_text = filter_to_evidence(llm_text, retrieved_chunks, sim_th=0.22)
            llm_text = clamp_answer_length(llm_text, max_chars=1400)

        answer_text = warning + llm_text + build_vet_offer_line()
    else:
        answer_text = warning + "\n\n".join(retrieved_chunks[:3]) + build_vet_offer_line()

    srcs = unique_sources(docs, limit=3)
    sources_md = "\n".join([f"• [{s['title']}]({s['url']})" for s in srcs]) if srcs else ""

    vets_md = ""
    if VET_LOCATOR_AVAILABLE and (location_text or "").strip():
        urgent_triggered = is_urgent_alert(alerts)
        if show_vets or (auto_vets_on_alert and urgent_triggered):
            vets_md = build_vets_md(location_text, radius_km, limit=5)

    assistant_payload = {"answer": answer_text, "sources_md": sources_md, "vets_md": vets_md}

    st.session_state.case_active = True
    st.session_state.triage_active = False
    st.session_state.triage_queue = []
    st.session_state.triage_context = final_ctx

    st.session_state.chat.append(("assistant", assistant_payload))
    st.rerun()


# -----------------------------
# Render chat history
# -----------------------------
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        if role != "assistant":
            st.markdown(msg)
        else:
            if isinstance(msg, dict):
                st.markdown(msg.get("answer", ""))

                sources_md = (msg.get("sources_md") or "").strip()
                vets_md = (msg.get("vets_md") or "").strip()

                if sources_md:
                    with st.expander("Sources"):
                        st.markdown(sources_md)

                if vets_md:
                    with st.expander("Nearby veterinary clinics"):
                        st.markdown(vets_md)
            else:
                st.markdown(msg)
