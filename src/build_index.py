# src/build_index.py
# Builds a hybrid retrieval index:
# 1 Dense embeddings + FAISS (cosine via inner product on normalized vectors)
# 2 Sparse TF IDF matrix for lexical matching
#
# Output files are saved under ./models and align by the same chunk order

import os
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


# =====================
# Paths and settings
# =====================

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CANDIDATE_INPUTS = [
    DATA_DIR / "pet_corpus_merged.csv",
    DATA_DIR / "pet_corpus_multi.csv",
    DATA_DIR / "seed_corpus.csv",
    DATA_DIR / "petmd_small.csv",
]

# If your csv is already chunked (your pet_corpus_multi.csv is), set this False
ENABLE_CHUNKING = False
CHUNK_WORDS = 260
OVERLAP_WORDS = 60
MIN_CHARS_PER_CHUNK = 200

# TF IDF settings
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 200000  # reduce if memory is an issue
TFIDF_MAX_DF = 0.98
TFIDF_MIN_DF = 1
TFIDF_SUBLINEAR_TF = True


# =====================
# Helpers
# =====================

def normalize_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()

def pick_input_path() -> Path:
    for p in CANDIDATE_INPUTS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No input csv found. Tried: {[str(p) for p in CANDIDATE_INPUTS]}\n"
        f"Make sure you have at least one under ./data"
    )

def maybe_build_merged_csv() -> Path:
    """
    If pet_corpus_merged.csv already exists, use it.
    Else, if pet_corpus_multi.csv and seed_corpus.csv exist, merge them and save as merged.
    Else, fallback to the first existing candidate.
    """
    merged = DATA_DIR / "pet_corpus_merged.csv"
    multi = DATA_DIR / "pet_corpus_multi.csv"
    seed = DATA_DIR / "seed_corpus.csv"

    if merged.exists():
        return merged

    if multi.exists() and seed.exists():
        df_multi = pd.read_csv(multi).fillna("")
        df_seed = pd.read_csv(seed).fillna("")

        df = pd.concat([df_multi, df_seed], ignore_index=True)

        for col in ["url", "title"]:
            if col not in df.columns:
                df[col] = ""

        if "text" not in df.columns and "body" in df.columns:
            df["text"] = df["body"].astype(str)
        if "text" not in df.columns and "body" not in df.columns:
            raise ValueError("Merged CSV requires either text or body column")

        for col in ["source", "species", "section"]:
            if col not in df.columns:
                df[col] = ""

        df["url"] = df["url"].astype(str).map(normalize_ws)
        df["title"] = df["title"].astype(str).map(normalize_ws)
        df["text"] = df["text"].astype(str).map(normalize_ws)
        df["source"] = df["source"].astype(str).map(normalize_ws)
        df["species"] = df["species"].astype(str).map(normalize_ws)
        df["section"] = df["section"].astype(str).map(normalize_ws)

        df = df[(df["url"] != "") & (df["text"] != "")].copy()
        df["dedup_key"] = (df["url"] + "||" + df["section"] + "||" + df["text"]).astype(str)
        df = df.drop_duplicates("dedup_key").drop(columns=["dedup_key"])

        df.to_csv(merged, index=False, encoding="utf-8")
        print(f"Saved merged corpus: {merged} rows={len(df)}")
        return merged

    return pick_input_path()

def chunk_text(text: str, chunk_words: int, overlap_words: int):
    words = (text or "").split()
    if len(words) <= chunk_words:
        return [" ".join(words)] if words else []

    chunks = []
    start = 0
    step = max(1, chunk_words - overlap_words)

    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) >= MIN_CHARS_PER_CHUNK:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks

def build_full_text(title: str, section: str, text: str) -> str:
    parts = []
    if title:
        parts.append(title)
    if section:
        parts.append(f"Section: {section}")
    prefix = " | ".join(parts).strip()
    full = f"{prefix}. {text}" if prefix else text
    return normalize_ws(full)

def light_clean_for_tfidf(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =====================
# Main
# =====================

if __name__ == "__main__":
    INPUT_CSV = maybe_build_merged_csv()
    print(f"Using input: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV).fillna("")
    df.columns = [c.strip() for c in df.columns]

    if "url" not in df.columns:
        raise ValueError("CSV must contain url column")

    if "title" not in df.columns:
        df["title"] = ""

    if "text" in df.columns:
        TEXT_COL = "text"
    elif "body" in df.columns:
        TEXT_COL = "body"
    else:
        raise ValueError("CSV must contain either text or body column")

    for col in ["source", "species", "section"]:
        if col not in df.columns:
            df[col] = ""

    for col in ["url", "title", TEXT_COL, "source", "species", "section"]:
        df[col] = df[col].astype(str)

    chunk_texts = []
    chunk_titles = []
    chunk_urls = []
    chunk_meta = []

    print("Preparing chunks...")
    for _, row in df.iterrows():
        url = normalize_ws(row["url"])
        title = normalize_ws(row["title"])
        text = normalize_ws(row[TEXT_COL])

        source = normalize_ws(row.get("source", ""))
        species = normalize_ws(row.get("species", ""))
        section = normalize_ws(row.get("section", ""))

        if not url or not text:
            continue

        full_text = build_full_text(title, section, text)

        if ENABLE_CHUNKING:
            chunks = chunk_text(full_text, CHUNK_WORDS, OVERLAP_WORDS)
        else:
            chunks = [full_text] if len(full_text) >= MIN_CHARS_PER_CHUNK else []

        for c in chunks:
            chunk_texts.append(c)
            chunk_titles.append(title)
            chunk_urls.append(url)
            chunk_meta.append(
                {"source": source, "species": species, "section": section}
            )

    print(f"Total chunks: {len(chunk_texts)}")
    if not chunk_texts:
        raise RuntimeError("No chunks produced. Check your csv content and MIN_CHARS_PER_CHUNK")

    # =====================
    # Dense embeddings + FAISS
    # =====================
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Embedding chunks...")
    embeddings = model.encode(
        chunk_texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=64
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(MODELS_DIR / "faiss_index.bin"))
    np.save(str(MODELS_DIR / "doc_embeddings.npy"), embeddings)

    with open(MODELS_DIR / "titles.pkl", "wb") as f:
        pickle.dump(chunk_titles, f)

    with open(MODELS_DIR / "urls.pkl", "wb") as f:
        pickle.dump(chunk_urls, f)

    with open(MODELS_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunk_texts, f)

    with open(MODELS_DIR / "meta.pkl", "wb") as f:
        pickle.dump(chunk_meta, f)

    print(f"Saved FAISS index: {MODELS_DIR / 'faiss_index.bin'}")
    print(f"Index vectors: {index.ntotal} dim={dim}")

    # =====================
    # TF IDF (sparse)
    # =====================
    print("Building TF IDF...")
    tfidf_corpus = [light_clean_for_tfidf(t) for t in chunk_texts]

    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        norm="l2",
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",  # keeps short symptom tokens and numbers
    )

    tfidf_matrix = vectorizer.fit_transform(tfidf_corpus)

    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    save_npz(MODELS_DIR / "tfidf_matrix.npz", tfidf_matrix)

    print(f"Saved TF IDF vectorizer: {MODELS_DIR / 'tfidf_vectorizer.pkl'}")
    print(f"Saved TF IDF matrix: {MODELS_DIR / 'tfidf_matrix.npz'}")
    print(f"TF IDF shape: {tfidf_matrix.shape}")

    print("Done. Artifacts are aligned by the same chunk order across all files.")
