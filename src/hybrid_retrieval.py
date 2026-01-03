# src/retrieval_hybrid.py
# Hybrid retrieval: Dense (FAISS) + Sparse (TF-IDF)
# Returns chunks aligned with your stored artifacts in ./models

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import pickle

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer


MODELS_DIR = Path("models")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_TOP_K = 6
DEFAULT_K_DENSE = 60
DEFAULT_K_SPARSE = 120

DEFAULT_ALPHA = 0.65  # 0.0 => only tf-idf, 1.0 => only dense
SPECIES_BOOST = 0.12  # small bonus to same species docs


def _norm_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _infer_species(text: str) -> Optional[str]:
    t = (text or "").lower()
    if re.search(r"\b(cat|kitten)\b", t):
        return "cat"
    if re.search(r"\b(dog|puppy)\b", t):
        return "dog"
    m = re.search(r"\bspecies\s*:\s*(cat|dog)\b", t)
    if m:
        return m.group(1)
    return None


def _clean_query_for_tfidf(q: str) -> str:
    q = (q or "").lower()
    q = q.replace("\u00a0", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


@dataclass
class HybridArtifacts:
    embedder: SentenceTransformer
    faiss_index: faiss.Index
    chunks: List[str]
    titles: List[str]
    urls: List[str]
    meta: List[Dict[str, str]]
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: "scipy.sparse.spmatrix"  # noqa


_ARTIFACTS: Optional[HybridArtifacts] = None


def load_artifacts() -> HybridArtifacts:
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS

    faiss_path = MODELS_DIR / "faiss_index.bin"
    chunks_path = MODELS_DIR / "chunks.pkl"
    titles_path = MODELS_DIR / "titles.pkl"
    urls_path = MODELS_DIR / "urls.pkl"
    meta_path = MODELS_DIR / "meta.pkl"

    tfidf_vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    tfidf_mat_path = MODELS_DIR / "tfidf_matrix.npz"

    missing = [p for p in [faiss_path, chunks_path, titles_path, urls_path, meta_path, tfidf_vec_path, tfidf_mat_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts:\n" + "\n".join(f"  {p}" for p in missing)
        )

    embedder = SentenceTransformer(EMBED_MODEL)
    faiss_index = faiss.read_index(str(faiss_path))

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    with open(titles_path, "rb") as f:
        titles = pickle.load(f)
    with open(urls_path, "rb") as f:
        urls = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    with open(tfidf_vec_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    tfidf_matrix = load_npz(str(tfidf_mat_path))

    if not (len(chunks) == len(titles) == len(urls) == len(meta) == tfidf_matrix.shape[0]):
        raise ValueError(
            "Artifacts are misaligned. Ensure you rebuilt index after changing corpus.\n"
            f"chunks={len(chunks)} titles={len(titles)} urls={len(urls)} meta={len(meta)} tfidf_rows={tfidf_matrix.shape[0]}"
        )

    _ARTIFACTS = HybridArtifacts(
        embedder=embedder,
        faiss_index=faiss_index,
        chunks=chunks,
        titles=titles,
        urls=urls,
        meta=meta,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
    )
    return _ARTIFACTS


def _topk_sparse(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.size:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def retrieve_hybrid(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    k_dense: int = DEFAULT_K_DENSE,
    k_sparse: int = DEFAULT_K_SPARSE,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[List[str], List[Dict]]:
    """
    Returns:
      chunks: list[str] length=top_k
      debug: list[dict] per returned chunk (scores + meta)
    """
    art = load_artifacts()

    query_clean = _norm_ws(query)
    if not query_clean:
        return [], []

    wanted_species = _infer_species(query_clean)

    # Dense
    q_emb = art.embedder.encode([query_clean], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    dense_scores, dense_ids = art.faiss_index.search(q_emb, k_dense)
    dense_scores = dense_scores[0]
    dense_ids = dense_ids[0]

    dense_map: Dict[int, float] = {int(i): float(s) for i, s in zip(dense_ids, dense_scores) if int(i) >= 0}

    # Sparse TF-IDF
    q_tfidf = art.tfidf_vectorizer.transform([_clean_query_for_tfidf(query_clean)])  # shape (1, V)
    sparse_scores = (art.tfidf_matrix @ q_tfidf.T).toarray().ravel()  # cosine because both l2 normalized
    sparse_ids = _topk_sparse(sparse_scores, k_sparse)
    sparse_map: Dict[int, float] = {int(i): float(sparse_scores[int(i)]) for i in sparse_ids}

    # Union candidates
    cand_ids = list(set(dense_map.keys()) | set(sparse_map.keys()))
    if not cand_ids:
        return [], []

    # Normalize scores into 0..1 within candidates
    dvals = np.array([dense_map.get(i, 0.0) for i in cand_ids], dtype=np.float32)
    svals = np.array([sparse_map.get(i, 0.0) for i in cand_ids], dtype=np.float32)

    def minmax(x: np.ndarray) -> np.ndarray:
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-8:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    dnorm = minmax(dvals)
    snorm = minmax(svals)

    fused = alpha * dnorm + (1.0 - alpha) * snorm

    # Species boost
    if wanted_species in ("cat", "dog"):
        for j, doc_id in enumerate(cand_ids):
            sp = (art.meta[doc_id].get("species", "") or "").lower().strip()
            if sp == wanted_species:
                fused[j] += SPECIES_BOOST

    order = np.argsort(-fused)[:top_k]

    out_chunks: List[str] = []
    out_debug: List[Dict] = []

    for j in order:
        doc_id = cand_ids[int(j)]
        out_chunks.append(art.chunks[doc_id])

        out_debug.append({
            "doc_id": doc_id,
            "score_fused": float(fused[int(j)]),
            "score_dense_raw": float(dense_map.get(doc_id, 0.0)),
            "score_sparse_raw": float(sparse_map.get(doc_id, 0.0)),
            "title": art.titles[doc_id],
            "url": art.urls[doc_id],
            "meta": art.meta[doc_id],
        })

    return out_chunks, out_debug
