# src/build_embeddings.py
import csv
import json
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer

def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    corpus_path = BASE_DIR / "data" / "pet_corpus_merged.csv"
    store_path = BASE_DIR / "data" / "doc_store.jsonl"
    emb_path = BASE_DIR / "data" / "doc_embeddings.npy"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    docs = []
    texts = []

    with open(corpus_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            text = norm(row.get("text"))
            if len(text) < 80:
                continue
            doc = {
                "url": norm(row.get("url")),
                "title": norm(row.get("title")),
                "source": norm(row.get("source")),
                "species": norm(row.get("species")),
                "section": norm(row.get("section")),
                "text": text,
            }
            docs.append(doc)
            texts.append(text)

    print(f"Docs kept for embedding: {len(docs)}")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    store_path.parent.mkdir(parents=True, exist_ok=True)

    with open(store_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    np.save(emb_path, embeddings)

    print(f"Saved store -> {store_path}")
    print(f"Saved embeddings -> {emb_path} shape={embeddings.shape}")
    print(f"Model -> {model_name}")

if __name__ == "__main__":
    main()
