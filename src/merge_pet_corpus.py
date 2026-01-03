# src/merge_corpora.py
import csv
from pathlib import Path

KEY_FIELDS = ["url", "section"]  # duplicate kontrolü için yeterli

def read_csv(path: Path):
    rows = []
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    pet_path = BASE_DIR / "data" / "pet_corpus_multi.csv"
    seed_path = BASE_DIR / "data" / "seed_corpus.csv"
    out_path = BASE_DIR / "data" / "pet_corpus_merged.csv"

    pet_rows = read_csv(pet_path)
    seed_rows = read_csv(seed_path)

    all_rows = pet_rows + seed_rows
    print(f"[IN] pet={len(pet_rows)} seed={len(seed_rows)} total={len(all_rows)}")

    seen = set()
    merged = []
    for row in all_rows:
        url = norm(row.get("url"))
        section = norm(row.get("section"))
        text = norm(row.get("text"))
        if not url or not text:
            continue
        key = (url, section)
        if key in seen:
            continue
        seen.add(key)
        row["url"] = url
        row["section"] = section
        row["text"] = text
        row["title"] = norm(row.get("title"))
        row["source"] = norm(row.get("source"))
        row["species"] = norm(row.get("species"))
        merged.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["url", "title", "source", "species", "section", "text"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged)

    dog = sum(1 for r in merged if r.get("species") == "dog")
    cat = sum(1 for r in merged if r.get("species") == "cat")
    triage = sum(1 for r in merged if r.get("section") == "when_to_see_vet")
    print(f"[OUT] merged={len(merged)} dog_chunks={dog} cat_chunks={cat} triage_chunks={triage}")
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()
