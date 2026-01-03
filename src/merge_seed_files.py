# src/merge_seed_files.py
from pathlib import Path
from urllib.parse import urldefrag

def norm_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if u.startswith("www."):
        u = "https://" + u
    u, _ = urldefrag(u)  # remove #fragment
    return u.strip()

def read_urls(path: Path):
    if not path.exists():
        print(f"[WARN] Not found: {path}")
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            u = norm_url(u)
            if u:
                out.append(u)
    return out

def uniq(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    p1 = BASE_DIR / "sources" / "seed_urls.txt"
    p2 = BASE_DIR / "sources" / "seed_urls_clean.txt"
    out = BASE_DIR / "sources" / "seed_urls_merged.txt"

    urls = uniq(read_urls(p1) + read_urls(p2))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    print(f"Saved merged seed -> {out} | count={len(urls)}")

if __name__ == "__main__":
    main()
