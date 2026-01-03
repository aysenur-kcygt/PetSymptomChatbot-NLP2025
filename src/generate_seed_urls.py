# generate_seed_urls.py (improved filtering for Merck/VCA/ASPCA)
import os
import re
import time
import random
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (PetSymptomChatbot; contact: you@example.com)"}
TIMEOUT = 30

START_URLS = [
    # VCA info hub (not shop)
    "https://vcahospitals.com/know-your-pet",
    # Merck owner sections (good)
    "https://www.merckvetmanual.com/dog-owners",
    "https://www.merckvetmanual.com/cat-owners",
    # ASPCA poison control info
    "https://www.aspca.org/pet-care/animal-poison-control",
]

ALLOWED_DOMAINS = {
    "vcahospitals.com",
    "www.merckvetmanual.com",
    "merckvetmanual.com",
    "www.aspca.org",
    "aspca.org",
}

# Block obvious non-content / ecommerce / user flows
BLOCK_PATH_PARTS = [
    "/shop/", "/store/", "/category/", "/collections/", "/product/", "/products/",
    "/cart", "/checkout", "/account", "/login", "/register", "/subscribe",
    "/search", "/tag/", "/tags/",
]

# Domain-specific "allow only article-like paths"
def path_allowed_by_domain(url: str) -> bool:
    d = urlparse(url).netloc.lower().replace("www.", "")
    p = urlparse(url).path.lower()

    if d == "vcahospitals.com":
        # only keep informational pages
        return p.startswith("/know-your-pet")

    if d == "merckvetmanual.com":
        # keep only owner sections
        return p.startswith("/dog-owners") or p.startswith("/cat-owners")

    if d == "aspca.org":
        # keep pet-care / poison-control
        return p.startswith("/pet-care")

    return True

MAX_TOTAL = 1200
MAX_DEPTH = 2
SLEEP_SEC = 1.2

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_FILE = BASE_DIR / "sources" / "seed_urls.txt"
SEED_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def norm_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    u, _ = urldefrag(u)  # drop #fragment
    return u.strip()

def is_http(u: str) -> bool:
    return u.startswith("http://") or u.startswith("https://")

def has_weird_path_chars(u: str) -> bool:
    """
    Reject paths that contain characters very unlikely in real article URLs.
    Your example: /n3wbr@nds  -> contains '@' -> reject.
    """
    path = urlparse(u).path
    return bool(re.search(r"[@<>\\\[\]{}|^`]", path))

def allowed(u: str) -> bool:
    if not u:
        return False
    u = norm_url(u)

    # ignore non-web links
    lu = u.lower()
    if lu.startswith(("mailto:", "tel:", "javascript:", "#")):
        return False

    if not is_http(u):
        return False

    d = urlparse(u).netloc.lower()
    if d not in ALLOWED_DOMAINS:
        return False

    if any(part in lu for part in BLOCK_PATH_PARTS):
        return False

    if has_weird_path_chars(u):
        return False

    # avoid file downloads
    if re.search(r"\.(pdf|jpg|jpeg|png|gif|webp|zip|rar|7z|mp4|mp3)$", lu):
        return False

    # domain-specific allow rules
    if not path_allowed_by_domain(u):
        return False

    return True

def fetch_html(url: str) -> str:
    """
    Fetch HTML with 429 backoff. For 404/403/other errors, just skip.
    """
    backoff = 5
    for attempt in range(5):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

            if r.status_code == 429:
                wait = backoff + random.uniform(0, 2)
                print(f"[429] {url} -> sleep {wait:.1f}s (attempt {attempt+1}/5)")
                time.sleep(wait)
                backoff *= 2
                continue

            if r.status_code == 404:
                # common with malformed links; just skip
                return ""

            r.raise_for_status()

            ct = (r.headers.get("Content-Type") or "").lower()
            if "text/html" not in ct:
                return ""
            return r.text

        except Exception as e:
            # skip any errors; do not crash run
            return ""

    return ""

def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        # skip weird href types early
        if href.lower().startswith(("mailto:", "tel:", "javascript:")):
            continue
        absu = urljoin(base_url, href)
        absu = norm_url(absu)
        if absu:
            out.append(absu)
    return out

def load_existing_seed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u and not u.startswith("#"):
                s.add(norm_url(u))
    return s

def save_seed(path: Path, urls: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for u in urls:
            f.write(u + "\n")

# ---------------- main ----------------
def main():
    existing = load_existing_seed(SEED_FILE)
    print(f"Start URLs queued: {len(START_URLS)}")
    print(f"Already in seed file: {len(existing)}")
    print(f"Max total: {MAX_TOTAL} | Max depth: {MAX_DEPTH}")

    q = deque()
    seen = set(existing)
    collected = list(existing)

    for u in START_URLS:
        u = norm_url(u)
        if allowed(u) and u not in seen:
            q.append((u, 0))
            seen.add(u)

    while q and len(collected) < MAX_TOTAL:
        url, depth = q.popleft()

        # keep it
        if url not in collected:
            collected.append(url)

        if depth >= MAX_DEPTH:
            continue

        html = fetch_html(url)
        if not html:
            continue

        links = extract_links(html, url)
        time.sleep(SLEEP_SEC)

        for link in links:
            if len(collected) >= MAX_TOTAL:
                break
            if not allowed(link):
                continue
            if link in seen:
                continue
            seen.add(link)
            q.append((link, depth + 1))

    # final unique list
    uniq_out = []
    s = set()
    for u in collected:
        u = norm_url(u)
        if u and u not in s and allowed(u):
            uniq_out.append(u)
            s.add(u)

    uniq_out = uniq_out[:MAX_TOTAL]
    save_seed(SEED_FILE, uniq_out)
    print(f"Saved {len(uniq_out)} seed urls -> {SEED_FILE}")

if __name__ == "__main__":
    main()
