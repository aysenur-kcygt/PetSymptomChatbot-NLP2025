import csv
import time
import re
import random
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

HEADERS = {"User-Agent": "Mozilla/5.0 (PetSymptomChatbot; contact: you@example.com)"}
TIMEOUT = 30

PETMD_BASES = [
    "https://www.petmd.com/cat/conditions",
    "https://www.petmd.com/dog/conditions",
]

# -----------------------------
# LIMITS (ARTTIRILDI ✅)
# -----------------------------
MAX_PETMD_URLS = 1200        # ✅ PetMD toplam URL (ör: 600 dog + 600 cat hedef)
MAX_SEED_URLS = 1000         # ✅ seed_urls.txt’ten en fazla kaç URL
MAX_TOTAL_URLS = 2000        # ✅ toplam scrape edilecek URL limiti (PetMD + seed)
SHUFFLE_URLS = True
SLEEP_SEC = 0.9

# PetMD içinde hedef dog oranı (0.5 -> eşit)
TARGET_DOG_RATIO = 0.5

# Seed için de denge istersen:
TARGET_SEED_DOG_RATIO = 0.5
USE_SEED_BALANCING = False   # True yaparsan seed’i de dog/cat dengeler

TRIAGE_KEYWORDS = [
    "when to call the vet",
    "when to see a vet",
    "when to see the vet",
    "when to contact your veterinarian",
    "call your veterinarian",
    "seek veterinary care",
    "emergency",
    "warning signs",
    "when to go to the vet",
    "go to an emergency vet",
    "take your pet to the vet",
]

def norm(s: str) -> str:
    return " ".join((s or "").split()).strip()

def detect_species_from_url(url: str) -> str:
    u = (url or "").lower()
    if "/dog/" in u:
        return "dog"
    if "/cat/" in u:
        return "cat"
    return "unknown"

def source_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "petmd.com" in host:
        return "petmd"
    if "vcahospitals.com" in host:
        return "vca"
    return host.replace("www.", "")

def safe_get(url: str):
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r

def uniq_list(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

# -----------------------------
# ✅ Project root finder (fallback’li)
# -----------------------------
def find_project_root(start: Path) -> Path:
    """
    Yukarı doğru çıkarak proje root’unu bulmaya çalışır.
    sources/ veya data/ klasörü bulursa root kabul eder.
    """
    p = start
    for _ in range(10):
        if (p / "sources").exists() or (p / "data").exists():
            return p
        p = p.parent
    return start

def petmd_get_article_links(base_url: str):
    """
    ✅ Base'e göre sadece ilgili türü çekiyoruz:
    - cat base -> /cat/conditions/
    - dog base -> /dog/conditions/
    """
    r = safe_get(base_url)
    soup = BeautifulSoup(r.text, "lxml")

    if "/cat/" in base_url:
        selector = "a[href^='/cat/conditions/']"
        prefix = "https://www.petmd.com"
    else:
        selector = "a[href^='/dog/conditions/']"
        prefix = "https://www.petmd.com"

    links = []
    for a in soup.select(selector):
        href = a.get("href", "")
        if not href:
            continue
        full = (prefix + href.split("?")[0]).strip()
        links.append(full)

    return uniq_list(links)

def balanced_urls(cat_urls, dog_urls, max_urls: int, target_dog_ratio: float = 0.5, label: str = "POOL"):
    """
    ✅ max_urls içinden mümkün olduğunca dengeli seçim yapar.
    """
    max_urls = int(max_urls)
    cat_urls = list(cat_urls)
    dog_urls = list(dog_urls)

    random.shuffle(cat_urls)
    random.shuffle(dog_urls)

    dog_target = int(round(max_urls * float(target_dog_ratio)))
    dog_target = max(0, min(dog_target, max_urls))
    cat_target = max_urls - dog_target

    chosen = []
    chosen.extend(dog_urls[:dog_target])
    chosen.extend(cat_urls[:cat_target])

    # Eğer bir taraf yetmediyse, diğer taraftan tamamla
    if len(chosen) < max_urls:
        remaining = max_urls - len(chosen)
        extra_pool = dog_urls[dog_target:] + cat_urls[cat_target:]
        chosen.extend(extra_pool[:remaining])

    chosen = uniq_list(chosen)[:max_urls]
    random.shuffle(chosen)

    print(
        f"[{label}] pools -> dog={len(dog_urls)} cat={len(cat_urls)} | "
        f"selected={len(chosen)} (dog_target={dog_target}, cat_target={cat_target})"
    )
    print(
        f"[{label}] selected split -> dog={sum('/dog/' in u for u in chosen)} "
        f"cat={sum('/cat/' in u for u in chosen)}"
    )
    return chosen

def extract_main_text_blocks(soup: BeautifulSoup):
    ps = soup.select("p")
    lis = soup.select("li")
    text = " ".join(
        [p.get_text(" ", strip=True) for p in ps] +
        [li.get_text(" ", strip=True) for li in lis]
    )
    return norm(text)

def split_into_sections(title: str, body: str):
    chunks = []

    if len(body) >= 200:
        chunks.append(("general", f"{title}. {body}"))

    sentences = re.split(r"(?<=[.!?])\s+", body)
    triage_sents = []
    for s in sentences:
        ls = s.lower()
        if any(k in ls for k in TRIAGE_KEYWORDS):
            triage_sents.append(s)

    triage_text = norm(" ".join(triage_sents))
    if len(triage_text) >= 120:
        chunks.append(("when_to_see_vet", f"{title}. {triage_text}"))

    return chunks

def scrape_url(url: str):
    r = safe_get(url)
    soup = BeautifulSoup(r.text, "lxml")

    h1 = soup.select_one("h1")
    title = norm(h1.get_text(" ", strip=True) if h1 else url)

    body = extract_main_text_blocks(soup)
    if len(body) < 200:
        return []

    src = source_from_url(url)
    sp = detect_species_from_url(url)

    out_rows = []
    for section, text in split_into_sections(title, body):
        out_rows.append({
            "url": url,
            "title": title,
            "source": src,
            "species": sp,
            "section": section,
            "text": text
        })
    return out_rows

def load_seed_urls(path: str):
    """
    ✅ seed’i sağlamlaştırdık:
    - path yoksa skip
    - www. ise https:// ekle
    - http(s) değilse skip
    - kaç tane okuduğunu yaz
    """
    if not os.path.exists(path):
        print(f"[WARN] Seed file not found: {path} -> skipping seed urls")
        return []

    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            if u.startswith("www."):
                u = "https://" + u
            if not (u.startswith("http://") or u.startswith("https://")):
                continue
            urls.append(u)

    urls = uniq_list(urls)
    print(f"[SEED] Loaded {len(urls)} urls from: {path}")
    return urls

def split_seed_by_species(seed_urls):
    dog, cat, other = [], [], []
    for u in seed_urls:
        sp = detect_species_from_url(u)
        if sp == "dog":
            dog.append(u)
        elif sp == "cat":
            cat.append(u)
        else:
            other.append(u)
    return dog, cat, other

if __name__ == "__main__":
    # ✅ root bulma: script nereden çalıştırılırsa çalıştırılsın
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = find_project_root(SCRIPT_DIR)

    seed_path = BASE_DIR / "sources" / "seed_urls.txt"
    out_path = BASE_DIR / "data" / "pet_corpus_multi.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ DEBUG (seed not found sebebini net gör)
    print("[DEBUG] __file__ =", Path(__file__).resolve())
    print("[DEBUG] SCRIPT_DIR =", SCRIPT_DIR)
    print("[DEBUG] BASE_DIR =", BASE_DIR)
    print("[DEBUG] seed_path =", seed_path)
    print("[DEBUG] seed exists? =", seed_path.exists())
    print("[DEBUG] cwd =", Path.cwd())

    # 1) PetMD linkleri: cat ve dog ayrı
    petmd_cat_pool = petmd_get_article_links("https://www.petmd.com/cat/conditions")
    petmd_dog_pool = petmd_get_article_links("https://www.petmd.com/dog/conditions")

    # 2) PetMD’den dengeli seçim (ARTTIRILMIŞ)
    if MAX_PETMD_URLS is not None:
        petmd_urls = balanced_urls(
            cat_urls=petmd_cat_pool,
            dog_urls=petmd_dog_pool,
            max_urls=MAX_PETMD_URLS,
            target_dog_ratio=TARGET_DOG_RATIO,
            label="PETMD"
        )
    else:
        petmd_urls = uniq_list(petmd_cat_pool + petmd_dog_pool)

    # 3) seed urls
    seed_urls = load_seed_urls(str(seed_path))
    if MAX_SEED_URLS is not None:
        seed_urls = seed_urls[:MAX_SEED_URLS]

    # Seed balancing istersen
    if USE_SEED_BALANCING and seed_urls:
        seed_dog, seed_cat, seed_other = split_seed_by_species(seed_urls)
        seed_balanced_main = balanced_urls(
            cat_urls=seed_cat,
            dog_urls=seed_dog,
            max_urls=min(len(seed_urls), MAX_SEED_URLS),
            target_dog_ratio=TARGET_SEED_DOG_RATIO,
            label="SEED"
        )
        seed_urls = uniq_list(seed_balanced_main + seed_other)

    all_urls = uniq_list(petmd_urls + seed_urls)

    if SHUFFLE_URLS:
        random.shuffle(all_urls)

    if MAX_TOTAL_URLS is not None:
        all_urls = all_urls[:MAX_TOTAL_URLS]

    print(
        f"Total URLs to scrape (after limits): {len(all_urls)} "
        f"| dog={sum('/dog/' in u for u in all_urls)} "
        f"cat={sum('/cat/' in u for u in all_urls)} "
        f"| seed={len(seed_urls)}"
    )

    rows = []
    stats = {"petmd": 0, "seed": 0, "dog": 0, "cat": 0, "triage_chunks": 0, "general_chunks": 0}

    for i, url in enumerate(all_urls, start=1):
        print(f"{i}/{len(all_urls)} {url}")
        try:
            new_rows = scrape_url(url)
            rows.extend(new_rows)

            src = source_from_url(url)
            stats["petmd" if src == "petmd" else "seed"] += 1

            for rr in new_rows:
                if rr["species"] == "dog":
                    stats["dog"] += 1
                elif rr["species"] == "cat":
                    stats["cat"] += 1
                if rr["section"] == "when_to_see_vet":
                    stats["triage_chunks"] += 1
                if rr["section"] == "general":
                    stats["general_chunks"] += 1

            time.sleep(SLEEP_SEC)

        except Exception as e:
            print("Error:", e)
            time.sleep(1.2)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "title", "source", "species", "section", "text"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {len(rows)} chunks -> {out_path}")
    print("Stats:", stats)
  