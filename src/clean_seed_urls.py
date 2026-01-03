import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "sources" / "seed_urls.txt"
OUT_PATH = BASE_DIR / "sources" / "seed_urls_clean.txt"

# --- knobs you can tune ---
MAX_VCA_DOG_BREEDS = 80        # cap dog breed pages
MAX_VCA_CAT_BREEDS = 120       # allow more cat pages
MAX_ASPCA_PLANT_PAGES = 200    # keep some plants, but not endless A-Z + page=68 spam
DROP_ASPCA_AZ_INDEX = True     # drop .../plants/a?& etc
DROP_ASPCA_PAGE_GT = 15        # drop plants?page=16+ (huge)
KEEP_ASPCA_FOODS_POISONS = True

def norm_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    # remove fragment
    p = urlparse(u)
    p = p._replace(fragment="")
    # normalize query: remove empty junk like "?&"
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if k and v]
    p = p._replace(query=urlencode(q))
    return urlunparse(p).strip()

def domain(u: str) -> str:
    return urlparse(u).netloc.lower().replace("www.", "")

def path(u: str) -> str:
    return urlparse(u).path.lower()

def is_aspca_plant_az_index(u: str) -> bool:
    # examples: .../toxic-and-non-toxic-plants/a?&  OR /c?&
    return bool(re.search(r"/toxic-and-non-toxic-plants/[a-z]\b", path(u)))

def aspca_page_num(u: str) -> int | None:
    qs = dict(parse_qsl(urlparse(u).query))
    if "page" in qs:
        try:
            return int(qs["page"])
        except:
            return None
    return None

def main():
    if not IN_PATH.exists():
        print(f"[ERR] Missing: {IN_PATH}")
        return

    raw = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            raw.append(norm_url(u))

    # unique preserving order
    seen = set()
    urls = []
    for u in raw:
        if u and u not in seen:
            urls.append(u)
            seen.add(u)

    kept = []
    stats = {
        "in": len(urls),
        "kept": 0,
        "drop_aspca_az": 0,
        "drop_aspca_page": 0,
        "drop_vca_dog_breeds_overcap": 0,
        "kept_vca_dog_breeds": 0,
        "kept_vca_cat_breeds": 0,
        "kept_merck_cat": 0,
        "kept_merck_dog": 0,
        "kept_aspca_plants": 0,
        "kept_aspca_poison_food": 0,
        "other": 0,
    }

    vca_dog_breeds = []
    vca_cat_breeds = []
    aspca_plants = []
    aspca_poison_food = []
    merck_cat = []
    merck_dog = []
    other = []

    for u in urls:
        d = domain(u)
        p = path(u)

        if d == "vcahospitals.com":
            if p.startswith("/know-your-pet/dog-breeds/"):
                vca_dog_breeds.append(u)
                continue
            if p.startswith("/know-your-pet/cat-breeds/"):
                vca_cat_breeds.append(u)
                continue
            other.append(u)
            continue

        if d in ("merckvetmanual.com", "www.merckvetmanual.com"):
            if p.startswith("/cat-owners"):
                merck_cat.append(u)
                continue
            if p.startswith("/dog-owners"):
                merck_dog.append(u)
                continue
            other.append(u)
            continue

        if d == "aspca.org":
            if "/toxic-and-non-toxic-plants" in p:
                if DROP_ASPCA_AZ_INDEX and is_aspca_plant_az_index(u):
                    stats["drop_aspca_az"] += 1
                    continue
                pn = aspca_page_num(u)
                if pn is not None and pn > DROP_ASPCA_PAGE_GT:
                    stats["drop_aspca_page"] += 1
                    continue
                aspca_plants.append(u)
                continue

            if KEEP_ASPCA_FOODS_POISONS and (
                "/animal-poison-control" in p or
                "/aspca-poison-control" in p or
                "people-foods-avoid-feeding-your-pets" in p or
                "poisonous-household-products" in p
            ):
                aspca_poison_food.append(u)
                continue

            other.append(u)
            continue

        other.append(u)

    # apply caps (this is the main “rebalance”)
    kept.extend(vca_cat_breeds[:MAX_VCA_CAT_BREEDS])
    stats["kept_vca_cat_breeds"] = len(vca_cat_breeds[:MAX_VCA_CAT_BREEDS])

    kept.extend(vca_dog_breeds[:MAX_VCA_DOG_BREEDS])
    stats["kept_vca_dog_breeds"] = len(vca_dog_breeds[:MAX_VCA_DOG_BREEDS])
    if len(vca_dog_breeds) > MAX_VCA_DOG_BREEDS:
        stats["drop_vca_dog_breeds_overcap"] = len(vca_dog_breeds) - MAX_VCA_DOG_BREEDS

    kept.extend(merck_cat)
    kept.extend(merck_dog)
    stats["kept_merck_cat"] = len(merck_cat)
    stats["kept_merck_dog"] = len(merck_dog)

    kept.extend(aspca_poison_food)
    stats["kept_aspca_poison_food"] = len(aspca_poison_food)

    kept.extend(aspca_plants[:MAX_ASPCA_PLANT_PAGES])
    stats["kept_aspca_plants"] = len(aspca_plants[:MAX_ASPCA_PLANT_PAGES])

    kept.extend(other)
    stats["other"] = len(other)

    # final unique
    seen2 = set()
    final = []
    for u in kept:
        if u and u not in seen2:
            final.append(u)
            seen2.add(u)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        for u in final:
            f.write(u + "\n")

    stats["kept"] = len(final)

    print(f"Saved cleaned seed -> {OUT_PATH}")
    print("Stats:", stats)

if __name__ == "__main__":
    main()
