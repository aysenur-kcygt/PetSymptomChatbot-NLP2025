import requests
from bs4 import BeautifulSoup
import csv
import time

BASE_URLS = [
    "https://www.petmd.com/cat/conditions",
    "https://www.petmd.com/dog/conditions",
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_article_links(base_url: str):
    r = requests.get(base_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    links = []
    for a in soup.select("a[href^='/cat/conditions/'], a[href^='/dog/conditions/']"):
        href = a.get("href", "")
        if not href:
            continue
        full = "https://www.petmd.com" + href.split("?")[0]
        if full not in links:
            links.append(full)

    return links

def scrape_article(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    h1 = soup.select_one("h1")
    title = h1.get_text(strip=True) if h1 else url

    paragraphs = soup.select("p")
    body = " ".join(p.get_text(" ", strip=True) for p in paragraphs)

    body = " ".join(body.split())
    return {"url": url, "title": title, "body": body}

if __name__ == "__main__":
    all_links = []
    for base in BASE_URLS:
        links = get_article_links(base)
        all_links.extend(links)

    uniq = []
    seen = set()
    for u in all_links:
        if u not in seen:
            uniq.append(u)
            seen.add(u)

    max_articles = 400
    uniq = uniq[:max_articles]

    print(f"Found {len(uniq)} unique article links.")

    rows = []
    for i, link in enumerate(uniq, start=1):
        print(f"{i}/{len(uniq)} Scraping: {link}")
        try:
            row = scrape_article(link)
            if len(row["body"]) >= 400:
                rows.append(row)
            time.sleep(0.8)
        except Exception as e:
            print("Error:", e)
            time.sleep(1.0)

    with open("data/petmd_small.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "title", "body"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {len(rows)} rows to data/petmd_small.csv")
