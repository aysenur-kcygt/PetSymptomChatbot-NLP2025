# src/vet_locator.py
# Lightweight Vet Locator (OpenStreetMap)
# Geocoding: Nominatim
# Nearby search: Overpass API
#
# Functions expected by app/chat_app.py
# geocode_location(location_text: str) -> tuple[float,float] | None
# find_vets_nearby(lat: float, lon: float, radius_km: int = 25, limit: int = 5) -> list[dict]

from __future__ import annotations

import os
import time
import json
import math
import hashlib
from typing import Optional, Tuple, List, Dict

import requests


# =============================
# Config
# =============================
NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")

# Prefer stable public endpoints first, keep your env override first if provided.
OVERPASS_URLS = [
    os.getenv("OVERPASS_URL", "").strip(),
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_URLS = [u for u in OVERPASS_URLS if u]

# IMPORTANT: Use a real UA identifying your app. Not "fake" — it should identify you/app.
DEFAULT_UA = "PetSymptomChatbot/1.0 (contact: your_email@example.com)"
USER_AGENT = os.getenv("OSM_USER_AGENT", DEFAULT_UA)

CONTACT_EMAIL = os.getenv("OSM_CONTACT_EMAIL", "").strip()

TIMEOUT_SEC = float(os.getenv("VET_LOCATOR_TIMEOUT_SEC", "15"))

CACHE_DIR = os.getenv("VET_LOCATOR_CACHE_DIR", "cache_vet")
CACHE_TTL_SEC = int(os.getenv("VET_LOCATOR_CACHE_TTL_SEC", str(60 * 60 * 24)))

MAX_RETRIES = int(os.getenv("VET_LOCATOR_MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("VET_LOCATOR_BACKOFF_BASE", "1.6"))

NOMINATIM_MIN_INTERVAL_SEC = float(os.getenv("NOMINATIM_MIN_INTERVAL_SEC", "1.0"))

# Debug prints (status codes, endpoints). Enable by setting env VET_LOCATOR_DEBUG=1
DEBUG = os.getenv("VET_LOCATOR_DEBUG", "0") == "1"

_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en",
    }
)

_mem_cache: Dict[str, Tuple[float, dict]] = {}
_last_nominatim_call_ts = 0.0


# =============================
# Cache helpers
# =============================
def _now() -> float:
    return time.time()

def _hash_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _cache_get(key: str) -> Optional[dict]:
    if key in _mem_cache:
        ts, payload = _mem_cache[key]
        if _now() - ts <= CACHE_TTL_SEC:
            return payload
        _mem_cache.pop(key, None)

    if CACHE_DIR:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            path = os.path.join(CACHE_DIR, f"{key}.json")
            if os.path.exists(path):
                stat = os.stat(path)
                if _now() - stat.st_mtime <= CACHE_TTL_SEC:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
        except Exception:
            return None
    return None

def _cache_set(key: str, payload: dict) -> None:
    _mem_cache[key] = (_now(), payload)
    if CACHE_DIR:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            path = os.path.join(CACHE_DIR, f"{key}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            pass


# =============================
# HTTP helpers
# =============================
def _sleep_backoff(attempt: int, extra: float = 0.0) -> None:
    time.sleep((BACKOFF_BASE ** attempt) + 0.4 + extra)

def _safe_json_response(r: requests.Response) -> Optional[dict]:
    try:
        return r.json()
    except Exception:
        return None

def _retry_after_seconds(r: requests.Response) -> Optional[float]:
    # If server provides Retry-After, respect it
    ra = r.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except Exception:
        return None

def _request_json(
    method: str,
    url: str,
    *,
    params=None,
    data=None,
    polite_nominatim: bool = False,
) -> Optional[dict]:
    global _last_nominatim_call_ts

    for attempt in range(MAX_RETRIES):
        try:
            if polite_nominatim:
                wait = NOMINATIM_MIN_INTERVAL_SEC - (_now() - _last_nominatim_call_ts)
                if wait > 0:
                    time.sleep(wait)
                _last_nominatim_call_ts = _now()

            r = _session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                timeout=TIMEOUT_SEC,
            )

            if DEBUG:
                print(f"[vet_locator] {method} {url} status={r.status_code}")

            # Common transient errors
            if r.status_code in (429, 502, 503, 504):
                ra = _retry_after_seconds(r)
                if ra is not None and ra > 0:
                    time.sleep(min(ra, 10.0))
                else:
                    _sleep_backoff(attempt, extra=0.8)
                continue

            r.raise_for_status()
            payload = _safe_json_response(r)
            if payload is None:
                _sleep_backoff(attempt, extra=0.6)
                continue
            return payload

        except Exception:
            _sleep_backoff(attempt)

    return None


# =============================
# Text normalization
# =============================
_MAJOR_TR_CITIES = {
    "istanbul", "ankara", "izmir", "bursa", "antalya", "adana", "konya", "gaziantep",
    "mersin", "kayseri", "eskisehir", "samsun", "trabzon", "kocaeli", "sakarya",
}

def _normalize_tr_chars(t: str) -> str:
    t = t.replace("İ", "I").replace("ı", "i")
    t = t.replace("Türkiye", "Turkey").replace("TÜRKİYE", "Turkey")
    t = t.replace("türkiye", "Turkey").replace("turkiye", "Turkey")
    return t

def _fix_common_typos(t: str) -> str:
    low = t.lower()
    if "istanbuk" in low:
        t = t.replace("Istanbuk", "Istanbul").replace("istanbuk", "Istanbul")
    if "ıstanbul" in low:
        t = t.replace("ıstanbul", "Istanbul")
    return t

def _normalize_location(text: str) -> str:
    t = (text or "").strip()
    t = " ".join(t.split())
    t = _normalize_tr_chars(t)
    t = _fix_common_typos(t)

    if "," not in t:
        parts = t.split()
        if len(parts) == 3 and parts[2].lower() in ("turkey", "turkiye", "türkiye"):
            first = parts[0].lower()
            if first in _MAJOR_TR_CITIES:
                t = f"{parts[1]}, {parts[0]}, {parts[2]}"
            else:
                t = f"{parts[0]}, {parts[1]}, {parts[2]}"
        elif len(parts) >= 3 and parts[-1].lower() in ("turkey", "turkiye", "türkiye"):
            t = ", ".join(parts)

    return t

def _ensure_country_hint(text: str) -> str:
    default_country = os.getenv("VET_LOCATOR_DEFAULT_COUNTRY", "Turkey").strip()
    if not default_country:
        return text

    lower = text.lower()
    if any(tok in lower for tok in ("turkey", "turkiye", "türkiye")):
        return text

    if "," in text and len(text.split(",")) >= 2:
        return text

    return f"{text}, {default_country}"


# =============================
# Public API
# =============================
def geocode_location(location_text: str) -> Optional[Tuple[float, float]]:
    loc = _normalize_location(location_text)
    if not loc:
        return None

    loc = _ensure_country_hint(loc)

    key = _hash_key(f"geocode::{loc}")
    cached = _cache_get(key)
    if cached is not None:
        lat = cached.get("lat")
        lon = cached.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
        return None

    params = {
        "q": loc,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
    }

    if any(tok in loc.lower() for tok in ("turkey", "turkiye", "türkiye")):
        params["countrycodes"] = "tr"

    if CONTACT_EMAIL:
        params["email"] = CONTACT_EMAIL

    # Nominatim is GET with params (not data)
    data = _request_json("GET", NOMINATIM_URL, params=params, polite_nominatim=True)
    if not data or not isinstance(data, list) or len(data) == 0:
        _cache_set(key, {"lat": None, "lon": None})
        return None

    try:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        _cache_set(key, {"lat": lat, "lon": lon})
        return lat, lon
    except Exception:
        _cache_set(key, {"lat": None, "lon": None})
        return None


def find_vets_nearby(lat: float, lon: float, radius_km: int = 25, limit: int = 5) -> List[Dict]:
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return []

    radius_m = int(max(1000, min(radius_km, 200)) * 1000)
    limit = int(max(1, min(limit, 20)))

    key = _hash_key(f"vets::{lat:.6f},{lon:.6f}::{radius_m}::{limit}")
    cached = _cache_get(key)
    if cached is not None:
        return cached.get("items", []) or []

    query = f"""
[out:json][timeout:25];
(
  nwr["amenity"="veterinary"](around:{radius_m},{lat},{lon});
  nwr["healthcare"="veterinary"](around:{radius_m},{lat},{lon});
  nwr["office"="veterinary"](around:{radius_m},{lat},{lon});
  nwr["amenity"="clinic"]["healthcare"="veterinary"](around:{radius_m},{lat},{lon});
);
out center tags;
""".strip()

    payload = None
    used_url = None

    # Try endpoints in order until we get a payload with "elements"
    for url in OVERPASS_URLS:
        p = _request_json("POST", url, data={"data": query})
        if p is not None and isinstance(p, dict) and "elements" in p:
            payload = p
            used_url = url
            break

    if payload is None:
        # Network failure / throttling: do NOT cache empty, so user can retry later
        if DEBUG:
            print("[vet_locator] Overpass failed on all endpoints")
        return []

    if DEBUG and used_url:
        print(f"[vet_locator] Overpass OK via {used_url}")

    elements = payload.get("elements", []) or []
    items: List[Dict] = []

    for el in elements:
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or tags.get("operator") or "Vet clinic"

        if "lat" in el and "lon" in el:
            vlat, vlon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center") or {}
            vlat, vlon = center.get("lat"), center.get("lon")

        if vlat is None or vlon is None:
            continue

        addr = _format_address(tags)
        dist_km = _haversine_km(lat, lon, float(vlat), float(vlon))

        phone = tags.get("phone") or tags.get("contact:phone") or ""
        website = tags.get("website") or tags.get("contact:website") or ""

        items.append(
            {
                "name": name.strip() if isinstance(name, str) else "Vet clinic",
                "address": addr,
                "distance_km": float(dist_km),
                "lat": float(vlat),
                "lon": float(vlon),
                "phone": phone,
                "website": website,
            }
        )

    items.sort(key=lambda x: x.get("distance_km", 1e9))
    items = items[:limit]

    # Important: cache only after we successfully got a response with elements key.
    # This caches "true empty" as well (no nearby vets in area), which is fine.
    _cache_set(key, {"items": items})
    return items


# =============================
# Utils
# =============================
def _format_address(tags: dict) -> str:
    parts = []
    street = tags.get("addr:street", "")
    house = tags.get("addr:housenumber", "")
    city = tags.get("addr:city", "") or tags.get("addr:town", "")
    district = tags.get("addr:district", "") or tags.get("addr:suburb", "")
    postcode = tags.get("addr:postcode", "")

    line1 = " ".join([p for p in [street, house] if p]).strip()
    line2 = ", ".join([p for p in [district, city] if p]).strip()
    line3 = postcode.strip()

    if line1:
        parts.append(line1)
    if line2:
        parts.append(line2)
    if line3:
        parts.append(line3)

    if not parts:
        full = tags.get("addr:full", "") or tags.get("contact:address", "")
        if full:
            return str(full).strip()

    return " · ".join(parts).strip()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2) + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
