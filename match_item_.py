#!/usr/bin/env python3
# match_item.py — strict brand → strict name → qty refinement + suggestions + optional image download
#
# deps:
#   pip install orjson unidecode rapidfuzz numpy requests

import sys, re, os, argparse, orjson
import time
import random
import threading
from unidecode import unidecode
from rapidfuzz import fuzz

# Optional image download
try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False


_OFF_SESSION = None
_OFF_LOCK = threading.Lock()
_OFF_LAST_REQUEST_TS = 0.0


def _off_session():
    global _OFF_SESSION
    if _OFF_SESSION is None:
        _OFF_SESSION = requests.Session()
    return _OFF_SESSION


def _off_throttle(min_interval_s: float):
    """Basic per-process throttle to avoid hammering OpenFoodFacts."""
    global _OFF_LAST_REQUEST_TS
    if min_interval_s <= 0:
        return
    with _OFF_LOCK:
        now = time.monotonic()
        wait_s = (_OFF_LAST_REQUEST_TS + float(min_interval_s)) - now
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.monotonic()
        _OFF_LAST_REQUEST_TS = now


# ---------------- text utils ----------------
def canon(s: str) -> str:
    if not s:
        return ""
    s = unidecode(s).lower()
    s = re.sub(r"[^a-z0-9\.\-\+\sx×]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str):
    return [t for t in canon(s).split() if t]


def contains_any(hay: str, needles):
    H = " " + canon(hay) + " "
    return any((" " + n + " ") in H for n in needles)

# ---------------- brand similarity & gating ----------------
BRAND_STOPWORDS = {
    "the","and","&","co","company","inc","llc","ltd","corp","corporation","brands","brand",
    "foods","food","market","farms","farm","dairy","beverage","kitchen","kitchens","group",
    "organic","natural","naturals","quality","select","selects","choice","choices"
}


def _split_brand_variants(brand: str):
    """Split comma/ampersand/slash separated brand strings into parts."""
    if not brand:
        return []
    parts = re.split(r"[,&/]+", brand)
    out = []
    for p in parts:
        c = canon(p).strip()
        if c:
            out.append(c)
    return out


def brand_strong_tokens(s: str):
    return {t for t in tokenize(s) if t not in BRAND_STOPWORDS}


def brand_similarity(user_brand: str, candidate_brand: str) -> float:
    """Max token-set fuzzy similarity between user brand and any candidate brand variant."""
    if not user_brand:
        return 1.0
    ub = canon(user_brand)
    if not ub:
        return 1.0
    cand_parts = _split_brand_variants(candidate_brand) or [canon(candidate_brand)]
    sims = [fuzz.token_set_ratio(ub, part)/100.0 for part in cand_parts if part]
    return max(sims) if sims else 0.0


def brand_ok(row, user_brand: str, min_sim: float, partial_floor: float = 0.50) -> bool:
    # Strict brand gate (used only when no exact-variant match exists in the catalog)
    if not user_brand.strip():
        return True
    cand_brand = row.get("brand", "")
    sim = brand_similarity(user_brand, cand_brand)
    if sim >= float(min_sim):
        return True
    u_tok = brand_strong_tokens(user_brand)
    c_tok = brand_strong_tokens(cand_brand)
    strong_overlap = len(u_tok & c_tok) >= 1
    return strong_overlap and sim >= float(partial_floor)


def has_exact_brand_variant(row, user_brand: str) -> bool:
    """True if ANY variant in row['brand'] exactly equals the user brand (canonical)."""
    if not user_brand.strip():
        return False
    target = canon(user_brand)
    raw = row.get("brand", "")
    parts = _split_brand_variants(raw) or [canon(raw)]
    for part in parts:
        if part and part == target:
            return True
    return False


def filter_brand(rows, brand, strict_min=0.60, partial_floor=0.50):
    """
    BRAND FILTER with exact-match priority:
      1) If ANY rows contain a brand VARIANT that equals the user brand (canonical),
         keep ONLY those rows.
      2) Else, keep rows that pass the strict brand gate (brand_ok).
    """
    if not brand.strip():
        return rows

    exact = [r for r in rows if has_exact_brand_variant(r, brand)]
    if exact:
        return exact

    kept = [r for r in rows if brand_ok(r, brand, strict_min, partial_floor)]
    return kept


def best_brand_suggestions(rows, brand, k=5, suggest_min=0.45):
    scored = []
    for i, r in enumerate(rows):
        sim = brand_similarity(brand, r.get("brand", ""))
        if sim >= suggest_min:
            scored.append((sim, i))
    if not scored:
        scored = [(brand_similarity(brand, r.get("brand", "")), i) for i, r in enumerate(rows)]
    scored.sort(reverse=True)
    return [rows[i] for s, i in scored[:k]]


# ---------------- quantity parsing & matching ----------------

# Fraction normalization: "1/2" -> "0.5", "3/4" -> "0.75"
_FRACTION_RE = re.compile(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)')


def _norm_qty_string(q: str) -> str:
    """Normalize quantity string but KEEP enough structure for numeric parsing."""
    if not q:
        return ""
    s = unidecode(q).lower()

    # convert fractions like "1/2" to decimal
    def _repl(m):
        num = float(m.group(1))
        den = float(m.group(2))
        if den == 0:
            return "0"
        return f"{num/den:g}"

    s = _FRACTION_RE.sub(_repl, s)
    # normalize separators
    s = s.replace(",", " ")
    return s


# number + unit; unit can contain letters, dots, hyphens, spaces, parentheses
_NUM_UNIT = re.compile(r'(\d+(?:\.\d+)?)[\s]*([a-zA-Z][a-zA-Z\.\-\s\(\)]*)')


def _collapse_unit(u: str) -> str:
    return re.sub(r'[^a-z]', '', (u or '').lower())


UNIT_ALIAS = {
    "floz": "fluid_ounce", "flounce": "fluid_ounce",
    "oz": "ounce",
    "ml": "milliliter", "millilitre": "milliliter", "cl": "milliliter",
    "l": "liter", "lt": "liter",
    "g": "gram", "gr": "gram",
    "kg": "kilogram", "mg": "milligram",
    "lb": "pound", "lbs": "pound",
    "ct": "count", "pack": "count", "pk": "count"
}

TO_BASE = {
    ("ounce", "gram"): 28.3495,
    ("pound", "gram"): 453.592,
    ("kilogram", "gram"): 1000.0,
    ("milligram", "gram"): 1/1000.0,
    ("fluid_ounce", "milliliter"): 29.5735,
    ("liter", "milliliter"): 1000.0
}

BASE_OF = {
    "ounce": "gram", "pound": "gram", "kilogram": "gram", "gram": "gram", "milligram": "gram",
    "fluid_ounce": "milliliter", "liter": "milliliter", "milliliter": "milliliter",
    "count": "count"
}


def _to_base(val, canon_unit):
    base = BASE_OF.get(canon_unit, canon_unit)
    if canon_unit != base:
        fac = TO_BASE.get((canon_unit, base))
        if fac:
            val *= fac
    return val, base


def parse_qty(q: str):
    """
    Parse a quantity string into approximate mass (grams), volume (milliliters),
    and/or count. Handles strings like '1/2 gallon (1.89 L)', '24 336g', '12 ct', etc.
    """
    if not q:
        return {"mass": None, "volume": None, "count": None}

    s = _norm_qty_string(q)
    # second pass through canon to normalize spaces/junk, *after* fraction conversion
    s = canon(s)

    mass, vol, cnt = None, None, None
    pack = None

    # pack-like prefixes: "24x", "24 ct", etc.
    pm = re.search(r'(\d+)\s*(?:x|×|pk|pack|ct)\b', s)
    if pm:
        pack = int(pm.group(1))
    # patterns like "24 336g" where 24 is a leading count
    leading_ct = re.match(r'^\s*(\d{1,3})\s+[a-zA-Z]*\s*\d', s)
    if (not pack) and leading_ct:
        pack = int(leading_ct.group(1))

    for m in _NUM_UNIT.finditer(s):
        val = float(m.group(1))
        unit_raw = _collapse_unit(m.group(2))
        if unit_raw == "cl":
            val *= 10.0  # cl -> ml
        canon_u = UNIT_ALIAS.get(unit_raw, unit_raw)
        base_val, base = _to_base(val, canon_u)
        mult = pack if pack else 1
        if base == "gram":
            mass = max(mass or 0, base_val * mult)
        elif base == "milliliter":
            vol = max(vol or 0, base_val * mult)
        elif base == "count":
            cnt = max(cnt or 0, int(val))

    if cnt is None and pack is not None:
        cnt = pack

    return {"mass": mass, "volume": vol, "count": cnt}


def qty_close(user_q: str, cat_q: str, tol=0.18):
    """
    Coarse filter: accept if ANY comparable dimension is within ±tol relative error.
    Neutral if catalog size is missing (do NOT reject for missing).
    Used for soft filtering, not for strict equality.
    """
    if not user_q.strip():
        return True
    U, C = parse_qty(user_q), parse_qty(cat_q)
    compared = False
    for k in ("mass", "volume", "count"):
        u, c = U[k], C[k]
        if u is None or c is None:
            continue
        compared = True
        rel = abs(u - c) / max(c, u, 1e-6)
        if rel <= tol:
            return True
    return not compared


def qty_numeric_similarity(user_q: str, cat_q: str):
    """
    Strict numeric similarity in [0,1] between user quantity and catalog quantity.
    1.0 means numerically equal (up to floating precision).
    None means no comparable numeric dimension.
    """
    U, C = parse_qty(user_q), parse_qty(cat_q)
    sims = []

    def _sim(u, c):
        if u is None or c is None:
            return None
        rel = abs(u - c) / max(u, c, 1e-6)
        return max(0.0, 1.0 - rel)

    for k in ("mass", "volume", "count"):
        s = _sim(U[k], C[k])
        if s is not None:
            sims.append(s)
    return max(sims) if sims else None


# ---------------- NAME MATCHING ----------------
def best_name_suggestions(rows, name, k=5):
    scored = [(fuzz.token_set_ratio(canon(name), canon(r.get("name", "")))/100.0, i)
              for i, r in enumerate(rows)]
    scored.sort(reverse=True)
    return [rows[i] for s, i in scored[:k]]


def filter_name_strict(rows, user_name: str):
    """
    Strict product name matching:
      1) canonical exact equality
      2) startswith
      3) substring
      4) fuzzy fallback
    """
    if not user_name.strip():
        return rows

    uq = canon(user_name)
    cname_list = [canon(r.get("name", "")) for r in rows]

    # 1) exact canonical equality
    exact = [r for r, cn in zip(rows, cname_list) if cn == uq]
    if len(exact) == 1:
        return exact
    if len(exact) > 1:
        return exact

    # 2) startswith
    starts = [r for r, cn in zip(rows, cname_list) if cn.startswith(uq)]
    if len(starts) == 1:
        return starts
    if len(starts) > 1:
        return starts

    # 3) substring
    sub = [r for r, cn in zip(rows, cname_list) if uq in cn]
    if len(sub) == 1:
        return sub
    if len(sub) > 1:
        return sub

    # 4) fuzzy fallback
    kept = []
    for r in rows:
        sim = fuzz.token_set_ratio(uq, canon(r.get("name", ""))) / 100.0
        if sim >= 0.62:
            kept.append(r)
    return kept


# ---------------- catalog I/O ----------------
def load_catalog(path):
    rows = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            d = orjson.loads(line)
            rows.append({
                "code": d.get("code", ""),
                "name": d.get("product_name", ""),
                "brand": d.get("brands", ""),
                "qty": d.get("quantity", ""),
                "keywords": " ".join(d.get("keywords", []) or [])
            })
    return rows


def print_block(title, rows):
    print(f"\n[{title}]")
    if not rows:
        print("  (none)")
        return
    for i, r in enumerate(rows, 1):
        print(f" {i:>2}. {r['brand']} — {r['name']} ({r['qty']}) [code={r['code']}]")

def _safe_filename(s: str, max_len: int = 120) -> str:
    s = canon(s)
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_\-\.]+", "", s)
    s = s.strip("._-")
    if not s:
        s = "item"
    return s[:max_len]


def download_image(
    row,
    out_dir="images",
    *,
    min_interval_s: float = 0.5,
    max_retries: int = 2,
):
    if not HAVE_REQUESTS:
        print("[image] requests not installed")
        return

    code = row.get("code", "").strip()
    brand = _safe_filename(row.get("brand", ""))
    name = _safe_filename(row.get("name", ""))

    # fallback if brand or name is missing
    filebase = f"{brand}-{name}" if brand or name else code or "unknown_product"

    headers = {
        "User-Agent": "Sixth-Sense-VIP/1.0 (ingredient matcher)",
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        "Referer": "https://world.openfoodfacts.org/",
    }

    url = f"https://world.openfoodfacts.org/api/v0/product/{code}.json"
    try:
        _off_throttle(min_interval_s)
        resp = _off_session().get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            print("[image] OF returned", resp.status_code)
            return
        data = resp.json()
        prod = data.get("product", {})
        img = prod.get("image_front_url") or prod.get("image_url")
        if not img:
            print("[image] no image available")
            return

        last_status = None
        for attempt in range(int(max_retries) + 1):
            if attempt > 0:
                time.sleep(min(6.0, (0.8 * (2 ** (attempt - 1)))) + random.uniform(0, 0.25))
            _off_throttle(min_interval_s)
            imgdata = _off_session().get(img, headers=headers, timeout=8)
            last_status = imgdata.status_code
            if last_status == 200:
                break
            if last_status in (403, 429, 500, 502, 503, 504):
                retry_after = imgdata.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(15.0, float(retry_after)))
                    except Exception:
                        pass
                continue
            break

        if last_status != 200:
            print("[image] img download failed:", last_status)
            return

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{filebase}.jpg")

        with open(out_path, "wb") as f:
            f.write(imgdata.content)

        print("[image] saved to", out_path)

    except Exception as e:
        print("[image] error:", e)


# ---------------- recipe/free-form matching ----------------
_QTY_SUFFIX_RE = re.compile(
    r"(?P<qty>\b\d[\d\s\.\/]*\s*(?:"
    r"fl\s*oz|floz|oz|ounce|ounces|lb|lbs|pound|pounds|g|gram|grams|kg|kilogram|kilograms|"
    r"ml|milliliter|milliliters|l|liter|liters|"
    r"ct|count|pk|pack|packs|x"
    r")\b.*)$",
    re.IGNORECASE,
)


def extract_qty_suffix(label: str):
    """
    Heuristic: extract a trailing quantity substring from a free-form product label.
    Returns (name_without_qty, qty_str_or_empty).
    """
    if not label:
        return "", ""
    s = label.strip()
    m = _QTY_SUFFIX_RE.search(s)
    if not m:
        return s, ""
    qty = (m.group("qty") or "").strip()
    name = s[: m.start()].strip(" ,-/\t")
    return name.strip(), qty


def build_brand_token_index(rows):
    """
    Build a token -> set(brand_variant) index so we can cheaply guess brands
    from a free-form label like "Kerrygold salted butter 8 oz".
    """
    idx = {}
    for r in rows:
        raw = r.get("brand", "") or ""
        parts = _split_brand_variants(raw) or [canon(raw)]
        for part in parts:
            part = canon(part)
            if not part:
                continue
            toks = brand_strong_tokens(part)
            if not toks:
                continue
            for t in toks:
                idx.setdefault(t, set()).add(part)
    return idx


def guess_brand_from_label(label: str, brand_token_index):
    """
    Guess the canonical brand variant contained in label, preferring the
    longest brand variant that appears as a whole-word substring.
    Returns "" if no confident brand hit is found.
    """
    if not label or not brand_token_index:
        return ""
    H = " " + canon(label) + " "
    tokens = brand_strong_tokens(label)
    if not tokens:
        return ""

    candidates = set()
    for t in tokens:
        candidates |= brand_token_index.get(t, set())

    hits = []
    for cand in candidates:
        needle = " " + cand + " "
        if needle in H:
            hits.append(cand)
    if not hits:
        return ""
    hits.sort(key=len, reverse=True)
    return hits[0]


def strip_brand_from_label(label: str, brand_canon: str):
    if not label or not brand_canon:
        return label or ""
    s = canon(label)
    b = canon(brand_canon)
    if not b:
        return label
    # remove the first whole-word occurrence of the brand variant
    s2 = re.sub(rf"(?<!\w){re.escape(b)}(?!\w)", " ", s, count=1)
    return re.sub(r"\s+", " ", s2).strip()


def match_freeform_item(rows, label: str, brand_token_index=None, max_candidates: int = 30):
    """
    Match a free-form label (single string) against a catalog loaded via load_catalog().
    Returns a dict with:
      - status: "single", "multiple", "none"
      - query: original label
      - brand_guess, name_guess, qty_guess
      - best: best row or None
      - candidates: up to max_candidates rows
    """
    brand_token_index = brand_token_index or {}
    name_wo_qty, qty = extract_qty_suffix(label)
    brand_guess = guess_brand_from_label(name_wo_qty, brand_token_index)
    name_guess = strip_brand_from_label(name_wo_qty, brand_guess) if brand_guess else name_wo_qty
    name_guess = name_guess.strip()

    working = rows
    if brand_guess:
        working = filter_brand(working, brand_guess)
        # If we over-filtered due to a bad brand guess, fall back to no brand gate.
        if not working:
            working = rows
            brand_guess = ""
            name_guess = name_wo_qty

    working2 = filter_name_strict(working, name_guess)
    if not working2:
        return {
            "status": "none",
            "query": label,
            "brand_guess": brand_guess,
            "name_guess": name_guess,
            "qty_guess": qty,
            "best": None,
            "candidates": [],
        }
    working = working2

    if qty:
        soft = [r for r in working if qty_close(qty, r.get("qty", ""))]
        if soft:
            working = soft

    # rank & choose best
    uq = canon(name_guess)
    scored = []
    for r in working:
        ns = fuzz.token_set_ratio(uq, canon(r.get("name", ""))) / 100.0 if uq else 0.0
        bs = brand_similarity(brand_guess, r.get("brand", "")) if brand_guess else 0.5
        qs = qty_numeric_similarity(qty, r.get("qty", "")) if qty else None
        qscore = qs if qs is not None else 0.0
        score = (0.70 * ns) + (0.20 * bs) + (0.10 * qscore)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    candidates = [r for _, r in scored[:max_candidates]]
    best = candidates[0] if candidates else None
    status = "single" if len(candidates) == 1 else "multiple"
    return {
        "status": status,
        "query": label,
        "brand_guess": brand_guess,
        "name_guess": name_guess,
        "qty_guess": qty,
        "best": best,
        "candidates": candidates,
    }


# ---------------- OpenFoodFacts search API (top-N suggestions) ----------------
def off_search_top_products(
    query_text: str,
    limit: int = 5,
    debug: bool = False,
    *,
    min_interval_s: float = 0.5,
    max_retries: int = 3,
    timeout_s: float = 10.0,
):
    """
    Query OpenFoodFacts search endpoint for the top products matching query_text.
    Returns a list of rows shaped like load_catalog() output: {code,name,brand,qty,keywords}.
    """
    if not HAVE_REQUESTS:
        if debug:
            print("[OFF] requests not installed (pip install requests)")
        return []
    q = (query_text or "").strip()
    if not q:
        return []

    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": int(limit),
        "search_terms": q,
    }
    headers = {"User-Agent": "Sixth-Sense-VIP/1.0 (ingredient matcher)"}
    retryable = {429, 500, 502, 503, 504}
    last_err = None
    for attempt in range(int(max_retries) + 1):
        if attempt > 0:
            # Exponential backoff + jitter (cap to keep UI responsive).
            backoff = min(8.0, (0.6 * (2 ** (attempt - 1)))) + random.uniform(0, 0.25)
            time.sleep(backoff)
        _off_throttle(min_interval_s)
        try:
            r = _off_session().get(url, params=params, headers=headers, timeout=float(timeout_s))
            if r.status_code in retryable:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(15.0, float(retry_after)))
                    except Exception:
                        pass
                last_err = RuntimeError(f"HTTP {r.status_code} {r.reason}")
                continue
            r.raise_for_status()
            products = r.json().get("products", []) or []
            last_err = None
            break
        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        if debug:
            print(f"[OFF] search failed for {q!r}: {last_err}")
        return []

    out = []
    for p in products:
        code = (p.get("code") or "").strip()
        if not code:
            continue
        out.append(
            {
                "code": code,
                "name": p.get("product_name") or "",
                "brand": p.get("brands") or "",
                "qty": p.get("quantity") or "",
                "keywords": "",
            }
        )
    return out



# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Filtering-only pipeline: brand → name → qty")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--brand", default="")
    ap.add_argument("--product", default="")
    ap.add_argument("--quantity", default="")
    ap.add_argument("--max_show", type=int, default=5)
    # brand strictness knobs
    ap.add_argument("--brand_min", type=float, default=0.60,
                    help="strict brand similarity threshold to keep (used only when no exact brand match exists)")
    ap.add_argument("--brand_partial_floor", type=float, default=0.50,
                    help="lower fuzzy floor when strong-token overlap exists (used only when no exact brand match)")
    ap.add_argument("--brand_suggest_min", type=float, default=0.45,
                    help="min similarity to show as suggestion when no strict matches")
    ap.add_argument("--download_image", action="store_true", default=True,
                    help="download OpenFoodFacts image for the final match")
    args = ap.parse_args()

    rows = load_catalog(args.catalog)

    # 1) BRAND (exact-variant priority; else strict gate; else suggestions)
    working = filter_brand(rows, args.brand,
                           strict_min=args.brand_min,
                           partial_floor=args.brand_partial_floor)

    if not working:
        sugg = best_brand_suggestions(rows, args.brand,
                                      k=args.max_show,
                                      suggest_min=args.brand_suggest_min)
        print_block("brand suggestions (no strict match)", sugg)
        print("\n[status] need brand clarification")
        sys.exit(0)
    #if 1 < len(working) <= args.max_show:
        #print_block("shortlist after brand", working)
        

    # 2) NAME (strict: exact > startswith > substring > fuzzy)
    working2 = filter_name_strict(working, args.product)
    if not working2:
        sugg = best_name_suggestions(working, args.product, k=args.max_show)
        print_block("name suggestions (no strict name match)", sugg)
        print("\n[status] need product-name clarification")
        sys.exit(0)
    working = working2
    #if 1 < len(working) <= args.max_show:
        #print_block("shortlist after name", working)

    # 3) OPTIONAL CLI QUANTITY (soft filter only; coarse narrowing)
    if args.quantity.strip():
        soft = [r for r in working if qty_close(args.quantity, r.get("qty", ""))]
        if soft:
            working = soft

    # 4) If too many remain → interactive quantity refinement
    if len(working) > args.max_show:
        print_block("Options", working[:args.max_show])
        qty_input = input(
            "\nMore than one item found. Enter quantity to refine "
            "(e.g., '1.85 l', '1/2 gallon', '12 ct'), or press ENTER to select manually: "
        ).strip()

        if qty_input:
            # compute numeric similarity for each candidate
            sims = [qty_numeric_similarity(qty_input, r.get("qty", "")) for r in working]
            pairs = [(s, i) for i, s in enumerate(sims) if s is not None]

            if pairs:
                max_sim = max(s for s, i in pairs)
                # STRICT group: nearly best AND almost exact numeric match
                STRICT_THRESH = 0.99
                strict_idx = [i for s, i in pairs
                              if (s >= max_sim - 1e-6) and (s >= STRICT_THRESH)]

                if len(strict_idx) == 1:
                    # single clear numeric match → auto-select
                    working = [working[strict_idx[0]]]
                elif len(strict_idx) > 1:
                    # multiple items share the same numeric size (e.g., several 1.89 L milks)
                    narrowed = [working[i] for i in strict_idx]
                    working = narrowed
                    print("\nQuantity narrowed candidates, but multiple remain.")
                    print_block("Options", working[:args.max_show])

                    choice = input("Choose item # (or press ENTER to cancel and keep all): ").strip()
                    if choice:
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(working):
                                working = [working[idx]]
                        except Exception:
                            print("Please enter a valid number.")

        # If STILL too many candidates → simple manual choice
        if len(working) > args.max_show:
            print("\nStill multiple candidates; please choose an item:")
            print_block("Options", working[:args.max_show])
            choice = input("Choose item # (or press ENTER to keep all): ").strip()
            if choice:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(working):
                        working = [working[idx]]
                except Exception:
                    print("Please enter a valid number.")

    # If 1 < len <= max_show and no quantity refinement happened (or not needed)
    elif 1 < len(working) <= args.max_show and not args.quantity.strip():
        print_block("Options", working)
        choice = input("\nMore than one item found. Choose item # (or press ENTER to keep all): ").strip()
        if choice:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(working):
                    working = [working[idx]]
            except Exception:
                print("Please enter a valid number.")

    # Final
    print_block("FINAL", working[:args.max_show])
    if len(working) == 1:
        print("\n[status] single match")
        if args.download_image:
            download_image(working[0])
        sys.exit(0)
    elif len(working) == 0:
        print("\n[status] no candidates")
        sys.exit(10)
    else:
        print("\n[status] multiple candidates")
        sys.exit(0)


if __name__ == "__main__":
    main()
